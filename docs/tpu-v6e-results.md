# TPU v6e results (experimental)

This is an experimental port of Nanochat D24 training steps to TPU v6e with PyTorch/XLA. The upstream `nanochat/` source is unchanged; the port lives in [`tpu_smoke/`](../tpu_smoke/README.md) and patches it from outside. These numbers are separate from the [eight-H20 results](h20-results.md) and cannot be compared with them (see [Comparability](#comparability-with-the-h20-results)).

## Summary

On 64 TPU v6e chips with FSDP over all 64, **Pallas splash attention makes a D24 training step 1.26× faster than nanochat's SDPA fallback attention: 252.2 → 200.3 ms per step**. Both arms use per-chip batch 2 (global batch 128 × 2,048 = 262,144 tokens per step), BF16 compute and synthetic data, timed baseline-candidate-baseline. The candidate is numerically tier N1 (not bitwise). It passes a 120-step paired-seed statistical gate, and the same gate catches a learning-rate ×1.02 control.

## Hardware and software

- One TPU v6e slice: 16 hosts × 4 chips = 64 chips, one process per host, SPMD.
- torch 2.9.0, torch_xla 2.9.0, libtpu 0.0.21. Splash attention also needs jax 0.7.1 and torchax (torch_xla's Pallas extra). Both timing arms ran in that environment.
- Mosaic IR version is pinned to 7 in `tpu_smoke/nanochat_splash.py`, because jax 0.7.1 otherwise serializes version 8 and libtpu 0.0.21 rejects it.

## Workload

- Model: upstream `GPT` built with the recipe's shape rule: depth 24, aspect 64, head dim 128. That gives width 1,536, 12 heads, `SSSL` window pattern (short window 768, long window 2,048) and vocabulary 32,768, for 1,384,122,122 parameters. Sequence length 2,048.
- Precision: `NANOCHAT_DTYPE=bfloat16` (nanochat's own precision switch). No FP8.
- Sharding: every parameter is sharded on its first dimension divisible by 64 (`data 1 × fsdp 64` mesh). The batch is sharded over the mesh, and optimizer state follows the parameters.
- Step: forward, backward and one `MuonAdamW` update on one 262,144-token batch, with no gradient accumulation.
- Data: synthetic. Timing runs reuse one fixed batch of uniformly random tokens. Gate runs draw a fresh batch every step from a fixed sparse Markov chain.

### Changes required to run on XLA (baseline and candidate)

- **Optimizer hyperparameters.** Upstream keeps them in 0-D CPU tensors, and XLA rejects CPU tensors in device math. `xla_optim_patch.py` passes Python numbers instead. On CPU it matches the upstream fused AdamW and Muon steps to a maximum absolute difference of 3.6e-7 (FP32 rounding). Removing the cautious weight-decay mask raises the difference to 7.2e-3, and the test catches it.
- **Step boundary.** An explicit `torch_xla.sync()` ends each step. Without it, the lazy graph carries pending optimizer updates into the next step, and step time grows with the step count.

### Candidate: splash attention

Off CUDA, nanochat uses its SDPA fallback, which builds a dense `[T, T]` mask for sliding-window layers and computes the full score matrix. At D24 and sequence length 2,048, dense attention costs about 302 MFLOP per token, while the windows and causality need about 94. `nanochat_splash.py` provides nanochat's `flash_attn_func` interface on top of torch_xla's Pallas splash attention, uses a local mask for sliding windows and uses 512 blocks. `--splash` installs it as a module-attribute replacement.

## Timing method

The method is baseline-candidate-baseline. Each arm is a separate launch of the same command on all 16 hosts, with an identical mesh, model, batch and environment, and only `--splash` differs. Each arm runs 8 warmup steps, which are excluded, then 3 blocking windows of 10 steps (per-step host trace, submit and device time), then one 10-step pipelined window. In the pipelined window the host traces step k+1 while the device runs step k, and the device is drained once at the end. **The reported step time is the pipelined ms per step, taking the maximum over the 16 hosts.** The acceptance threshold is max(1.02, 1 + 2 × the drift between the two baselines).

## Results: FSDP 64, per-chip batch 2

Measured 2026-11-06 02:23–02:30 SGT (UTC+8).

| Arm | Pipelined ms/step ↓ | Blocking total (ms) | Host trace (ms) | Device (ms) | Final loss |
|---|---:|---:|---:|---:|---|
| A: SDPA fallback | 252.15 | 410.6–410.7 | 159–166 | 231–237 | 0.014972478151321411 |
| **B: splash** | **200.25** | 342.5 | 144–148 | 181–186 | 0.014949975535273552 |
| A2: SDPA fallback | 252.41 | 411.1 | 163–166 | 230–235 | 0.014972478151321411 |

- The two baselines differ by 0.10%, so the threshold is 1.02×.
- Speedup: 252.41 / 200.25 = **1.260×** against the slower baseline, and 1.259× against the faster one.
- Device time drops from about 232 to 182 ms (−22%). Host trace drops by about 17 ms because no mask is built per layer. The pipelined step remains device-bound.
- Ranges are across the 16 hosts. In blocking multi-host windows, "device" also includes waiting at collectives for the slowest host, so the pipelined column is the measure.

### Baseline choice: the 2-D mesh run is not the headline

An earlier A/B/A on a `data 16 × fsdp 4` mesh (parameters sharded only within a host) measured 455.84 / 399.62 / 455.54 ms, or 1.141×. Its SDPA baseline is much slower than the FSDP 64 baseline (455.8 vs 252.0 ms), so that ratio is measured against a weaker configuration. The headline 1.26× uses the FSDP 64 baseline, which is the fastest SDPA configuration measured. Writing that layout as `data 1 × fsdp 64` adds no overhead: 252.15 ms vs 252.0 ms for the 1-D FSDP form.

## Numerical validation

**Tier N1.** Splash uses the same dtype as SDPA but an online softmax with a different accumulation order, so results are close but not bitwise.

The kernel test (`tpu_smoke/test_splash_equiv.py`, 4 chips, B=4, T=2,048, H=12, D=128, BF16, causal) gives these maximum relative errors against the SDPA fallback, with tolerance 2e-2:

| Window | out | dq | dk | dv |
|---|---:|---:|---:|---:|
| 2,048 (long layers) | 5.0e-3 | 6.7e-3 | 6.4e-3 | 5.3e-3 |
| 768 (short layers) | 5.0e-3 | 6.7e-3 | 8.6e-3 | 5.4e-3 |

Window check: splash at window 768 compared with SDPA at windows 767, 768 and 769 gives distances 3.7e-2, 5.0e-3 and 3.8e-2. The applied window is therefore exactly 768, and an off-by-one is detectable.

In the timing runs, the two baselines' final losses are bitwise identical, and splash differs from them by 1.5e-3 (relative). Timing runs train on one fixed batch, so this loss is a determinism check, not a quality measure.

## Statistical gate

The gate runs 120 steps from initialization on 64 chips, FSDP 64, per-chip batch 2, sequence length 2,048. Each step draws a fresh batch from a sparse Markov chain in which each token has 4 possible successors. Seeds change only the data, and baseline and candidate runs are paired by seed.

- **Endpoints:** mean loss over 5 consecutive windows after the first 10 steps.
- **δ:** for each endpoint, 2 × the baseline's standard deviation across seeds, frozen before the candidate is examined.
- **Verdicts:** **PASS** if the mean paired difference is within δ at every endpoint. **INCORRECT** if some endpoint exceeds δ with paired t-test p < 0.05. Otherwise **UNRESOLVED**.
- **Sensitivity control:** the unchanged baseline with every learning rate ×1.02. It must be judged INCORRECT, or the gate is too blunt to trust.

The baseline loss goes from 1.56 to 1.41.

| Arm | Verdict | e0 | e1 | e2 | e3 | e4 |
|---|---|---:|---:|---:|---:|---:|
| δ (frozen) | | 0.00019 | 0.00016 | 0.00028 | 0.00021 | 0.00012 |
| **Candidate: splash** | **PASS** | −0.00002 | +0.00001 | −0.00001 | −0.00000 | −0.00000 |
| Control: lr ×1.02 | INCORRECT | −0.0950 | −0.0145 | −0.0077 | −0.0048 | −0.0039 |

Endpoint columns are mean paired differences from the baseline. Every control endpoint has p < 0.001.

This validates a short synthetic trajectory. It does not establish final model quality.

## Comparability with the H20 results

**On TPU, the comparison is like-for-like.** Baseline and candidate both run per-chip batch 2 on the same 64-chip FSDP 64 layout, so they process the same 262,144 tokens per step.

**It is not comparable to [h20-results.md](h20-results.md), and no TPU-versus-GPU claim is made.**

| | H20 benchmark | TPU v6e port |
|---|---|---|
| Devices | 8 GPUs | 64 chips |
| Batch per device | 16 | 2 |
| Gradient accumulation | 4 | none |
| Tokens per optimizer update | 1,048,576 | 262,144 |
| Matmul precision | tensorwise FP8 linear matmuls | BF16, no FP8 |
| Faster-Nanochat optimizations | active | not used on TPU |

- Per-chip batch on TPU is capped at 2 by memory (see below).
- Eight of Faster-Nanochat's nine GPU optimization switches are CUDA-specific (FP8 scaling, cuBLASLt, packed CUDA kernels). The remaining one, MLP layout, has not been ported. On TPU the baseline is the upstream model code with the SDPA fallback.
- The TPU time for a full 1,048,576-token update has not been measured.

## Memory limits

64 chips, FSDP, sequence length 2,048:

| Per-chip batch | Result |
|---:|---|
| 8 | Compile-time OOM: 47.0G needed / 31.25G available |
| 4 | Load-time OOM: 121.6M requested, 104.7M free. Pinning block outputs to batch sharding does not change it |
| **2** | **Fits; the working point above** |

On 4 chips with FSDP, per-chip batch 4 fits. Why it does not fit on 64 chips has not been located.

## Other items measured

| Item | Result |
|---|---|
| libtpu async-collective flags in `LIBTPU_INIT_ARGS` (4 chips, splash, FSDP 4) | **No benefit:** 176.14 / 176.04 / 176.22 ms, bitwise-identical loss. On v6e, torch_xla 2.9 already sets `xla_tpu_prefer_async_allgather_to_allreduce` and reruns the latency-hiding scheduler by default. |
| `torch.compile` (openxla) on the forward, D24 + 64-chip FSDP | **Rejected:** load-time OOM (7.97G needed, 4.05G available); the same configuration runs eagerly. |

After splash, host trace (about 145–160 ms) and device time (about 157–181 ms) are close. Further reductions on either side are bounded by the other under torch_xla 2.9.

## Limitations

- **Synthetic data, random initialization.** The tokenizer, dataset and learning-rate schedule are not used. Timing runs reuse one fixed batch, and gate runs use a synthetic Markov chain.
- **Precision differs from upstream GPU training.** The TPU runs use BF16 compute without FP8. The H20 benchmark and the upstream D24 recipe use tensorwise FP8 linear matmuls on GPU.
- **No long training or downstream evaluation on TPU.** Full-schedule pretraining, held-out BPB and CORE have not been run. The statistical gate covers 120 steps.
- **Not a recipe step.** One measured step is a single 262,144-token batch plus one optimizer update. The recipe update is 1,048,576 tokens (accumulation 4), which has not been timed on TPU.
- **One pipelined window per arm.** The 1.26× rests on one 10-step pipelined window per arm, bracketed by two baselines that agree to 0.10%.
- **Constant learning rate.** Optimizer hyperparameters are Python numbers on TPU. Whether changing values during a schedule triggers recompilation has not been measured.
- **Port, not upstream.** Running on TPU requires the external patches in `tpu_smoke/`. Upstream `nanochat/` is unchanged and still targets CUDA.

## Reproduction

[`tpu_smoke/README.md`](../tpu_smoke/README.md) lists the environment, the CPU tests, the exact flags and environment variables for the baseline and candidate arms, and the statistical-gate commands.

# tpu_smoke: Nanochat D24 on TPU v6e (experimental)

PyTorch/XLA harness used for the [TPU v6e results](../docs/tpu-v6e-results.md). It imports the unchanged `nanochat/` package from this repository and patches it from outside; no upstream file is edited. Run every command from the repository root.

## Files

| File | Role |
|---|---|
| `multihost_split.py` | Entry point. Builds D24, shards it with SPMD (FSDP), times blocking and pipelined steps (`SUMMARY` line) or runs a gate trajectory (`--traj N`, `TRAJ` line). |
| `tpu_bench.py` | `build()`: forces `NANOCHAT_DTYPE=bfloat16`, builds the upstream `GPT` at the recipe shape, installs the XLA optimizer patch. Its own `main()` is an older single-host timer, not used for the reported numbers. |
| `xla_optim_patch.py` | `MuonAdamW` fused AdamW/Muon steps with Python-number hyperparameters (upstream's 0-D CPU tensors cannot mix into XLA device math). |
| `nanochat_splash.py` | Pallas splash attention behind nanochat's `flash_attn_func` interface; sliding windows use a local mask instead of a dense `[T, T]` mask. Enabled by `--splash`. |
| `gate_eval.py` | Statistical gate over paired-seed trajectories: PASS / INCORRECT / UNRESOLVED. |
| `test_xla_optim_patch.py` | CPU: patched optimizer steps vs upstream fused functions. |
| `test_gate_eval.py` | CPU: gate verdicts on synthetic trajectories (matching → PASS, shifted by 0.05 → INCORRECT). |
| `test_splash_equiv.py` | TPU: splash vs nanochat's SDPA fallback, forward and q/k/v gradients, windows 2048 and 768, plus an off-by-one window check. |

## Environment

Same on every host of the slice. Python 3.11 venv:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
pip install torch-xla==2.9.0 libtpu==0.0.21 numpy filelock
# Pallas environment (needed by --splash; the reported baseline and candidate arms both ran in it)
pip install --no-deps jax==0.7.1 jaxlib==0.7.1 ml_dtypes opt_einsum scipy torchax==0.0.7
```

`nanochat_splash.py` pins the Mosaic IR version to 7 at import, because jax 0.7.1 otherwise emits version 8, which libtpu 0.0.21 rejects.

Placeholders below: `<REPO>` is this repository's root on each host, `<PY>` the Pallas environment's `python`, `<LOGS>` a log directory.

## CPU tests

Need only CPU `torch` (and `filelock`); they do not import `torch_xla` or `jax`.

```bash
python -m tpu_smoke.test_xla_optim_patch   # expect: RESULT worst=3.576e-07 equivalent=True
python -m tpu_smoke.test_gate_eval         # expect: PASS gate_eval: ...
```

## TPU equivalence test (one host, 4 chips)

```bash
cd <REPO> && PJRT_DEVICE=TPU <PY> -m tpu_smoke.test_splash_equiv
```

On one host of a multi-host slice, isolate it first: `TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-64 TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CLOUD_TPU_TASK_ID=0 TPU_VISIBLE_CHIPS=0,1,2,3`.

## Timing: baseline, candidate, baseline (64 chips, 16 hosts)

Launch the identical command on all hosts at once (for example `gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> --worker=all --command="..."`). Do **not** set the single-host isolation variables; the runtime must read the slice topology. Run the three arms as three separate launches, in this order:

```bash
cd <REPO>
export PJRT_DEVICE=TPU TORCHDYNAMO_DISABLE=1 ALLOW_MULTIPLE_LIBTPU_LOAD=1 PYTHONPATH=<REPO>
COMMON="--fsdp --fsdp-size 64 --prod-shape --depth 24 --seq 2048 --vocab 32768 \
  --per-chip-batch 2 --warmup 8 --window 10 --windows 3"
<PY> -m tpu_smoke.multihost_split $COMMON          > <LOGS>/A_sdpa.log   2>&1   # baseline (SDPA fallback)
<PY> -m tpu_smoke.multihost_split $COMMON --splash > <LOGS>/B_splash.log 2>&1   # candidate
<PY> -m tpu_smoke.multihost_split $COMMON          > <LOGS>/A2_sdpa.log  2>&1   # baseline again
```

Each process prints one `SUMMARY {json}` line. The step is as slow as the slowest host, so report the **maximum `pipelined_ms` over all 16 hosts**. `trace_ms`, `submit_ms` and `device_ms` come from the blocking windows; `last_loss` must be bitwise equal between the two baseline arms. `--fsdp-size 64` gives a `data 1 × fsdp 64` mesh; `--splash` requires `--fsdp-size`.

## Statistical gate (120 steps, paired seeds)

Same environment and launch method. For each seed, run three arms: baseline, candidate, and a sensitivity control (baseline with every learning rate ×1.02), which the gate must judge INCORRECT.

```bash
GATE="--fsdp --fsdp-size 64 --prod-shape --depth 24 --seq 2048 --vocab 32768 --per-chip-batch 2 --traj 120"
for seed in 0 1 2; do
  <PY> -m tpu_smoke.multihost_split $GATE --seed $seed                 > <LOGS>/up_s$seed.log   2>&1
  <PY> -m tpu_smoke.multihost_split $GATE --seed $seed --splash        > <LOGS>/cand_s$seed.log 2>&1
  <PY> -m tpu_smoke.multihost_split $GATE --seed $seed --lr-scale 1.02 > <LOGS>/sens_s$seed.log 2>&1
done
# on one host, after all runs:
for seed in 0 1 2; do for arm in up cand sens; do
  echo "h0 $arm s$seed $(grep -m1 '^TRAJ' <LOGS>/${arm}_s$seed.log)"
done; done > <LOGS>/gate.txt
python -m tpu_smoke.gate_eval <LOGS>/gate.txt     # --base up --skip 10 --windows 5 are the defaults
```

`--seed` changes only the synthetic data; model initialization is fixed.

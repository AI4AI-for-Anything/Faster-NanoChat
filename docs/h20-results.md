# H20 results and validation protocol

The final retained implementation is `c46b67a6bf20de83039b38903252b9b4a133af09` (Round43), based on upstream `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

## Training-update benchmark

Measured on eight NVIDIA H20 GPUs: D24 (~1.384B parameters), tensorwise FP8 linear matmuls, BF16 activations, sequence length 2,048, batch 16 per rank, accumulation 4 and 1,048,576 tokens per optimizer update.

Final full validation ran 2026-10-23 23:52:34–2026-10-24 00:17:23 UTC. It timed the original baseline, retained candidate, then original baseline again. Each arm uses 10 warmup updates followed by three windows of eight updates.

| Timing arm | Window 1 (s/update) | Window 2 | Window 3 |
|---|---:|---:|---:|
| baseline-a | 3.358429909 | 3.359958172 | 3.359157324 |
| candidate | 3.126013756 | 3.125822544 | 3.126221657 |
| baseline-b | 3.356888056 | 3.358042955 | 3.358271360 |

The frozen judge reports **1.074368586×, PASS**: a **7.44% throughput increase**, equivalent to approximately **6.92% less update time**. The baseline median across six windows is 3.358350635 seconds; candidate median is 3.126013756 seconds. The headline speedup is the median of paired-window ratios, so it need not exactly equal the ratio of these descriptive medians.

Compilation, warmup, startup and final evaluation are excluded from this steady-state timing. Peak allocated memory was about 55.3 GB per rank. The final run reused compiler caches; no physically cold-cache performance claim is made.

## Numerical validation

The final full check passes at **N1, statistical equivalence**. Against the original baseline, **3,184 / 3,208** recorded tensor hashes match; the remaining 24 are FP32 optimizer deltas in MLP projection weights at layers 21–23 across eight ranks. All 3,208 hashes match the historical retained-version full run. Errors, exact failures, violations, unresolved comparisons, dtype changes and precision-mode changes are all empty.

Seeds 42–47 each complete 24 updates. The frozen `train_early`, `train_later` and `head_update` endpoints satisfy their equivalence bounds. Seventy parent CPU audit jobs and the independent final coverage review pass. These checks establish bounded short-trajectory validation, not universal bitwise equality or final model quality.

## Selection and full-training status

Fifty exploration rounds produced eight retained commits. Round43's historical full check measured 1.073976947×; its unchanged-source final revalidation measured 1.074368586×. This small difference is measurement variation, not a new optimization. Round50's full check measured 1.073987756× without a resolved incremental benefit, so its changes were not promoted.

The campaign's **1.5× target was not reached**. **Full-schedule pretraining, held-out BPB and CORE evaluation remain unrun.** This result does not establish a better time-to-GPT-2 score, final checkpoint quality, or end-to-end training runtime. Hardware/compiler/shape generalization is untested.

## Evidence and reproduction

The [figure data](performance-views/history.json) records the timing windows, checkpoint scores, source commits and SHA256 of the original final report. The original full report, source snapshots, evaluator, numerical tensor fixtures and complete research log are retained in a separate companion research archive. They are not dependencies of this source repository.

The historical report was `final-best-validation-full/check.json` in campaign `20261022T153329Z-502ad690`. Its SHA256 is recorded in the figure data. The original judge is tied to the certified eight-H20 environment and public asset snapshot. An exact judge replay uses that research archive; porting it requires fresh environment and numerical certification. The [training guide](training.md) covers the public training entry point.

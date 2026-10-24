# Environment, training and reproduction

## Validated environment

One node with eight exclusive NVIDIA H20 GPUs, Python 3.12.3, PyTorch **2.9.1+cu128**, Triton **3.5.1**, CUDA 12.8-compatible development toolkit and driver, `kernels==0.11.7`, `numpy==1.26.4`, `pyarrow==21.0.0`, `rustbpe==0.1.0`, `tiktoken==0.11.0`. The native Lt extension needs a C++17 compiler, Ninja and cuBLAS headers/libraries from the CUDA PyTorch environment. Hopper FA3 uses `varunneal/flash-attention-3` through the upstream kernel loader.

The retained hooks depend on exact generated source and compiler metadata. They are guarded for this pinned environment. A different environment may fall back or fail before benchmarking; no cross-version acceleration claim is made.

A starting installation on an appropriate GPU node is:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra gpu --frozen
uv pip install ninja requests 'triton==3.5.1' 'kernels==0.11.7'   'numpy==1.26.4' 'pyarrow==21.0.0' 'rustbpe==0.1.0' 'tiktoken==0.11.0'
```

The upstream lockfile is preserved, not regenerated into a claim of identical deployment. Match these documented measured versions before re-measurement. The packaging session did not install or run the GPU environment.

## Public data and tokenizer

Set an absolute `NANOCHAT_BASE_DIR`. Required children are `base_data_climbmix/` (public ClimbMix parquet shards) and `tokenizer/` (the trained 32,768-token tokenizer). Training writes checkpoints under this base directory; use an output location you own. To preserve shared inputs, symlink the prepared data/tokenizer into a new output directory.

For a new public-data setup, upstream provides:

```bash
export NANOCHAT_BASE_DIR=/absolute/path/to/data-and-output
python -m nanochat.dataset -n 170
python -m scripts.tok_train
python -m scripts.tok_eval
```

Download sufficient shards for the intended schedule. These commands prepare a new tokenizer/data setup; they do not prove byte identity with the historical assets. Exact replay requires the original certified public assets and tokenizer. The dataset is `karpathy/climbmix-400b-shuffle` on Hugging Face. Preload the FA3 kernel while online before using an offline job. Evaluation additionally needs the upstream evaluation bundle; see [the upstream README](../README.upstream.md).

## Fixed D24 launch

```bash
export NANOCHAT_BASE_DIR=/absolute/path/to/prepared-data-and-output
export NANOCHAT_OPT_LT_BUILD_DIR=/absolute/path/to/node-local/native-cache
bash scripts/train_faster_d24.sh --dry-run
bash scripts/train_faster_d24.sh
```

The script fixes eight ranks, depth24, aspect64, head dimension128, SSSL attention windows, sequence2048, microbatch16, total1,048,576 tokens and tensorwise FP8. Gradient accumulation is derived as `1048576 / (8 * 16 * 2048) = 4`. It preserves the upstream optimizer/schedule defaults and uses data:param ratio8. The frozen campaign recipe reports the corresponding full schedule as5,568 updates; only short observed update windows were run. The launcher is a proposed full-training follow-up and does not recreate the frozen observer's determinism configuration or numerical observations. Evaluation/sampling are disabled in this launcher, matching the timing recipe; final model evaluation is a separate step.

For other memory budgets, use the upstream interface, preserving the total token batch if comparing outcomes. Microbatch or GPU changes are outside this measured configuration. Per-rank benchmark allocated memory was about55.3GB; reserve additional headroom for compilation and full-run evaluation. No new GPU allocation or training was started during archival.

## Benchmark reproduction

The historical benchmark protocol, measured windows and numerical gates are documented in [H20 results](h20-results.md). The compact [figure data](performance-views/history.json) binds those values to the original report hash and retained source commit.

The original frozen judge, baseline/candidate snapshots, binary numerical fixtures and deployment-specific replay scripts belong to the separate research archive. The source repository contains no copy of that environment. An exact historical replay requires that archive and its certified public assets; it is separate from running the portable pretraining entry point above. A different machine or software stack needs fresh measurement and numerical validation.

## Required quality follow-up

Run complete baseline and optimized training with identical certified data/tokenizer, recipe, seeds and evaluation settings. Then compare held-out BPB and CORE. Use upstream `scripts/base_eval.py` and the upstream evaluation assets. No weights or scored BPB/CORE results are included because these runs have not been performed.

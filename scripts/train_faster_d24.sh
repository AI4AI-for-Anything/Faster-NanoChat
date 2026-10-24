#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
# Fixed supported recipe; --dry-run only prints, never allocates GPUs.
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ "$#" != 0 ]]; then echo 'Usage: train_faster_d24.sh [--dry-run]; see docs/training.md for environment options' >&2; exit 2; fi
: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to prepared public data/tokenizer and output directory}"
source scripts/configs/faster_d24.sh
[[ "$NANOCHAT_OPT_LT_BUILD_DIR" == /* ]] || { echo 'NANOCHAT_OPT_LT_BUILD_DIR must be absolute' >&2; exit 2; }
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
# Deterministic mode is configured by the frozen observer; this standalone launch
# is a full-training follow-up, not the historical benchmark invocation.
CMD=("${PYTHON:-python3}" -m torch.distributed.run --standalone --nproc_per_node=8
     -m scripts.base_train --depth=24 --aspect-ratio=64 --head-dim=128
     --max-seq-len=2048 --window-pattern=SSSL --device-batch-size=16
     --total-batch-size=1048576 --target-param-data-ratio=8 --fp8 --fp8-recipe=tensorwise
     --num-iterations=-1 --run=dummy --eval-every=-1 --core-metric-every=-1
     --sample-every=-1 --save-every=-1 --model-tag="${MODEL_TAG:-faster-d24}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
[[ -d "$NANOCHAT_BASE_DIR/base_data_climbmix" && -d "$NANOCHAT_BASE_DIR/tokenizer" ]] || { echo 'Prepared ClimbMix data and tokenizer are required' >&2; exit 2; }
exec "${CMD[@]}"

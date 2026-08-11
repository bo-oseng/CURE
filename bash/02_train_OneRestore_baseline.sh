#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_GPUS="${GPUS:-4}"
CURE_EMBEDDER_CHECKPOINT="${EMBEDDER_CHECKPOINT:-checkpoints/OneRestore_embedder.tar}"
CURE_BASELINE_OUTPUT="${BASELINE_OUTPUT:-checkpoints/02_OneRestore}"
CURE_BASELINE_H5="${BASELINE_H5:-datasets_h5/half_og_train.h5}"

for CURE_REQUIRED_FILE in "$CURE_EMBEDDER_CHECKPOINT" "$CURE_BASELINE_H5"; do
  if [[ ! -f "$CURE_REQUIRED_FILE" ]]; then
    echo "Missing required file: $CURE_REQUIRED_FILE" >&2
    exit 1
  fi
done

torchrun --standalone --nproc_per_node="$CURE_GPUS" 02_train_Onerestore_baseline.py \
  --embedder-checkpoint "$CURE_EMBEDDER_CHECKPOINT" \
  --train-h5 "$CURE_BASELINE_H5" \
  --output-dir "$CURE_BASELINE_OUTPUT" \
  --epochs 300 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --lr-decay-every 30 \
  "$@"

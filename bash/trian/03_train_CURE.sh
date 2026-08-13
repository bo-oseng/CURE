#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_GPUS="${GPUS:-4}"
CURE_EMBEDDER_CHECKPOINT="${EMBEDDER_CHECKPOINT:-checkpoints/OneRestore_embedder.tar}"
CURE_BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-checkpoints/OneRestore_restorer.tar}"
CURE_TRAIN_OUTPUT="${CURE_OUTPUT:-outputs/03_cure}"
CURE_TRAIN_H5="${CURE_H5:-datasets_h5/half_train.h5}"

for CURE_REQUIRED_FILE in \
  "$CURE_EMBEDDER_CHECKPOINT" \
  "$CURE_BASELINE_CHECKPOINT" \
  "$CURE_TRAIN_H5"; do
  if [[ ! -f "$CURE_REQUIRED_FILE" ]]; then
    echo "Missing required file: $CURE_REQUIRED_FILE" >&2
    exit 1
  fi
done

torchrun --standalone --nproc_per_node="$CURE_GPUS" 03_train_CURE.py \
  --embedder-checkpoint "$CURE_EMBEDDER_CHECKPOINT" \
  --baseline-checkpoint "$CURE_BASELINE_CHECKPOINT" \
  --train-h5 "$CURE_TRAIN_H5" \
  --output-dir "$CURE_TRAIN_OUTPUT" \
  --epochs 300 \
  --batch-size 6 \
  --learning-rate 2.5e-5 \
  --lr-decay-every 30 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_GPUS="${GPUS:-4}"
CURE_EMBEDDER_OUTPUT="${EMBEDDER_OUTPUT:-checkpoints/01_embedder}"

if [[ ! -d data/half_train/main_data || ! -d data/half_test/main_data ]]; then
  echo "Missing CCDD-11 data. Link or place it under data/half_{train,test}/main_data." >&2
  exit 1
fi

torchrun --standalone --nproc_per_node="$CURE_GPUS" 01_train_OneRestore_embedder.py \
  --train-dir data/half_train/main_data \
  --val-dir data/half_test/main_data \
  --glove assets/glove.6B.300d.txt \
  --output-dir "$CURE_EMBEDDER_OUTPUT" \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --lr-decay-every 50 \
  "$@"

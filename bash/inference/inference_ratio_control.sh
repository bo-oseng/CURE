#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_PYTHON="${PYTHON:-python}"
CURE_PROMPT="${PROMPT:-haze}"
CURE_OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference}"
CURE_OUTPUT="${OUTPUT:-$CURE_OUTPUT_ROOT/ratio_control/$CURE_PROMPT}"
CURE_INPUT_ARGS=()

if [[ -n "${INPUT:-}" ]]; then
  CURE_INPUT_ARGS=(--input "$INPUT")
fi

"$CURE_PYTHON" inference_ratio_control.py \
  "${CURE_INPUT_ARGS[@]}" \
  --prompt "$CURE_PROMPT" \
  --output "$CURE_OUTPUT" \
  "$@"

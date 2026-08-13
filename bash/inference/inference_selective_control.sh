#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_PYTHON="${PYTHON:-python}"
CURE_SOURCE_PROMPT="${SOURCE_PROMPT:-low_haze}"
CURE_REMOVE="${REMOVE:-haze}"
CURE_OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference}"
CURE_OUTPUT="${OUTPUT:-$CURE_OUTPUT_ROOT/selective_control/$CURE_SOURCE_PROMPT/remove_${CURE_REMOVE// /_}}"
CURE_INPUT_ARGS=()
CURE_REMOVE_FACTORS=()

if [[ -n "${INPUT:-}" ]]; then
  CURE_INPUT_ARGS=(--input "$INPUT")
fi
read -r -a CURE_REMOVE_FACTORS <<< "$CURE_REMOVE"

"$CURE_PYTHON" inference_selective_control.py \
  "${CURE_INPUT_ARGS[@]}" \
  --source-prompt "$CURE_SOURCE_PROMPT" \
  --remove "${CURE_REMOVE_FACTORS[@]}" \
  --output "$CURE_OUTPUT" \
  "$@"

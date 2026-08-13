#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_PYTHON="${PYTHON:-python}"
CURE_SOURCE_PROMPT="${SOURCE_PROMPT:-low_haze}"
CURE_OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference}"
CURE_OUTPUT="${OUTPUT:-$CURE_OUTPUT_ROOT/twostage/$CURE_SOURCE_PROMPT}"
CURE_INPUT_ARGS=()
CURE_SEQUENCE_ARGS=()

if [[ -n "${INPUT:-}" ]]; then
  CURE_INPUT_ARGS=(--input "$INPUT")
fi
if [[ -n "${SEQUENCE:-}" ]]; then
  read -r -a CURE_SEQUENCE <<< "$SEQUENCE"
  CURE_SEQUENCE_ARGS=(--sequence "${CURE_SEQUENCE[@]}")
fi

"$CURE_PYTHON" inference_twostage.py \
  "${CURE_INPUT_ARGS[@]}" \
  --source-prompt "$CURE_SOURCE_PROMPT" \
  "${CURE_SEQUENCE_ARGS[@]}" \
  --output "$CURE_OUTPUT" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

CURE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURE_REPO_ROOT"

CURE_PYTHON="${PYTHON:-python}"
CURE_SOURCE_PROMPT="${SOURCE_PROMPT:-clear}"
CURE_OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference}"
CURE_OUTPUT="${OUTPUT:-$CURE_OUTPUT_ROOT/identity/$CURE_SOURCE_PROMPT}"
CURE_INPUT_ARGS=()

if [[ -n "${INPUT:-}" ]]; then
  CURE_INPUT_ARGS=(--input "$INPUT")
fi

"$CURE_PYTHON" inference_identity.py \
  "${CURE_INPUT_ARGS[@]}" \
  --source-prompt "$CURE_SOURCE_PROMPT" \
  --output "$CURE_OUTPUT" \
  "$@"

#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_SOURCE="${IMAGE_SOURCE:-data/testdata}"
REPEAT="${REPEAT:-1}"
WARMUP="${WARMUP:-1}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-2048}"
OUT_DIR="${OUT_DIR:-runs/timing_$(date +%Y%m%d_%H%M%S)}"

cd "$SCRIPT_DIR" || exit 1

"$PYTHON_BIN" run_timed_infer.py \
  --image-source "$IMAGE_SOURCE" \
  --model models/best.pt \
  --device "$DEVICE" \
  --imgsz "$IMGSZ" \
  --repeat "$REPEAT" \
  --warmup "$WARMUP" \
  --out "$OUT_DIR" \
  "$@"

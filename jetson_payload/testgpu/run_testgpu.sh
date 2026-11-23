#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") --onnx MODEL.onnx [options]

Wrapper for run_onnx_gpu_monitor.py.

Examples:
  bash testgpu/run_testgpu.sh --onnx /path/to/model.onnx
  bash testgpu/run_testgpu.sh --onnx /path/to/model.onnx --iterations 1000
  bash testgpu/run_testgpu.sh --onnx /path/to/model.onnx --provider tensorrt
EOF
}

if [ "$#" -eq 0 ]; then
  usage
  exit 2
fi

python3 "$SCRIPT_DIR/run_onnx_gpu_monitor.py" "$@"

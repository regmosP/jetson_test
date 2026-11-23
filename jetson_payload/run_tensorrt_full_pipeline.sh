#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-trt_pipeline_$(date +%Y%m%d_%H%M%S)}"
PRECISION="${PRECISION:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HEIGHT="${HEIGHT:-224}"
WIDTH="${WIDTH:-224}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
WORKSPACE_MB="${WORKSPACE_MB:-512}"
STATUS_INTERVAL="${STATUS_INTERVAL:-1}"
STATUS_LOGGER="${STATUS_LOGGER:-$SCRIPT_DIR/continuous_status_logger.sh}"
INPUT_NAME="${INPUT_NAME:-input}"
OUTPUT_NAME="${OUTPUT_NAME:-logits}"
ONNX_PATH="${ONNX_PATH:-}"
ENGINE_PATH="${ENGINE_PATH:-}"
IMAGE_PATH="${IMAGE_PATH:-}"
INPUT_SCALE="${INPUT_SCALE:-0.00392156862745098}"
INPUT_MEAN="${INPUT_MEAN:-0.485 0.456 0.406}"
INPUT_STD="${INPUT_STD:-0.229 0.224 0.225}"
GENERATE_ONNX="${GENERATE_ONNX:-1}"
MAX_PERF="${MAX_PERF:-1}"
OPEN_REPORT="${OPEN_REPORT:-0}"
VERBOSE="${VERBOSE:-0}"
REBUILD="${REBUILD:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run a complete TensorRT pipeline test on Jetson Nano:
environment capture -> optional max performance -> ONNX -> TensorRT engine -> inference benchmark -> JSON/CSV/HTML results.

Options:
  --onnx PATH             Use an existing ONNX model.
  --engine PATH           Use or write TensorRT engine path.
  --image PATH            Repeatedly infer the same image instead of random input.
  --out DIR               Result directory, default: $OUT_DIR.
  --precision fp32|fp16   Precision, default: $PRECISION.
  --batch-size N          Batch size, default: $BATCH_SIZE.
  --input-hw H W          Input size, default: $HEIGHT $WIDTH.
  --warmup N              Warmup iterations, default: $WARMUP.
  --iterations N          Measured iterations, default: $ITERATIONS.
  --workspace-mb N        TensorRT workspace MB, default: $WORKSPACE_MB.
  --status-interval SEC   Continuous status logging interval, default: $STATUS_INTERVAL.
  --input-name NAME       ONNX input tensor name, default: $INPUT_NAME.
  --output-name NAME      ONNX output tensor name, default: $OUTPUT_NAME.
  --input-scale X         Image pixel scale, default: $INPUT_SCALE.
  --input-mean R G B      RGB mean, default: $INPUT_MEAN.
  --input-std R G B       RGB std, default: $INPUT_STD.
  --no-generate-onnx      Require --onnx instead of generating a small model.
  --no-max-perf           Do not run nvpmodel/jetson_clocks.
  --rebuild               Rebuild engine even if it exists.
  --verbose               Enable verbose TensorRT logger.
  --open-report           Open HTML report after completion when xdg-open exists.
  -h, --help              Show this help.

Examples:
  sudo bash $(basename "$0") --precision fp16 --iterations 200 --open-report
  sudo bash $(basename "$0") --image test.jpg --precision fp16 --iterations 200 --open-report
  bash $(basename "$0") --onnx model.onnx --engine model.trt --batch-size 1 --precision fp16
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --onnx) ONNX_PATH="$2"; GENERATE_ONNX=0; shift 2 ;;
    --engine) ENGINE_PATH="$2"; shift 2 ;;
    --image) IMAGE_PATH="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --input-hw) HEIGHT="$2"; WIDTH="$3"; shift 3 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --workspace-mb) WORKSPACE_MB="$2"; shift 2 ;;
    --status-interval) STATUS_INTERVAL="$2"; shift 2 ;;
    --input-name) INPUT_NAME="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --input-scale) INPUT_SCALE="$2"; shift 2 ;;
    --input-mean) INPUT_MEAN="$2 $3 $4"; shift 4 ;;
    --input-std) INPUT_STD="$2 $3 $4"; shift 4 ;;
    --no-generate-onnx) GENERATE_ONNX=0; shift ;;
    --no-max-perf) MAX_PERF=0; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --open-report) OPEN_REPORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
ENV_LOG="$OUT_DIR/environment.log"
TEGR_STATS_LOG="$OUT_DIR/tegrastats.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_LOG"; }
have() { command -v "$1" >/dev/null 2>&1; }

capture_environment() {
  {
    echo "== command =="
    printf '%q ' "$0" "$@"; echo
    echo "== date =="; date
    echo "== uname =="; uname -a
    echo "== jetson model =="; cat /proc/device-tree/model 2>/dev/null || true; echo
    echo "== l4t =="; dpkg-query --show nvidia-l4t-core 2>/dev/null || true
    echo "== tensorrt packages =="; dpkg -l 2>/dev/null | grep -i tensorrt || true
    echo "== cuda =="; nvcc --version 2>/dev/null || true
    echo "== python packages =="
    "$PYTHON_BIN" - <<'PY' 2>/dev/null || true
for name in ('tensorrt', 'pycuda', 'numpy', 'torch', 'onnx'):
    try:
        module = __import__(name)
        print(f'{name}={getattr(module, "__version__", "installed")}')
    except Exception as exc:
        print(f'{name}=missing ({exc})')
PY
    echo "== nvpmodel =="; nvpmodel -q 2>/dev/null || true
    echo "== jetson_clocks =="; jetson_clocks --show 2>/dev/null || true
    echo "== memory =="; free -h
    echo "== disk =="; df -h
  } > "$ENV_LOG" 2>&1
}

enable_max_perf() {
  [ "$MAX_PERF" = "1" ] || return 0
  log "Enabling Jetson maximum performance mode"
  sudo nvpmodel -m 0 >> "$RUN_LOG" 2>&1 || nvpmodel -m 0 >> "$RUN_LOG" 2>&1 || true
  sudo jetson_clocks >> "$RUN_LOG" 2>&1 || jetson_clocks >> "$RUN_LOG" 2>&1 || true
}

start_tegrastats() {
  TEGRASTATS_PID=""
  if have tegrastats; then
    tegrastats --interval 1000 > "$TEGR_STATS_LOG" 2>&1 &
    TEGRASTATS_PID="$!"
    log "tegrastats logging to $TEGR_STATS_LOG"
  else
    log "tegrastats not found; GPU utilization and temperature timeline will be unavailable"
  fi
}

stop_tegrastats() {
  if [ -n "${TEGRASTATS_PID:-}" ]; then
    kill "$TEGRASTATS_PID" >/dev/null 2>&1 || true
    wait "$TEGRASTATS_PID" >/dev/null 2>&1 || true
  fi
}

start_continuous_logger() {
  STATUS_LOGGER_PID=""
  if [ -x "$STATUS_LOGGER" ]; then
    bash "$STATUS_LOGGER" --out "$OUT_DIR" --interval "$STATUS_INTERVAL" --prefix continuous_status >/dev/null 2>&1 &
    STATUS_LOGGER_PID="$!"
    log "continuous status logging to $OUT_DIR/continuous_status_metrics.tsv"
  else
    log "continuous status logger not executable: $STATUS_LOGGER"
  fi
}

stop_continuous_logger() {
  if [ -n "${STATUS_LOGGER_PID:-}" ]; then
    kill "$STATUS_LOGGER_PID" >/dev/null 2>&1 || true
    wait "$STATUS_LOGGER_PID" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_tegrastats
  stop_continuous_logger
}

trap cleanup EXIT INT TERM

build_args=(
  "$SCRIPT_DIR/tensorrt_full_pipeline.py"
  --out-dir "$OUT_DIR"
  --precision "$PRECISION"
  --batch-size "$BATCH_SIZE"
  --input-hw "$HEIGHT" "$WIDTH"
  --warmup "$WARMUP"
  --iterations "$ITERATIONS"
  --workspace-mb "$WORKSPACE_MB"
  --input-name "$INPUT_NAME"
  --output-name "$OUTPUT_NAME"
)

[ -n "$ONNX_PATH" ] && build_args+=(--onnx "$ONNX_PATH")
[ -n "$ENGINE_PATH" ] && build_args+=(--engine "$ENGINE_PATH")
[ -n "$IMAGE_PATH" ] && build_args+=(--image "$IMAGE_PATH")
build_args+=(--input-scale "$INPUT_SCALE")
read -r mean_r mean_g mean_b <<< "$INPUT_MEAN"
read -r std_r std_g std_b <<< "$INPUT_STD"
build_args+=(--input-mean "$mean_r" "$mean_g" "$mean_b")
build_args+=(--input-std "$std_r" "$std_g" "$std_b")
[ "$GENERATE_ONNX" = "1" ] && build_args+=(--generate-onnx)
[ "$REBUILD" = "1" ] && build_args+=(--rebuild)
[ "$VERBOSE" = "1" ] && build_args+=(--verbose)

log "Result directory: $OUT_DIR"
capture_environment "$@"
enable_max_perf
start_continuous_logger
start_tegrastats
log "Running TensorRT full pipeline"
"$PYTHON_BIN" "${build_args[@]}" 2>&1 | tee -a "$RUN_LOG"
rc="${PIPESTATUS[0]}"
stop_tegrastats

if [ "$rc" -eq 0 ]; then
  log "TensorRT pipeline completed successfully"
  log "HTML report: $OUT_DIR/tensorrt_report.html"
  log "JSON result: $OUT_DIR/tensorrt_pipeline_result.json"
  log "CSV iterations: $OUT_DIR/tensorrt_iteration_metrics.csv"
  if [ "$OPEN_REPORT" = "1" ] && have xdg-open; then
    xdg-open "$OUT_DIR/tensorrt_report.html" >/dev/null 2>&1 || true
  fi
else
  log "TensorRT pipeline failed with exit code $rc"
fi

exit "$rc"

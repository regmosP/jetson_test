#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-yolo_trt_pipeline_$(date +%Y%m%d_%H%M%S)}"
YOLO_REPO="${YOLO_REPO:-$OUT_DIR/yolov5}"
YOLO_VERSION="${YOLO_VERSION:-v7.0}"
WEIGHTS="${WEIGHTS:-yolov5n.pt}"
IMAGE_PATH="${IMAGE_PATH:-}"
IMG_SIZE="${IMG_SIZE:-640}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PRECISION="${PRECISION:-fp16}"
WORKSPACE_MB="${WORKSPACE_MB:-512}"
STATUS_INTERVAL="${STATUS_INTERVAL:-1}"
STATUS_LOGGER="${STATUS_LOGGER:-$SCRIPT_DIR/continuous_status_logger.sh}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-200}"
MAX_PERF="${MAX_PERF:-1}"
OPEN_REPORT="${OPEN_REPORT:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
ONNX_PATH="${ONNX_PATH:-}"
ENGINE_PATH="${ENGINE_PATH:-}"
REBUILD="${REBUILD:-1}"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Complete YOLO -> ONNX -> TensorRT -> performance -> analysis pipeline for Jetson Nano.

Options:
  --weights PATH|NAME      YOLOv5 PyTorch weights, default: $WEIGHTS.
  --image PATH             Same image used repeatedly for TensorRT inference.
  --img-size N             YOLO input size, default: $IMG_SIZE.
  --batch-size N           Batch size, default: $BATCH_SIZE.
  --precision fp32|fp16    TensorRT precision, default: $PRECISION.
  --iterations N           Measured inference iterations, default: $ITERATIONS.
  --warmup N               Warmup iterations, default: $WARMUP.
  --workspace-mb N         TensorRT workspace MB, default: $WORKSPACE_MB.
  --status-interval SEC    Continuous status logging interval, default: $STATUS_INTERVAL.
  --out DIR                Result directory, default: timestamped directory.
  --yolo-repo PATH         YOLOv5 repo directory, default: OUT_DIR/yolov5.
  --yolo-version TAG       YOLOv5 git tag/branch, default: $YOLO_VERSION.
  --onnx PATH              Use existing ONNX path and skip export.
  --engine PATH            Use or write TensorRT engine path.
  --skip-export            Skip ONNX export and require --onnx.
  --no-max-perf            Do not run max performance mode.
  --no-rebuild             Do not rebuild existing TensorRT engine.
  --open-report            Open TensorRT HTML report if xdg-open exists.
  -h, --help               Show this help.

Examples:
  sudo bash $(basename "$0") --image test.jpg --weights yolov5n.pt --precision fp16 --open-report
  bash $(basename "$0") --onnx model.onnx --image test.jpg --skip-export --iterations 200
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --image) IMAGE_PATH="$2"; shift 2 ;;
    --img-size) IMG_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --workspace-mb) WORKSPACE_MB="$2"; shift 2 ;;
    --status-interval) STATUS_INTERVAL="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --yolo-repo) YOLO_REPO="$2"; shift 2 ;;
    --yolo-version) YOLO_VERSION="$2"; shift 2 ;;
    --onnx) ONNX_PATH="$2"; SKIP_EXPORT=1; shift 2 ;;
    --engine) ENGINE_PATH="$2"; shift 2 ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --no-max-perf) MAX_PERF=0; shift ;;
    --no-rebuild) REBUILD=0; shift ;;
    --open-report) OPEN_REPORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/yolo_pipeline.log"
ENV_LOG="$OUT_DIR/environment.log"
EXPORT_LOG="$OUT_DIR/yolo_onnx_export.log"
RUN_CONFIG="$OUT_DIR/run_config.json"
PRE_STATUS_LOG="$OUT_DIR/pre_perf_status.log"
POST_STATUS_LOG="$OUT_DIR/post_perf_status.log"
MANIFEST="$OUT_DIR/artifacts_manifest.tsv"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_LOG"; }
have() { command -v "$1" >/dev/null 2>&1; }

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

write_run_config() {
  cat > "$RUN_CONFIG" <<EOF
{
  "started_at": "$(date '+%F %T')",
  "command": "$(json_escape "$0 $*")",
  "script_dir": "$(json_escape "$SCRIPT_DIR")",
  "out_dir": "$(json_escape "$OUT_DIR")",
  "python_bin": "$(json_escape "$PYTHON_BIN")",
  "yolo_repo": "$(json_escape "$YOLO_REPO")",
  "yolo_version": "$(json_escape "$YOLO_VERSION")",
  "weights": "$(json_escape "$WEIGHTS")",
  "image_path": "$(json_escape "$IMAGE_PATH")",
  "img_size": $IMG_SIZE,
  "batch_size": $BATCH_SIZE,
  "precision": "$(json_escape "$PRECISION")",
  "workspace_mb": $WORKSPACE_MB,
  "status_interval": $STATUS_INTERVAL,
  "warmup": $WARMUP,
  "iterations": $ITERATIONS,
  "max_perf": $MAX_PERF,
  "skip_export": $SKIP_EXPORT,
  "onnx_path": "$(json_escape "$ONNX_PATH")",
  "engine_path": "$(json_escape "$ENGINE_PATH")",
  "rebuild": $REBUILD
}
EOF
}

capture_status_snapshot() {
  local target="$1"
  {
    echo "== date =="; date
    echo "== nvpmodel =="; nvpmodel -q 2>/dev/null || true
    echo "== jetson_clocks =="; jetson_clocks --show 2>/dev/null || true
    echo "== tegrastats sample =="; timeout 3s tegrastats --interval 1000 2>/dev/null || true
    echo "== cpu frequencies =="
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do [ -e "$path" ] && echo "$path=$(cat "$path")"; done
    echo "== gpu frequencies =="
    for path in /sys/devices/gpu.0/devfreq/*/cur_freq /sys/kernel/debug/clk/override.gbus/rate; do [ -e "$path" ] && echo "$path=$(cat "$path" 2>/dev/null || true)"; done
    echo "== thermal zones =="
    for path in /sys/devices/virtual/thermal/thermal_zone*/temp; do [ -e "$path" ] && echo "$path=$(cat "$path")"; done
  } > "$target" 2>&1
}

start_continuous_logger() {
  STATUS_LOGGER_PID=""
  if [ -x "$STATUS_LOGGER" ]; then
    bash "$STATUS_LOGGER" --out "$OUT_DIR" --interval "$STATUS_INTERVAL" --prefix yolo_continuous_status >/dev/null 2>&1 &
    STATUS_LOGGER_PID="$!"
    log "continuous status logging to $OUT_DIR/yolo_continuous_status_metrics.tsv"
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
  stop_continuous_logger
}

trap cleanup EXIT INT TERM

capture_environment() {
  {
    echo "== date =="; date
    echo "== uname =="; uname -a
    echo "== jetson model =="; cat /proc/device-tree/model 2>/dev/null || true; echo
    echo "== l4t =="; dpkg-query --show nvidia-l4t-core 2>/dev/null || true
    echo "== cuda =="; nvcc --version 2>/dev/null || true
    echo "== tensorrt =="; dpkg -l 2>/dev/null | grep -i tensorrt || true
    echo "== nvpmodel =="; nvpmodel -q 2>/dev/null || true
    echo "== jetson_clocks =="; jetson_clocks --show 2>/dev/null || true
    echo "== disk =="; df -h
    echo "== memory =="; free -h
    echo "== pip freeze =="; "$PYTHON_BIN" -m pip freeze 2>/dev/null || true
    echo "== apt nvidia packages =="; dpkg -l 2>/dev/null | grep -Ei 'nvidia|cuda|cudnn|tensorrt' || true
    echo "== python packages =="
    "$PYTHON_BIN" - <<'PY' 2>/dev/null || true
for name in ('torch', 'torchvision', 'onnx', 'tensorrt', 'pycuda', 'numpy', 'PIL', 'cv2'):
    try:
        module = __import__(name)
        print(f'{name}={getattr(module, "__version__", "installed")}')
    except Exception as exc:
        print(f'{name}=missing ({exc})')
PY
  } > "$ENV_LOG" 2>&1
}

enable_max_perf() {
  [ "$MAX_PERF" = "1" ] || return 0
  log "Enabling max performance mode"
  sudo nvpmodel -m 0 >> "$RUN_LOG" 2>&1 || nvpmodel -m 0 >> "$RUN_LOG" 2>&1 || true
  sudo jetson_clocks >> "$RUN_LOG" 2>&1 || jetson_clocks >> "$RUN_LOG" 2>&1 || true
}

prepare_yolo_repo() {
  if [ -d "$YOLO_REPO/.git" ]; then
    log "Using existing YOLO repo: $YOLO_REPO"
    git -C "$YOLO_REPO" fetch --tags >> "$RUN_LOG" 2>&1 || true
    git -C "$YOLO_REPO" checkout "$YOLO_VERSION" >> "$RUN_LOG" 2>&1 || true
    git -C "$YOLO_REPO" rev-parse HEAD > "$OUT_DIR/yolov5_git_commit.txt" 2>/dev/null || true
    return
  fi
  if ! have git; then
    echo "git is required to clone YOLOv5" >&2
    exit 1
  fi
  log "Cloning YOLOv5 $YOLO_VERSION into $YOLO_REPO"
  git clone --depth 1 --branch "$YOLO_VERSION" https://github.com/ultralytics/yolov5.git "$YOLO_REPO" >> "$RUN_LOG" 2>&1
  git -C "$YOLO_REPO" rev-parse HEAD > "$OUT_DIR/yolov5_git_commit.txt" 2>/dev/null || true
}

prepare_weights() {
  if [ -f "$WEIGHTS" ]; then
    WEIGHTS_PATH="$WEIGHTS"
    cp "$WEIGHTS_PATH" "$OUT_DIR/$(basename "$WEIGHTS_PATH")" 2>/dev/null || true
    return
  fi
  local target="$OUT_DIR/$WEIGHTS"
  if [ -f "$target" ]; then
    WEIGHTS_PATH="$target"
    return
  fi
  if [[ "$WEIGHTS" == yolov5*.pt ]]; then
    local url="https://github.com/ultralytics/yolov5/releases/download/$YOLO_VERSION/$WEIGHTS"
    log "Downloading weights: $url"
    if have wget; then
      wget -O "$target" "$url" >> "$RUN_LOG" 2>&1
    elif have curl; then
      curl -L -o "$target" "$url" >> "$RUN_LOG" 2>&1
    else
      echo "wget or curl is required to download weights" >&2
      exit 1
    fi
    WEIGHTS_PATH="$target"
    return
  fi
  echo "Weights not found: $WEIGHTS" >&2
  exit 1
}

export_onnx() {
  if [ "$SKIP_EXPORT" = "1" ]; then
    [ -n "$ONNX_PATH" ] || { echo "--skip-export requires --onnx" >&2; exit 1; }
    log "Skipping ONNX export, using $ONNX_PATH"
    return
  fi
  prepare_yolo_repo
  local weights_path
  prepare_weights
  weights_path="$WEIGHTS_PATH"
  log "Exporting YOLO ONNX from $weights_path"
  (
    cd "$YOLO_REPO" || exit 1
    "$PYTHON_BIN" export.py \
      --weights "$weights_path" \
      --include onnx \
      --imgsz "$IMG_SIZE" "$IMG_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --opset 12 \
      --simplify \
      --device cpu
  ) > "$EXPORT_LOG" 2>&1
  local exported="${weights_path%.pt}.onnx"
  if [ ! -f "$exported" ]; then
    exported="$(find "$OUT_DIR" "$YOLO_REPO" -name '*.onnx' -print 2>/dev/null | head -n 1)"
  fi
  [ -n "$exported" ] && [ -f "$exported" ] || { echo "ONNX export failed; see $EXPORT_LOG" >&2; exit 1; }
  ONNX_PATH="$OUT_DIR/yolo_model.onnx"
  cp "$exported" "$ONNX_PATH"
  [ -n "$WEIGHTS_PATH" ] && cp "$WEIGHTS_PATH" "$OUT_DIR/$(basename "$WEIGHTS_PATH")" 2>/dev/null || true
  log "ONNX exported to $ONNX_PATH"
}

run_trt_benchmark() {
  local args=(
    "$SCRIPT_DIR/run_tensorrt_full_pipeline.sh"
    --onnx "$ONNX_PATH"
    --out "$OUT_DIR"
    --precision "$PRECISION"
    --batch-size "$BATCH_SIZE"
    --input-hw "$IMG_SIZE" "$IMG_SIZE"
    --warmup "$WARMUP"
    --iterations "$ITERATIONS"
    --workspace-mb "$WORKSPACE_MB"
    --status-interval "$STATUS_INTERVAL"
    --input-name images
    --output-name output0
    --input-scale 0.00392156862745098
    --input-mean 0 0 0
    --input-std 1 1 1
    --no-generate-onnx
  )
  [ -n "$ENGINE_PATH" ] && args+=(--engine "$ENGINE_PATH")
  [ -n "$IMAGE_PATH" ] && args+=(--image "$IMAGE_PATH")
  [ "$MAX_PERF" = "0" ] && args+=(--no-max-perf)
  [ "$REBUILD" = "1" ] && args+=(--rebuild)
  [ "$OPEN_REPORT" = "1" ] && args+=(--open-report)
  log "Running TensorRT performance benchmark"
  bash "${args[@]}" 2>&1 | tee -a "$RUN_LOG"
  return "${PIPESTATUS[0]}"
}

analyze_results() {
  local result_json="$OUT_DIR/tensorrt_pipeline_result.json"
  [ -f "$result_json" ] || { echo "Missing TensorRT result JSON: $result_json" >&2; exit 1; }
  log "Analyzing YOLO TensorRT results"
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_yolo_trt_results.py" \
    --result-json "$result_json" \
    --tegrastats "$OUT_DIR/tegrastats.log" \
    --out "$OUT_DIR/yolo_result_analysis.md" | tee -a "$RUN_LOG"
}

copy_input_image() {
  [ -n "$IMAGE_PATH" ] || return 0
  [ -f "$IMAGE_PATH" ] || return 0
  cp "$IMAGE_PATH" "$OUT_DIR/input_image_$(basename "$IMAGE_PATH")" 2>/dev/null || true
}

generate_manifest() {
  {
    printf 'path\tsize_bytes\tsha256\n'
    find "$OUT_DIR" -type f -print 2>/dev/null | sort | while read -r file; do
      size="$(wc -c < "$file" 2>/dev/null || echo 0)"
      if have sha256sum; then
        hash="$(sha256sum "$file" | awk '{print $1}')"
      else
        hash="sha256sum_unavailable"
      fi
      printf '%s\t%s\t%s\n' "$file" "$size" "$hash"
    done
  } > "$MANIFEST"
}

log "YOLO TensorRT result directory: $OUT_DIR"
write_run_config "${ORIGINAL_ARGS[@]}"
copy_input_image
capture_environment
capture_status_snapshot "$PRE_STATUS_LOG"
enable_max_perf
start_continuous_logger
export_onnx
run_trt_benchmark
rc="$?"
stop_continuous_logger
capture_status_snapshot "$POST_STATUS_LOG"
if [ "$rc" -eq 0 ]; then
  analyze_results
  generate_manifest
  log "YOLO TensorRT pipeline completed"
  log "Analysis: $OUT_DIR/yolo_result_analysis.md"
  log "HTML report: $OUT_DIR/tensorrt_report.html"
  log "Artifact manifest: $MANIFEST"
else
  generate_manifest
  log "YOLO TensorRT benchmark failed with exit code $rc"
  log "Artifact manifest: $MANIFEST"
fi
exit "$rc"

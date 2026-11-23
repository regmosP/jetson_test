#!/usr/bin/env bash
set -u

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-jetson_benchmark_$(date +%Y%m%d_%H%M%S)}"
DURATION="${DURATION:-20}"
STATUS_INTERVAL="${STATUS_INTERVAL:-1}"
STATUS_LOGGER="${STATUS_LOGGER:-$SCRIPT_DIR/continuous_status_logger.sh}"
DISK_TEST_DIR="${DISK_TEST_DIR:-$PWD}"
DISK_TEST_SIZE_MB="${DISK_TEST_SIZE_MB:-512}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_MAX_PERF="${RUN_MAX_PERF:-0}"
CPU_WORKERS="${CPU_WORKERS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
: > "$SUMMARY"
printf "category\ttest\tresult\tdetails\n" >> "$SUMMARY"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4//$'\n'/ }" >> "$SUMMARY"; }

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Jetson Nano benchmark and health script.

Options:
  --max-perf              Enable maximum performance mode before testing.
  --duration SEC          Per-test monitor duration, default: $DURATION.
  --disk-dir PATH         Directory used for disk/SD card test, default: current dir.
  --disk-size MB          Disk test file size, default: $DISK_TEST_SIZE_MB.
  --out DIR               Output directory, default: timestamped directory.
  --cpu-workers N         CPU stress workers, default: nproc.
  --status-interval SEC   Continuous status logging interval, default: $STATUS_INTERVAL.
  --open-report           Open HTML report after completion when xdg-open exists.
  -h, --help              Show this help.

Environment overrides: OUT_DIR, DURATION, DISK_TEST_DIR, DISK_TEST_SIZE_MB,
PYTHON_BIN, RUN_MAX_PERF, CPU_WORKERS.

Examples:
  sudo ./$SCRIPT_NAME --max-perf --duration 30 --disk-dir /media/$USER/sdcard
  ./$SCRIPT_NAME --duration 15
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --max-perf) RUN_MAX_PERF=1; shift ;;
    --duration) DURATION="$2"; shift 2 ;;
    --disk-dir) DISK_TEST_DIR="$2"; shift 2 ;;
    --disk-size) DISK_TEST_SIZE_MB="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; mkdir -p "$OUT_DIR"; SUMMARY="$OUT_DIR/summary.tsv"; : > "$SUMMARY"; printf "category\ttest\tresult\tdetails\n" >> "$SUMMARY"; shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --status-interval) STATUS_INTERVAL="$2"; shift 2 ;;
    --open-report) OPEN_REPORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() {
  local category="$1" test_name="$2" cmd="$3" log_file="$OUT_DIR/${category}_${test_name}.log"
  log "Running $category/$test_name"
  if bash -lc "$cmd" > "$log_file" 2>&1; then
    record "$category" "$test_name" "PASS" "log=$log_file"
    return 0
  fi
  record "$category" "$test_name" "FAIL" "log=$log_file"
  return 1
}

start_continuous_logger() {
  STATUS_LOGGER_PID=""
  if [ -x "$STATUS_LOGGER" ]; then
    bash "$STATUS_LOGGER" --out "$OUT_DIR" --interval "$STATUS_INTERVAL" --prefix continuous_status >/dev/null 2>&1 &
    STATUS_LOGGER_PID="$!"
    record "status" "continuous_logger" "PASS" "pid=$STATUS_LOGGER_PID; interval=${STATUS_INTERVAL}s"
  else
    record "status" "continuous_logger" "SKIP" "logger not executable: $STATUS_LOGGER"
  fi
}

stop_continuous_logger() {
  if [ -n "${STATUS_LOGGER_PID:-}" ]; then
    kill "$STATUS_LOGGER_PID" >/dev/null 2>&1 || true
    wait "$STATUS_LOGGER_PID" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_monitor
  stop_continuous_logger
}

trap cleanup EXIT INT TERM

start_monitor() {
  local name="$1"
  MONITOR_LOG="$OUT_DIR/tegrastats_${name}.log"
  MONITOR_PID=""
  if have tegrastats; then
    tegrastats --interval 1000 > "$MONITOR_LOG" 2>&1 &
    MONITOR_PID="$!"
  fi
}

stop_monitor() {
  if [ -n "${MONITOR_PID:-}" ]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
}

summarize_monitor() {
  local category="$1" test_name="$2" log_file="${MONITOR_LOG:-}"
  if [ -z "$log_file" ] || [ ! -s "$log_file" ]; then
    record "$category" "${test_name}_monitor" "SKIP" "tegrastats not available"
    return
  fi
  "$PYTHON_BIN" - "$category" "$test_name" "$log_file" >> "$SUMMARY" <<'PY'
import re, statistics, sys
category, test_name, path = sys.argv[1:4]
text = open(path, errors='ignore').read().splitlines()
cpu_vals, gpu_vals, temps = [], [], []
for line in text:
    cpu_block = re.search(r'CPU \[([^\]]+)\]', line)
    if cpu_block:
        for value in re.findall(r'(\d+)%@', cpu_block.group(1)):
            cpu_vals.append(int(value))
    gpu = re.search(r'GR3D_FREQ (\d+)%', line)
    if gpu:
        gpu_vals.append(int(gpu.group(1)))
    for value in re.findall(r'([A-Za-z0-9_]+)@([0-9.]+)C', line):
        temps.append(float(value[1]))
def stat(values):
    if not values:
        return 'n/a'
    return f'avg={statistics.mean(values):.1f},max={max(values):.1f}'
thermal_risk = 'YES' if temps and max(temps) >= 80 else 'NO'
details = f'cpu_usage({stat(cpu_vals)}); gpu_usage({stat(gpu_vals)}); temp_c({stat(temps)}); thermal_risk={thermal_risk}; log={path}'
print(f'{category}\t{test_name}_monitor\tPASS\t{details}')
PY
}

check_throttle() {
  local category="$1" test_name="$2" details=""
  for path in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_cur_freq /sys/devices/gpu.0/devfreq/*/cur_freq /sys/kernel/debug/clk/override.gbus/rate; do
    [ -e "$path" ] || continue
    details="$details $path=$(cat "$path" 2>/dev/null || true)"
  done
  for path in /sys/devices/virtual/thermal/thermal_zone*/temp; do
    [ -e "$path" ] || continue
    details="$details $path=$(cat "$path" 2>/dev/null || true)"
  done
  record "$category" "${test_name}_freq_temp_snapshot" "INFO" "${details:-no sysfs freq/temp data}"
}

system_status() {
  local log_file="$OUT_DIR/system_status.log"
  {
    echo "== date =="; date
    echo "== uname =="; uname -a
    echo "== lsb_release =="; lsb_release -a 2>/dev/null || true
    echo "== jetson model =="; cat /proc/device-tree/model 2>/dev/null || true; echo
    echo "== jetpack =="; dpkg-query --show nvidia-l4t-core 2>/dev/null || true
    echo "== nvpmodel =="; nvpmodel -q 2>/dev/null || true
    echo "== jetson_clocks =="; jetson_clocks --show 2>/dev/null || true
    echo "== cpu =="; lscpu 2>/dev/null || cat /proc/cpuinfo
    echo "== memory =="; free -h
    echo "== disk =="; df -h; lsblk 2>/dev/null || true
    echo "== cuda =="; nvcc --version 2>/dev/null || true
    echo "== python =="; "$PYTHON_BIN" --version 2>&1 || true
  } > "$log_file" 2>&1
  record "status" "basic_status" "PASS" "log=$log_file"
}

enable_max_perf() {
  if [ "$RUN_MAX_PERF" != "1" ]; then
    record "performance" "max_perf" "SKIP" "use --max-perf to enable"
    return
  fi
  local log_file="$OUT_DIR/max_perf.log"
  {
    echo "Setting 10W nvpmodel when available"
    sudo nvpmodel -m 0 2>/dev/null || nvpmodel -m 0 2>/dev/null || true
    echo "Enabling jetson_clocks"
    sudo jetson_clocks 2>/dev/null || jetson_clocks 2>/dev/null || true
    nvpmodel -q 2>/dev/null || true
    jetson_clocks --show 2>/dev/null || true
  } > "$log_file" 2>&1
  record "performance" "max_perf" "PASS" "log=$log_file"
}

cpu_test() {
  local log_file="$OUT_DIR/cpu_stress.log"
  start_monitor "cpu"
  "$PYTHON_BIN" - "$DURATION" "$CPU_WORKERS" > "$log_file" 2>&1 <<'PY'
import multiprocessing as mp, sys, time
duration = float(sys.argv[1])
workers = int(sys.argv[2])
def burn(stop_time):
    value = 0
    while time.time() < stop_time:
        value = (value * 1664525 + 1013904223) & 0xffffffff
    return value
stop_time = time.time() + duration
start = time.time()
with mp.Pool(workers) as pool:
    pool.map(burn, [stop_time] * workers)
elapsed = time.time() - start
print(f'workers={workers}')
print(f'elapsed_s={elapsed:.2f}')
print('result=completed')
PY
  local rc=$?
  stop_monitor
  [ "$rc" -eq 0 ] && record "cpu" "stress" "PASS" "log=$log_file" || record "cpu" "stress" "FAIL" "log=$log_file"
  summarize_monitor "cpu" "stress"
  check_throttle "cpu" "stress"
}

memory_test() {
  local log_file="$OUT_DIR/memory_bandwidth.log"
  start_monitor "memory"
  "$PYTHON_BIN" - "$DURATION" > "$log_file" 2>&1 <<'PY'
import os, sys, time
duration = float(sys.argv[1])
size = 128 * 1024 * 1024
src = bytearray(os.urandom(1024)) * (size // 1024)
dst = bytearray(size)
loops = 0
start = time.time()
while time.time() - start < duration:
    dst[:] = src
    loops += 1
elapsed = time.time() - start
gb = loops * size / 1e9
print(f'memory_copy_gb={gb:.2f}')
print(f'elapsed_s={elapsed:.2f}')
print(f'bandwidth_gb_s={gb/elapsed:.2f}')
PY
  local rc=$?
  stop_monitor
  [ "$rc" -eq 0 ] && record "memory" "copy_bandwidth" "PASS" "log=$log_file" || record "memory" "copy_bandwidth" "FAIL" "log=$log_file"
  summarize_monitor "memory" "copy_bandwidth"
  check_throttle "memory" "copy_bandwidth"
}

disk_test() {
  local file="$DISK_TEST_DIR/jetson_disk_test_$$.bin" log_file="$OUT_DIR/disk_sdcard.log"
  start_monitor "disk"
  {
    echo "write_test_mb=$DISK_TEST_SIZE_MB"
    sync
    dd if=/dev/zero of="$file" bs=1M count="$DISK_TEST_SIZE_MB" conv=fdatasync status=progress
    sync
    echo "read_test_mb=$DISK_TEST_SIZE_MB"
    dd if="$file" of=/dev/null bs=1M status=progress
    rm -f "$file"
  } > "$log_file" 2>&1
  local rc=$?
  stop_monitor
  rm -f "$file"
  [ "$rc" -eq 0 ] && record "disk" "sdcard_read_write" "PASS" "log=$log_file" || record "disk" "sdcard_read_write" "FAIL" "log=$log_file"
  summarize_monitor "disk" "sdcard_read_write"
  check_throttle "disk" "sdcard_read_write"
}

cuda_test() {
  local log_file="$OUT_DIR/gpu_cuda.log"
  start_monitor "cuda"
  if have deviceQuery; then
    deviceQuery > "$log_file" 2>&1
  else
    "$PYTHON_BIN" - "$DURATION" > "$log_file" 2>&1 <<'PY'
import time
try:
    import torch
except Exception as exc:
    raise SystemExit(f'PyTorch unavailable: {exc}')
print(f'torch={torch.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available in PyTorch')
device = 'cuda'
a = torch.randn(512, 512, device=device)
b = torch.randn(512, 512, device=device)
torch.cuda.synchronize()
loops = 0
start = time.time()
duration = float(__import__('sys').argv[1])
while time.time() - start < duration:
    c = a @ b
    loops += 1
torch.cuda.synchronize()
elapsed = time.time() - start
ops = loops * 2 * 512**3
print(f'matmul_loops={loops}')
print(f'elapsed_s={elapsed:.2f}')
print(f'estimated_gflops={ops/elapsed/1e9:.2f}')
PY
  fi
  local rc=$?
  stop_monitor
  [ "$rc" -eq 0 ] && record "gpu" "cuda_compute" "PASS" "log=$log_file" || record "gpu" "cuda_compute" "FAIL" "log=$log_file"
  summarize_monitor "gpu" "cuda_compute"
  check_throttle "gpu" "cuda_compute"
}

bandwidth_test() {
  local log_file="$OUT_DIR/bandwidth.log"
  {
    echo "== network interfaces =="
    ip -brief addr 2>/dev/null || ifconfig 2>/dev/null || true
    echo "== link speeds =="
    for iface in /sys/class/net/*; do
      name="$(basename "$iface")"
      [ "$name" = "lo" ] && continue
      speed="$(cat "$iface/speed" 2>/dev/null || echo unknown)"
      duplex="$(cat "$iface/duplex" 2>/dev/null || echo unknown)"
      echo "$name speed_mbps=$speed duplex=$duplex"
    done
    if have iperf3; then
      echo "iperf3 is installed. Run client mode manually: iperf3 -c SERVER_IP -t $DURATION"
    else
      echo "iperf3 not installed; only interface link bandwidth was checked."
    fi
  } > "$log_file" 2>&1
  record "bandwidth" "network_link" "PASS" "log=$log_file"
}

ai_inference_test() {
  local log_file="$OUT_DIR/ai_inference.log"
  start_monitor "ai"
  "$PYTHON_BIN" - "$DURATION" > "$log_file" 2>&1 <<'PY'
import time, sys
duration = float(sys.argv[1])
try:
    import torch
    import torchvision.models as models
except Exception as exc:
    raise SystemExit(f'PyTorch/torchvision unavailable: {exc}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet18(weights=None).eval().to(device)
x = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    for _ in range(5):
        y = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    frames = 0
    start = time.time()
    while time.time() - start < duration:
        y = model(x)
        frames += 1
    if device == 'cuda':
        torch.cuda.synchronize()
elapsed = time.time() - start
top1 = int(y.argmax(dim=1).item())
print(f'test_type=image_classification')
print(f'model=resnet18_random_weights')
print(f'device={device}')
print(f'frames={frames}')
print(f'elapsed_s={elapsed:.2f}')
print(f'fps={frames/elapsed:.2f}')
print(f'top1_class_index={top1}')
PY
  local rc=$?
  stop_monitor
  [ "$rc" -eq 0 ] && record "ai" "classification_fps" "PASS" "log=$log_file" || record "ai" "classification_fps" "FAIL" "log=$log_file"
  summarize_monitor "ai" "classification_fps"
  check_throttle "ai" "classification_fps"
}

generate_report() {
  local report="$OUT_DIR/report.html"
  "$PYTHON_BIN" - "$SUMMARY" "$OUT_DIR" "$report" <<'PY'
import csv, html, os, re, sys
from datetime import datetime

summary_path, out_dir, report_path = sys.argv[1:4]
rows = []
with open(summary_path, newline='', errors='ignore') as handle:
    for row in csv.DictReader(handle, delimiter='\t'):
        rows.append(row)

def extract_metric(details, name):
    match = re.search(rf'{re.escape(name)}=([^;\s]+)', details)
    return match.group(1) if match else ''

cards = []
for row in rows:
    details = row.get('details', '')
    if row.get('test') == 'classification_fps':
        log_match = re.search(r'log=([^\s]+)', details)
        fps = ''
        if log_match and os.path.exists(log_match.group(1)):
            text = open(log_match.group(1), errors='ignore').read()
            fps_match = re.search(r'^fps=([0-9.]+)', text, re.M)
            device_match = re.search(r'^device=(\S+)', text, re.M)
            fps = fps_match.group(1) if fps_match else ''
            device = device_match.group(1) if device_match else ''
            cards.append(('AI FPS', fps or 'N/A', f'classification on {device or "unknown"}'))
    if row.get('test', '').endswith('_monitor'):
        thermal = extract_metric(details, 'thermal_risk')
        if thermal:
            cards.append((f'{row.get("category", "")}: Thermal Risk', thermal, row.get('test', '')))

status_counts = {}
for row in rows:
    status_counts[row.get('result', 'UNKNOWN')] = status_counts.get(row.get('result', 'UNKNOWN'), 0) + 1

logs = sorted(name for name in os.listdir(out_dir) if name.endswith(('.log', '.tsv')))

def esc(value):
    return html.escape(str(value), quote=True)

summary_rows = '\n'.join(
    '<tr>'
    f'<td>{esc(row.get("category", ""))}</td>'
    f'<td>{esc(row.get("test", ""))}</td>'
    f'<td><span class="badge {esc(row.get("result", "").lower())}">{esc(row.get("result", ""))}</span></td>'
    f'<td>{esc(row.get("details", ""))}</td>'
    '</tr>'
    for row in rows
)

card_html = '\n'.join(
    f'<article class="card"><div class="label">{esc(title)}</div><div class="value">{esc(value)}</div><p>{esc(desc)}</p></article>'
    for title, value, desc in cards[:12]
)
if not card_html:
    card_html = '<article class="card"><div class="label">Report</div><div class="value">Ready</div><p>See the summary table and logs.</p></article>'

log_links = '\n'.join(f'<li><a href="{esc(name)}">{esc(name)}</a></li>' for name in logs)
status_html = ', '.join(f'{esc(k)}: {v}' for k, v in sorted(status_counts.items()))

document = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jetson Nano Benchmark Report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101622; --panel:#172033; --text:#e9eef8; --muted:#9cadc9; --ok:#43d17a; --bad:#ff6b6b; --warn:#ffd166; --line:#29364f; }}
    body {{ margin:0; font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; background:linear-gradient(135deg,#0b1020,#18223a); color:var(--text); }}
    main {{ max-width:1180px; margin:0 auto; padding:32px 18px 48px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    .meta {{ color:var(--muted); margin-bottom:24px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:24px; }}
    .card {{ background:rgba(23,32,51,.92); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 12px 30px rgba(0,0,0,.22); }}
    .label {{ color:var(--muted); font-size:13px; }}
    .value {{ font-size:28px; font-weight:750; margin:4px 0; }}
    .card p {{ margin:0; color:var(--muted); }}
    section {{ background:rgba(23,32,51,.9); border:1px solid var(--line); border-radius:16px; padding:18px; margin-top:18px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:650; }}
    .badge {{ display:inline-block; min-width:48px; text-align:center; padding:2px 8px; border-radius:999px; background:#344057; color:var(--text); font-weight:700; }}
    .pass {{ background:rgba(67,209,122,.16); color:var(--ok); }}
    .fail {{ background:rgba(255,107,107,.16); color:var(--bad); }}
    .skip,.info {{ background:rgba(255,209,102,.16); color:var(--warn); }}
    a {{ color:#8cc8ff; }}
    ul {{ columns:2; padding-left:20px; }}
    @media (max-width:700px) {{ h1 {{ font-size:24px; }} ul {{ columns:1; }} main {{ padding:22px 12px; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Jetson Nano Benchmark Report</h1>
    <div class="meta">Generated: {esc(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | Output: {esc(out_dir)} | Status: {status_html}</div>
    <div class="grid">{card_html}</div>
    <section><h2>Summary</h2><table><thead><tr><th>Category</th><th>Test</th><th>Result</th><th>Details</th></tr></thead><tbody>{summary_rows}</tbody></table></section>
    <section><h2>Saved Files</h2><ul>{log_links}</ul></section>
  </main>
</body>
</html>
'''

with open(report_path, 'w', encoding='utf-8') as handle:
    handle.write(document)
print(report_path)
PY
  record "report" "html_report" "PASS" "log=$report"
  log "HTML report: $report"
  if [ "${OPEN_REPORT:-0}" = "1" ] && have xdg-open; then
    xdg-open "$report" >/dev/null 2>&1 || true
  fi
}

log "Output directory: $OUT_DIR"
start_continuous_logger
system_status
enable_max_perf
cpu_test
memory_test
disk_test
cuda_test
bandwidth_test
ai_inference_test
generate_report

log "Benchmark finished. Summary: $SUMMARY"
if have column; then
  column -t -s $'\t' "$SUMMARY"
else
  cat "$SUMMARY"
fi

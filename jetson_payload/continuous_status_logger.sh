#!/usr/bin/env bash
set -u

OUT_DIR="${OUT_DIR:-status_logs_$(date +%Y%m%d_%H%M%S)}"
INTERVAL="${INTERVAL:-1}"
PREFIX="${PREFIX:-system_status}"
mkdir -p "$OUT_DIR"

RAW_LOG="$OUT_DIR/${PREFIX}_raw.log"
METRICS_TSV="$OUT_DIR/${PREFIX}_metrics.tsv"
EVENT_LOG="$OUT_DIR/${PREFIX}_events.log"

have() { command -v "$1" >/dev/null 2>&1; }
log_event() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$EVENT_LOG"; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Continuously log Jetson status until stopped with Ctrl+C or SIGTERM.

Options:
  --out DIR        Output directory, default: $OUT_DIR.
  --interval SEC   Sampling interval, default: $INTERVAL.
  --prefix NAME    Log file prefix, default: $PREFIX.
  -h, --help       Show this help.

Outputs:
  ${PREFIX}_raw.log      Human-readable status snapshots.
  ${PREFIX}_metrics.tsv  Timestamped numeric metrics.
  ${PREFIX}_events.log   Start/stop events.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out) OUT_DIR="$2"; mkdir -p "$OUT_DIR"; RAW_LOG="$OUT_DIR/${PREFIX}_raw.log"; METRICS_TSV="$OUT_DIR/${PREFIX}_metrics.tsv"; EVENT_LOG="$OUT_DIR/${PREFIX}_events.log"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; RAW_LOG="$OUT_DIR/${PREFIX}_raw.log"; METRICS_TSV="$OUT_DIR/${PREFIX}_metrics.tsv"; EVENT_LOG="$OUT_DIR/${PREFIX}_events.log"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

printf 'timestamp\tuptime_s\tload1\tload5\tload15\tmem_total_kb\tmem_available_kb\tswap_total_kb\tswap_free_kb\tcpu_avg_percent\tgpu_percent\ttemp_max_c\tcpu_freq_max_khz\tgpu_freq_hz\tdisk_used_percent\n' > "$METRICS_TSV"

stop_requested=0
trap 'stop_requested=1; log_event "stop requested"' INT TERM

read_meminfo() {
  awk '
    $1=="MemTotal:" {mt=$2}
    $1=="MemAvailable:" {ma=$2}
    $1=="SwapTotal:" {st=$2}
    $1=="SwapFree:" {sf=$2}
    END {printf "%s\t%s\t%s\t%s", mt+0, ma+0, st+0, sf+0}
  ' /proc/meminfo 2>/dev/null || printf '0\t0\t0\t0'
}

read_load() {
  awk '{printf "%s\t%s\t%s", $1, $2, $3}' /proc/loadavg 2>/dev/null || printf '0\t0\t0'
}

read_uptime() {
  awk '{printf "%.0f", $1}' /proc/uptime 2>/dev/null || printf '0'
}

read_cpu_avg() {
  if have mpstat; then
    mpstat 1 1 2>/dev/null | awk '/Average:/ && $2=="all" {printf "%.1f", 100-$NF; found=1} END {if (!found) printf ""}'
    return
  fi
  printf ''
}

read_gpu_percent() {
  if have tegrastats; then
    timeout 2s tegrastats --interval 1000 2>/dev/null | sed -n 's/.*GR3D_FREQ \([0-9][0-9]*\)%.*/\1/p' | tail -n 1
    return
  fi
  printf ''
}

read_temp_max() {
  local max=""
  for path in /sys/devices/virtual/thermal/thermal_zone*/temp; do
    [ -e "$path" ] || continue
    value="$(cat "$path" 2>/dev/null || true)"
    [ -n "$value" ] || continue
    if [ "$value" -gt 1000 ] 2>/dev/null; then
      value="$(awk "BEGIN {printf \"%.1f\", $value/1000}")"
    fi
    if [ -z "$max" ] || awk "BEGIN {exit !($value > $max)}"; then
      max="$value"
    fi
  done
  printf '%s' "$max"
}

read_cpu_freq_max() {
  local max=0
  for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_cur_freq; do
    [ -e "$path" ] || continue
    value="$(cat "$path" 2>/dev/null || echo 0)"
    [ "$value" -gt "$max" ] 2>/dev/null && max="$value"
  done
  printf '%s' "$max"
}

read_gpu_freq() {
  for path in /sys/devices/gpu.0/devfreq/*/cur_freq /sys/kernel/debug/clk/override.gbus/rate; do
    [ -e "$path" ] || continue
    cat "$path" 2>/dev/null && return
  done
  printf ''
}

read_disk_used_percent() {
  df -P "$OUT_DIR" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); printf "%s", $5}' || printf ''
}

write_raw_snapshot() {
  {
    echo "===== $(date '+%F %T') ====="
    echo "== uptime/load =="; uptime 2>/dev/null || true
    echo "== memory =="; free -h 2>/dev/null || true
    echo "== disk =="; df -h "$OUT_DIR" 2>/dev/null || true
    echo "== cpu freq =="
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_cur_freq; do [ -e "$path" ] && echo "$path=$(cat "$path" 2>/dev/null || true)"; done
    echo "== gpu freq =="
    for path in /sys/devices/gpu.0/devfreq/*/cur_freq /sys/kernel/debug/clk/override.gbus/rate; do [ -e "$path" ] && echo "$path=$(cat "$path" 2>/dev/null || true)"; done
    echo "== thermal =="
    for path in /sys/devices/virtual/thermal/thermal_zone*/temp; do [ -e "$path" ] && echo "$path=$(cat "$path" 2>/dev/null || true)"; done
    if have tegrastats; then
      echo "== tegrastats =="
      timeout 2s tegrastats --interval 1000 2>/dev/null || true
    fi
  } >> "$RAW_LOG" 2>&1
}

log_event "started out_dir=$OUT_DIR interval=$INTERVAL prefix=$PREFIX"
while [ "$stop_requested" -eq 0 ]; do
  timestamp="$(date '+%F %T')"
  uptime_s="$(read_uptime)"
  load_values="$(read_load)"
  mem_values="$(read_meminfo)"
  cpu_avg="$(read_cpu_avg)"
  gpu_percent="$(read_gpu_percent)"
  temp_max="$(read_temp_max)"
  cpu_freq="$(read_cpu_freq_max)"
  gpu_freq="$(read_gpu_freq)"
  disk_used="$(read_disk_used_percent)"
  load1="$(printf '%s' "$load_values" | awk '{print $1}')"
  load5="$(printf '%s' "$load_values" | awk '{print $2}')"
  load15="$(printf '%s' "$load_values" | awk '{print $3}')"
  mem_total="$(printf '%s' "$mem_values" | awk '{print $1}')"
  mem_available="$(printf '%s' "$mem_values" | awk '{print $2}')"
  swap_total="$(printf '%s' "$mem_values" | awk '{print $3}')"
  swap_free="$(printf '%s' "$mem_values" | awk '{print $4}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$timestamp" "$uptime_s" "${load1:-}" "${load5:-}" "${load15:-}" \
    "${mem_total:-}" "${mem_available:-}" "${swap_total:-}" "${swap_free:-}" \
    "${cpu_avg:-}" "${gpu_percent:-}" "${temp_max:-}" "${cpu_freq:-}" "${gpu_freq:-}" "${disk_used:-}" >> "$METRICS_TSV"
  write_raw_snapshot
  sleep "$INTERVAL" || break
done
log_event "stopped"

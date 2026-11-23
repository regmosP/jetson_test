#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Continuously run an ONNX model on GPU and report status/performance.")
    parser.add_argument("--onnx", required=True, help="Path to the .onnx model.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: testgpu_runs/run_TIMESTAMP.")
    parser.add_argument("--provider", choices=("auto", "cuda", "tensorrt"), default="auto", help="GPU provider to use.")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow CPUExecutionProvider fallback.")
    parser.add_argument("--iterations", type=int, default=0, help="Number of measured iterations. 0 means run until stopped.")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Run duration in seconds. 0 means no duration limit.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before measurement.")
    parser.add_argument("--report-interval", type=float, default=5.0, help="Seconds between live status reports.")
    parser.add_argument("--input-shape", action="append", default=[], help="Override input shape, format: name:1,3,224,224. Can be repeated.")
    parser.add_argument("--input-scale", type=float, default=1.0, help="Scale for generated floating point random inputs.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for generated inputs.")
    return parser.parse_args()


def import_required():
    missing = []
    try:
        import numpy as np
    except Exception as exc:
        missing.append(f"numpy: {exc}")
        np = None
    try:
        import onnxruntime as ort
    except Exception as exc:
        missing.append(f"onnxruntime: {exc}")
        ort = None
    if missing:
        raise SystemExit("Missing required Python modules:\n" + "\n".join(missing))
    return np, ort


def default_out_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("testgpu_runs", f"run_{timestamp}")


def parse_shape_overrides(values):
    overrides = {}
    for value in values:
        if ":" not in value:
            raise SystemExit(f"Invalid --input-shape value: {value}")
        name, raw_shape = value.split(":", 1)
        dims = []
        for raw_dim in raw_shape.split(","):
            raw_dim = raw_dim.strip()
            if not raw_dim:
                raise SystemExit(f"Invalid --input-shape value: {value}")
            dims.append(int(raw_dim))
        overrides[name.strip()] = dims
    return overrides


def choose_providers(ort, requested, allow_cpu_fallback):
    available = ort.get_available_providers()
    providers = []
    if requested in ("auto", "tensorrt") and "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")
    if requested in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if requested == "tensorrt" and "TensorrtExecutionProvider" not in providers:
        raise SystemExit("TensorrtExecutionProvider is not available in this ONNX Runtime installation.")
    if requested == "cuda" and "CUDAExecutionProvider" not in providers:
        raise SystemExit("CUDAExecutionProvider is not available in this ONNX Runtime installation.")
    if not providers:
        raise SystemExit("No GPU ONNX Runtime provider is available. Install/configure onnxruntime-gpu first.")
    if allow_cpu_fallback:
        providers.append("CPUExecutionProvider")
    return providers, available


def normalize_shape(shape, override):
    if override is not None:
        return override
    normalized = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        else:
            normalized.append(1)
    return normalized


def numpy_dtype(np, type_name):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    if type_name not in mapping:
        raise SystemExit(f"Unsupported input tensor type: {type_name}")
    return mapping[type_name]


def build_inputs(np, session, overrides, input_scale, seed):
    rng = np.random.default_rng(seed)
    feeds = {}
    metadata = []
    for input_info in session.get_inputs():
        shape = normalize_shape(input_info.shape, overrides.get(input_info.name))
        dtype = numpy_dtype(np, input_info.type)
        if np.issubdtype(dtype, np.floating):
            value = (rng.random(shape, dtype=np.float32) * input_scale).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 2, size=shape, dtype=dtype)
        elif dtype == np.bool_:
            value = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            raise SystemExit(f"Unsupported input dtype: {dtype}")
        feeds[input_info.name] = value
        metadata.append({"name": input_info.name, "type": input_info.type, "shape": shape})
    return feeds, metadata


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percent / 100.0))
    return ordered[index]


def latency_stats(values):
    if not values:
        return {"min_ms": None, "avg_ms": None, "p50_ms": None, "p90_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "min_ms": min(values),
        "avg_ms": statistics.mean(values),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def read_system_gpu_status():
    if shutil.which("nvidia-smi"):
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2)
            first_line = completed.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in first_line.split(",")]
            return {
                "source": "nvidia-smi",
                "gpu_util_percent": float(parts[0]),
                "gpu_mem_used_mb": float(parts[1]),
                "gpu_mem_total_mb": float(parts[2]),
                "gpu_temp_c": float(parts[3]),
                "gpu_power_w": float(parts[4]) if parts[4] != "[Not Supported]" else None,
            }
        except Exception:
            return {"source": "nvidia-smi", "error": "failed_to_read"}
    if shutil.which("tegrastats"):
        try:
            completed = subprocess.run(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=1.5)
            return {"source": "tegrastats", "raw": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""}
        except Exception:
            return {"source": "tegrastats", "error": "failed_to_read"}
    return {"source": "none"}


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    np, ort = import_required()
    if not os.path.exists(args.onnx):
        raise SystemExit(f"ONNX model not found: {args.onnx}")

    out_dir = args.out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, "status.jsonl")
    metrics_path = os.path.join(out_dir, "iteration_metrics.csv")
    summary_path = os.path.join(out_dir, "summary.json")
    config_path = os.path.join(out_dir, "run_config.json")

    providers, available_providers = choose_providers(ort, args.provider, args.allow_cpu_fallback)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 2
    session = ort.InferenceSession(args.onnx, sess_options=session_options, providers=providers)
    active_providers = session.get_providers()
    gpu_active = any(provider in active_providers for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"))
    if not gpu_active:
        raise SystemExit(f"No GPU provider is active. Active providers: {active_providers}")

    overrides = parse_shape_overrides(args.input_shape)
    feeds, input_metadata = build_inputs(np, session, overrides, args.input_scale, args.seed)
    output_names = [output.name for output in session.get_outputs()]

    config = {
        "onnx": os.path.abspath(args.onnx),
        "out_dir": os.path.abspath(out_dir),
        "requested_provider": args.provider,
        "available_providers": available_providers,
        "active_providers": active_providers,
        "allow_cpu_fallback": args.allow_cpu_fallback,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "duration_sec": args.duration_sec,
        "report_interval": args.report_interval,
        "inputs": input_metadata,
        "outputs": output_names,
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(config_path, config)

    for _ in range(max(args.warmup, 0)):
        session.run(output_names, feeds)

    latencies = []
    iteration = 0
    started = time.perf_counter()
    last_report = started
    last_count = 0
    status = "running"

    print(f"status=running provider={','.join(active_providers)} out_dir={out_dir}", flush=True)

    with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file, open(status_path, "a", encoding="utf-8") as status_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "timestamp", "latency_ms", "fps"])
        writer.writeheader()
        try:
            while True:
                if args.iterations > 0 and iteration >= args.iterations:
                    status = "completed"
                    break
                elapsed_total = time.perf_counter() - started
                if args.duration_sec > 0 and elapsed_total >= args.duration_sec:
                    status = "completed"
                    break

                inference_started = time.perf_counter()
                session.run(output_names, feeds)
                latency_ms = (time.perf_counter() - inference_started) * 1000.0
                iteration += 1
                latencies.append(latency_ms)
                writer.writerow({
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "latency_ms": f"{latency_ms:.6f}",
                    "fps": f"{1000.0 / latency_ms:.6f}" if latency_ms > 0 else "",
                })

                now = time.perf_counter()
                if now - last_report >= args.report_interval:
                    window_elapsed = now - last_report
                    window_count = iteration - last_count
                    total_elapsed = now - started
                    record = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "status": "running",
                        "iterations": iteration,
                        "elapsed_sec": total_elapsed,
                        "window_fps": window_count / window_elapsed if window_elapsed > 0 else None,
                        "overall_fps": iteration / total_elapsed if total_elapsed > 0 else None,
                        "latency": latency_stats(latencies),
                        "gpu_status": read_system_gpu_status(),
                    }
                    status_file.write(json.dumps(record, sort_keys=True) + "\n")
                    status_file.flush()
                    metrics_file.flush()
                    avg_ms = record["latency"]["avg_ms"]
                    print(
                        f"status=running iterations={iteration} overall_fps={record['overall_fps']:.2f} avg_latency_ms={avg_ms:.3f}",
                        flush=True,
                    )
                    last_report = now
                    last_count = iteration
        except KeyboardInterrupt:
            status = "stopped"

    elapsed = time.perf_counter() - started
    summary = {
        "status": status,
        "iterations": iteration,
        "elapsed_sec": elapsed,
        "overall_fps": iteration / elapsed if elapsed > 0 else None,
        "latency": latency_stats(latencies),
        "active_providers": active_providers,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "run_config": config_path,
            "status_jsonl": status_path,
            "iteration_metrics_csv": metrics_path,
            "summary_json": summary_path,
        },
    }
    write_json(summary_path, summary)
    avg_ms = summary["latency"]["avg_ms"]
    avg_text = f"{avg_ms:.3f}" if avg_ms is not None else "n/a"
    fps_text = f"{summary['overall_fps']:.2f}" if summary["overall_fps"] is not None else "n/a"
    print(f"status={status} iterations={iteration} overall_fps={fps_text} avg_latency_ms={avg_text} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()

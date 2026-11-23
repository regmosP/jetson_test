#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_tegrastats(path):
    if not path or not os.path.exists(path):
        return {"available": False, "details": "tegrastats.log not found"}
    cpu_values = []
    gpu_values = []
    temperatures = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            cpu = re.search(r"CPU \[([^\]]+)\]", line)
            if cpu:
                cpu_values.extend(int(value) for value in re.findall(r"(\d+)%@", cpu.group(1)))
            gpu = re.search(r"GR3D_FREQ (\d+)%", line)
            if gpu:
                gpu_values.append(int(gpu.group(1)))
            temperatures.extend(float(value) for value in re.findall(r"@[0-9]+/([0-9.]+)C", line))
            temperatures.extend(float(value) for value in re.findall(r"[A-Za-z0-9_]+@([0-9.]+)C", line))

    def stat(values):
        if not values:
            return {"avg": None, "max": None}
        return {"avg": statistics.mean(values), "max": max(values)}

    temp_stat = stat(temperatures)
    return {
        "available": True,
        "cpu_usage_percent": stat(cpu_values),
        "gpu_usage_percent": stat(gpu_values),
        "temperature_c": temp_stat,
        "thermal_risk": bool(temp_stat["max"] is not None and temp_stat["max"] >= 80.0),
    }


def fmt(value, digits=3):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO TensorRT pipeline results")
    parser.add_argument("--result-json", required=True, help="Path to tensorrt_pipeline_result.json")
    parser.add_argument("--tegrastats", default="", help="Path to tegrastats.log")
    parser.add_argument("--out", default="", help="Output markdown report path")
    args = parser.parse_args()

    result = read_json(args.result_json)
    summary = result["benchmark"]["summary"]
    pipeline = result["pipeline"]
    parameters = result["parameters"]
    tegra = parse_tegrastats(args.tegrastats)
    out_path = args.out or os.path.join(os.path.dirname(args.result_json), "yolo_result_analysis.md")

    lines = [
        "# YOLO TensorRT Result Analysis",
        "",
        "## Environment And Model",
        "",
        f"- TensorRT version: `{parameters.get('tensorrt_version', 'N/A')}`",
        f"- Python version: `{parameters.get('python', 'N/A')}`",
        f"- ONNX path: `{pipeline['onnx'].get('path', 'N/A')}`",
        f"- Engine path: `{pipeline['engine'].get('path', 'N/A')}`",
        f"- Precision: `{parameters.get('precision')}`",
        f"- Batch size: `{parameters.get('batch_size')}`",
        f"- Input shape: `{parameters.get('batch_size')}x3x{parameters.get('input_hw')[0]}x{parameters.get('input_hw')[1]}`",
        f"- Input source: `{result['benchmark'].get('input_info', [{}])[0].get('source', 'unknown')}`",
        "",
        "## Pipeline Timing",
        "",
        f"- ONNX export/prepare time: `{fmt(pipeline['onnx'].get('elapsed_s'))} s`",
        f"- TensorRT engine build time: `{fmt(pipeline['engine'].get('elapsed_s'))} s`",
        f"- Engine size: `{pipeline['engine'].get('engine_bytes', 'N/A')} bytes`",
        "",
        "## Performance Summary",
        "",
        f"- GPU compute FPS: `{fmt(summary.get('inferences_per_second_gpu_avg'), 2)}`",
        f"- End-to-end FPS: `{fmt(summary.get('inferences_per_second_end_to_end_avg'), 2)}`",
        f"- Single inference GPU compute latency avg: `{fmt(summary['gpu_compute_ms'].get('avg'))} ms`",
        f"- Single inference GPU compute latency p95: `{fmt(summary['gpu_compute_ms'].get('p95'))} ms`",
        f"- Single inference end-to-end latency avg: `{fmt(summary['end_to_end_ms'].get('avg'))} ms`",
        f"- Single inference end-to-end latency p95: `{fmt(summary['end_to_end_ms'].get('p95'))} ms`",
        f"- Host-to-device copy avg: `{fmt(summary['host_to_device_ms'].get('avg'))} ms`",
        f"- Device-to-host copy avg: `{fmt(summary['device_to_host_ms'].get('avg'))} ms`",
        "",
        "## GPU And Thermal",
        "",
    ]

    if tegra["available"]:
        lines.extend([
            f"- CPU usage avg/max: `{fmt(tegra['cpu_usage_percent']['avg'], 1)}% / {fmt(tegra['cpu_usage_percent']['max'], 1)}%`",
            f"- GPU usage avg/max: `{fmt(tegra['gpu_usage_percent']['avg'], 1)}% / {fmt(tegra['gpu_usage_percent']['max'], 1)}%`",
            f"- Temperature avg/max: `{fmt(tegra['temperature_c']['avg'], 1)}C / {fmt(tegra['temperature_c']['max'], 1)}C`",
            f"- Thermal throttling risk: `{'YES' if tegra['thermal_risk'] else 'NO'}`",
        ])
    else:
        lines.append(f"- `{tegra['details']}`")

    lines.extend([
        "",
        "## Result Interpretation",
        "",
        "- `GPU compute FPS` only measures TensorRT execution on GPU by CUDA events.",
        "- `End-to-end FPS` includes input copy, TensorRT execution, output copy, and synchronization.",
        "- If end-to-end latency is much higher than GPU compute latency, host/device copies or synchronization dominate.",
        "- If temperature max is near or above 80C, add cooling or reduce run duration before comparing results.",
    ])

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()

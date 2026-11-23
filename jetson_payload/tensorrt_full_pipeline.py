#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime


def import_required():
    missing = []
    modules = {}
    for name in ("numpy", "tensorrt", "pycuda.driver", "pycuda.autoinit"):
        try:
            modules[name] = __import__(name, fromlist=["*"])
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise SystemExit("Missing required modules:\n" + "\n".join(missing))
    return modules["numpy"], modules["tensorrt"], modules["pycuda.driver"]


class TrtLogger:
    def __init__(self, trt, verbose=False):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percent / 100.0))
    return ordered[index]


def stats_ms(values):
    if not values:
        return {"min": None, "avg": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "min": min(values),
        "avg": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def run_command(command, log_path):
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("command=" + " ".join(command) + "\n")
        log_file.flush()
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return {"command": command, "returncode": completed.returncode, "elapsed_s": time.time() - started, "log": log_path}


def maybe_generate_onnx(args, np):
    if args.onnx and os.path.exists(args.onnx):
        return {"status": "provided", "path": args.onnx, "elapsed_s": 0.0, "details": "Using existing ONNX model"}
    if not args.generate_onnx:
        raise SystemExit("Provide --onnx PATH or enable --generate-onnx")

    onnx_path = args.onnx or os.path.join(args.out_dir, "generated_convnet.onnx")
    started = time.time()
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise SystemExit(f"--generate-onnx requires torch: {exc}")

    class SmallConvNet(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, classes),
            )

        def forward(self, value):
            return self.net(value)

    height, width = args.input_hw
    model = SmallConvNet(args.classes).eval()
    dummy = torch.randn(args.batch_size, 3, height, width)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=[args.input_name],
        output_names=[args.output_name],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    return {"status": "generated", "path": onnx_path, "elapsed_s": time.time() - started, "details": "Small ConvNet classification ONNX"}


def network_flags(trt, explicit_batch):
    if explicit_batch and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        return 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    return 0


def set_config_memory(trt, config, workspace_mb):
    bytes_value = int(workspace_mb) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_value)
    else:
        config.max_workspace_size = bytes_value


def add_precision_flags(trt, builder, config, precision):
    requested = []
    precision = precision.lower()
    if precision in ("fp16", "int8"):
        requested.append("fp16")
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        requested.append("int8")
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            raise SystemExit("INT8 requested but platform_has_fast_int8 is false. Calibration is not implemented in this standalone test.")
    return requested


def build_engine(args, trt):
    if args.engine and os.path.exists(args.engine) and not args.rebuild:
        return {"status": "provided", "path": args.engine, "elapsed_s": 0.0, "details": "Using existing TensorRT engine"}

    engine_path = args.engine or os.path.join(args.out_dir, "model.trt")
    logger = TrtLogger(trt, args.verbose).logger
    started = time.time()
    builder = trt.Builder(logger)
    explicit_batch = True
    network = builder.create_network(network_flags(trt, explicit_batch))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as onnx_file:
        if not parser.parse(onnx_file.read()):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise SystemExit("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    set_config_memory(trt, config, args.workspace_mb)
    precision_flags = add_precision_flags(trt, builder, config, args.precision)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise SystemExit("TensorRT build_serialized_network returned None")
        engine_bytes = bytes(serialized)
    else:
        builder.max_batch_size = args.batch_size
        engine = builder.build_engine(network, config)
        if engine is None:
            raise SystemExit("TensorRT build_engine returned None")
        engine_bytes = engine.serialize()

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as handle:
        handle.write(engine_bytes)

    return {
        "status": "built",
        "path": engine_path,
        "elapsed_s": time.time() - started,
        "precision_flags": precision_flags,
        "workspace_mb": args.workspace_mb,
        "engine_bytes": len(engine_bytes),
    }


def load_engine(engine_path, trt, verbose):
    logger = TrtLogger(trt, verbose).logger
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as handle:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise SystemExit("Failed to deserialize TensorRT engine")
    return engine


def binding_names(engine):
    if hasattr(engine, "num_bindings"):
        return [engine.get_binding_name(index) for index in range(engine.num_bindings)]
    return [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]


def binding_index(engine, name):
    if hasattr(engine, "get_binding_index"):
        return engine.get_binding_index(name)
    return None


def is_input(engine, name):
    if hasattr(engine, "binding_is_input"):
        return engine.binding_is_input(binding_index(engine, name))
    import tensorrt as trt
    return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT


def tensor_shape(engine, context, name):
    if hasattr(engine, "get_binding_shape"):
        return tuple(context.get_binding_shape(binding_index(engine, name)))
    return tuple(context.get_tensor_shape(name))


def tensor_dtype(engine, trt, name):
    if hasattr(engine, "get_binding_dtype"):
        return trt.nptype(engine.get_binding_dtype(binding_index(engine, name)))
    return trt.nptype(engine.get_tensor_dtype(name))


def set_input_shapes(args, engine, context):
    height, width = args.input_hw
    wanted = (args.batch_size, 3, height, width)
    for name in binding_names(engine):
        if not is_input(engine, name):
            continue
        current = tensor_shape(engine, context, name)
        if any(dim < 0 for dim in current):
            if hasattr(context, "set_binding_shape"):
                context.set_binding_shape(binding_index(engine, name), wanted)
            else:
                context.set_input_shape(name, wanted)


def load_repeated_image(args, np, dtype, expected_size):
    if not args.image:
        rng = np.random.default_rng(args.seed)
        return rng.standard_normal(expected_size).astype(dtype, copy=False), {"source": "random", "seed": args.seed}

    height, width = args.input_hw
    image = None
    backend = ""
    try:
        from PIL import Image
        image = Image.open(args.image).convert("RGB").resize((width, height))
        image = np.asarray(image).astype("float32")
        backend = "Pillow"
    except Exception as pil_exc:
        try:
            import cv2
            raw = cv2.imread(args.image, cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError("cv2.imread returned None")
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            raw = cv2.resize(raw, (width, height))
            image = raw.astype("float32")
            backend = "OpenCV"
        except Exception as cv_exc:
            raise SystemExit(f"Failed to load --image with Pillow ({pil_exc}) or OpenCV ({cv_exc})")

    if args.input_scale:
        image *= args.input_scale
    mean = np.array(args.input_mean, dtype="float32").reshape(1, 1, 3)
    std = np.array(args.input_std, dtype="float32").reshape(1, 1, 3)
    image = (image - mean) / std
    chw = np.transpose(image, (2, 0, 1))
    batched = np.repeat(chw[np.newaxis, :, :, :], args.batch_size, axis=0)
    flattened = batched.astype(dtype, copy=False).reshape(-1)
    if flattened.size != expected_size:
        raise SystemExit(f"Image input size mismatch: got {flattened.size}, expected {expected_size}")
    return flattened, {
        "source": "image",
        "path": args.image,
        "backend": backend,
        "layout": "NCHW_RGB",
        "height": height,
        "width": width,
        "batch_size": args.batch_size,
        "input_scale": args.input_scale,
        "input_mean": args.input_mean,
        "input_std": args.input_std,
        "repeat_same_image_each_iteration": True,
    }


def allocate_buffers(args, np, trt, cuda, engine, context):
    stream = cuda.Stream()
    names = binding_names(engine)
    set_input_shapes(args, engine, context)
    buffers = []
    bindings = [0] * len(names)
    tensor_addresses = {}
    input_info = []

    for index, name in enumerate(names):
        shape = tensor_shape(engine, context, name)
        dtype = tensor_dtype(engine, trt, name)
        size = int(np.prod(shape))
        host = cuda.pagelocked_empty(size, dtype)
        if is_input(engine, name):
            values, info = load_repeated_image(args, np, dtype, size)
            host[:] = values
            input_info.append({"binding": name, **info})
        device = cuda.mem_alloc(host.nbytes)
        buffers.append({"name": name, "shape": shape, "dtype": str(dtype), "size": size, "host": host, "device": device, "is_input": is_input(engine, name)})
        bindings[index] = int(device)
        tensor_addresses[name] = int(device)
        if hasattr(context, "set_tensor_address"):
            context.set_tensor_address(name, int(device))
    return stream, buffers, bindings, tensor_addresses, input_info


def execute_once(context, stream, bindings):
    if hasattr(context, "execute_async_v2"):
        return context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    return context.execute_async_v3(stream_handle=stream.handle)


def benchmark(args, np, trt, cuda, engine):
    context = engine.create_execution_context()
    stream, buffers, bindings, _, input_info = allocate_buffers(args, np, trt, cuda, engine, context)

    input_buffers = [item for item in buffers if item["is_input"]]
    output_buffers = [item for item in buffers if not item["is_input"]]

    for item in input_buffers:
        cuda.memcpy_htod_async(item["device"], item["host"], stream)
    stream.synchronize()

    for _ in range(args.warmup):
        execute_once(context, stream, bindings)
    stream.synchronize()

    rows = []
    gpu_times = []
    end_to_end_times = []
    h2d_times = []
    d2h_times = []

    for iteration in range(args.iterations):
        e2e_started = time.perf_counter()
        h2d_started = time.perf_counter()
        for item in input_buffers:
            cuda.memcpy_htod_async(item["device"], item["host"], stream)
        stream.synchronize()
        h2d_ms = (time.perf_counter() - h2d_started) * 1000.0

        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record(stream)
        ok = execute_once(context, stream, bindings)
        end_event.record(stream)
        stream.synchronize()
        if not ok:
            raise SystemExit("TensorRT execution returned false")
        gpu_ms = start_event.time_till(end_event)

        d2h_started = time.perf_counter()
        for item in output_buffers:
            cuda.memcpy_dtoh_async(item["host"], item["device"], stream)
        stream.synchronize()
        d2h_ms = (time.perf_counter() - d2h_started) * 1000.0
        e2e_ms = (time.perf_counter() - e2e_started) * 1000.0

        gpu_times.append(gpu_ms)
        end_to_end_times.append(e2e_ms)
        h2d_times.append(h2d_ms)
        d2h_times.append(d2h_ms)
        rows.append({
            "iteration": iteration + 1,
            "h2d_ms": h2d_ms,
            "gpu_compute_ms": gpu_ms,
            "d2h_ms": d2h_ms,
            "end_to_end_ms": e2e_ms,
            "inferences_per_second_gpu": 1000.0 * args.batch_size / gpu_ms if gpu_ms > 0 else None,
            "inferences_per_second_end_to_end": 1000.0 * args.batch_size / e2e_ms if e2e_ms > 0 else None,
        })

    return {
        "bindings": [{k: v for k, v in item.items() if k not in ("host", "device")} for item in buffers],
        "input_info": input_info,
        "iterations": rows,
        "summary": {
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "gpu_compute_ms": stats_ms(gpu_times),
            "end_to_end_ms": stats_ms(end_to_end_times),
            "host_to_device_ms": stats_ms(h2d_times),
            "device_to_host_ms": stats_ms(d2h_times),
            "inferences_per_second_gpu_avg": 1000.0 * args.batch_size / statistics.mean(gpu_times),
            "inferences_per_second_end_to_end_avg": 1000.0 * args.batch_size / statistics.mean(end_to_end_times),
        },
    }


def write_iteration_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report_html(path, result):
    summary = result["benchmark"]["summary"]
    rows = result["benchmark"]["iterations"]
    cards = [
        ("GPU Inferences/S", f"{summary['inferences_per_second_gpu_avg']:.2f}", "CUDA event compute only"),
        ("End-To-End Inferences/S", f"{summary['inferences_per_second_end_to_end_avg']:.2f}", "H2D + compute + D2H + sync"),
        ("GPU Compute Avg", f"{summary['gpu_compute_ms']['avg']:.3f} ms", "single inference batch latency"),
        ("End-To-End Avg", f"{summary['end_to_end_ms']['avg']:.3f} ms", "host observed latency"),
    ]
    row_html = "\n".join(
        "<tr>" + "".join(f"<td>{value:.4f}</td>" if isinstance(value, float) else f"<td>{value}</td>" for value in row.values()) + "</tr>"
        for row in rows
    )
    headers = "".join(f"<th>{key}</th>" for key in rows[0].keys()) if rows else ""
    card_html = "\n".join(f"<article><div>{title}</div><strong>{value}</strong><p>{desc}</p></article>" for title, value, desc in cards)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorRT Full Pipeline Report</title>
<style>
body{{margin:0;background:#0e1422;color:#edf3ff;font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1180px;margin:auto;padding:30px 16px}}h1{{margin-bottom:6px}}.meta{{color:#9fb0cf;margin-bottom:22px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}article,section{{background:#172137;border:1px solid #2b3a5b;border-radius:16px;padding:16px;box-shadow:0 12px 28px rgba(0,0,0,.25)}}article div{{color:#9fb0cf}}article strong{{display:block;font-size:28px;margin:4px 0}}article p{{margin:0;color:#9fb0cf}}section{{margin-top:18px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:880px}}th,td{{border-bottom:1px solid #2b3a5b;padding:8px;text-align:left}}th{{color:#9fb0cf}}code{{color:#a9d6ff}}pre{{white-space:pre-wrap;background:#10192c;padding:12px;border-radius:12px}}</style>
</head><body><main>
<h1>TensorRT Full Pipeline Report</h1>
<div class="meta">Generated: {result['generated_at']} | Precision: {result['parameters']['precision']} | Batch: {result['parameters']['batch_size']}</div>
<div class="grid">{card_html}</div>
<section><h2>Parameters</h2><pre>{json.dumps(result['parameters'], indent=2)}</pre></section>
<section><h2>Pipeline Results</h2><pre>{json.dumps(result['pipeline'], indent=2)}</pre></section>
<section><h2>Binding Info</h2><pre>{json.dumps(result['benchmark']['bindings'], indent=2)}</pre></section>
<section><h2>Per Iteration</h2><table><thead><tr>{headers}</tr></thead><tbody>{row_html}</tbody></table></section>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone TensorRT full pipeline benchmark for Jetson Nano")
    parser.add_argument("--onnx", default="", help="Input ONNX path. If omitted with --generate-onnx, a small model is generated.")
    parser.add_argument("--engine", default="", help="TensorRT engine path. Defaults to OUT_DIR/model.trt.")
    parser.add_argument("--out-dir", default="trt_pipeline_results", help="Directory for all retained results.")
    parser.add_argument("--generate-onnx", action="store_true", help="Generate a small ConvNet ONNX model using PyTorch.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild engine even if --engine exists.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16", help="TensorRT build precision.")
    parser.add_argument("--workspace-mb", type=int, default=512, help="TensorRT workspace memory in MB.")
    parser.add_argument("--batch-size", type=int, default=1, help="Static inference batch size.")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(224, 224), metavar=("H", "W"), help="Input height and width for generated model and dynamic input shape.")
    parser.add_argument("--classes", type=int, default=1000, help="Generated classification class count.")
    parser.add_argument("--input-name", default="input", help="Generated ONNX input name.")
    parser.add_argument("--output-name", default="logits", help="Generated ONNX output name.")
    parser.add_argument("--image", default="", help="Image path used as repeated input for every measured inference iteration.")
    parser.add_argument("--input-scale", type=float, default=1.0 / 255.0, help="Scale applied to image pixels before normalization.")
    parser.add_argument("--input-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406), metavar=("R", "G", "B"), help="RGB mean used for image normalization.")
    parser.add_argument("--input-std", type=float, nargs=3, default=(0.229, 0.224, 0.225), metavar=("R", "G", "B"), help="RGB std used for image normalization.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset for generated model.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup inference iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured inference iterations.")
    parser.add_argument("--seed", type=int, default=1234, help="Input tensor random seed.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logger.")
    args = parser.parse_args()
    args.input_mean = list(args.input_mean)
    args.input_std = list(args.input_std)
    if args.image and not os.path.exists(args.image):
        raise SystemExit(f"--image does not exist: {args.image}")
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np, trt, cuda = import_required()

    parameters = vars(args).copy()
    parameters["tensorrt_version"] = trt.__version__
    parameters["python"] = sys.version.split()[0]

    onnx_result = maybe_generate_onnx(args, np)
    args.onnx = onnx_result["path"]

    trtexec_log = os.path.join(args.out_dir, "trtexec_available.log")
    trtexec_result = {"status": "skipped", "details": "trtexec not found"}
    if subprocess.run(["which", "trtexec"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        trtexec_result = run_command(["trtexec", f"--onnx={args.onnx}", f"--shapes={args.input_name}:" + "x".join(map(str, (args.batch_size, 3, *args.input_hw))), f"--workspace={args.workspace_mb}", "--dryRun"], trtexec_log)

    engine_result = build_engine(args, trt)
    args.engine = engine_result["path"]
    engine = load_engine(args.engine, trt, args.verbose)
    benchmark_result = benchmark(args, np, trt, cuda, engine)

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": parameters,
        "pipeline": {
            "onnx": onnx_result,
            "trtexec_dry_run": trtexec_result,
            "engine": engine_result,
        },
        "benchmark": benchmark_result,
    }

    json_path = os.path.join(args.out_dir, "tensorrt_pipeline_result.json")
    csv_path = os.path.join(args.out_dir, "tensorrt_iteration_metrics.csv")
    html_path = os.path.join(args.out_dir, "tensorrt_report.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_iteration_csv(csv_path, benchmark_result["iterations"])
    write_report_html(html_path, result)

    summary = benchmark_result["summary"]
    print("TensorRT full pipeline completed")
    print(f"onnx_path={args.onnx}")
    print(f"engine_path={args.engine}")
    print(f"json_result={json_path}")
    print(f"csv_iterations={csv_path}")
    print(f"html_report={html_path}")
    print(f"gpu_inferences_per_second_avg={summary['inferences_per_second_gpu_avg']:.2f}")
    print(f"end_to_end_inferences_per_second_avg={summary['inferences_per_second_end_to_end_avg']:.2f}")
    print(f"single_inference_gpu_compute_latency_avg_ms={summary['gpu_compute_ms']['avg']:.3f}")
    print(f"single_inference_end_to_end_latency_avg_ms={summary['end_to_end_ms']['avg']:.3f}")


if __name__ == "__main__":
    main()

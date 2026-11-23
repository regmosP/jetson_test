#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from anomaly import AnomalyChecker
from config import Settings
from infer import (
    deduplicate_overlapping_items,
    filter_confident_items,
    filter_visible_items,
    list_images,
    make_no_cookie_item,
    parse_items,
    review_reasons,
    save_outputs,
    save_review_sample,
    summarize_defects,
)
from rules import judge_result


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percent / 100.0))
    return ordered[index]


def summarize_ms(values):
    if not values:
        return {"count": 0, "min": None, "avg": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "avg": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Biscuit YOLO inference and record Jetson timing logs.")
    parser.add_argument("--image-source", help="Image file or directory. Defaults to IMAGE_SOURCE in .env.")
    parser.add_argument("--model", help="Model path. Defaults to YOLO_MODEL_PATH in .env.")
    parser.add_argument("--device", help="Ultralytics device, for example 0 or cpu. Defaults to DEVICE in .env.")
    parser.add_argument("--imgsz", type=int, help="Inference image size. Defaults to IMAGE_SIZE in .env.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured repeats per image, default: 1")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repeats on first image, default: 1")
    parser.add_argument("--out", default="runs/timing", help="Timing output directory, default: runs/timing")
    parser.add_argument("--no-save-output", action="store_true", help="Disable visualization/json/txt output while timing")
    return parser.parse_args()


def run_one(model, image_path, settings):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    started = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=settings.image_size,
        conf=settings.conf_threshold,
        iou=settings.iou_threshold,
        device=settings.device,
        verbose=False,
    )[0]
    infer_elapsed_ms = (time.perf_counter() - started) * 1000.0

    post_started = time.perf_counter()
    items = parse_items(result)
    output_items = filter_confident_items(items, settings)
    if not output_items:
        output_items = [make_no_cookie_item(image)]
    else:
        output_items = deduplicate_overlapping_items(output_items, settings)
    anomaly_score = None
    judgment = judge_result(output_items, anomaly_score, settings)
    visible_items = filter_visible_items(output_items, settings)
    defect_summary = summarize_defects(output_items, settings)
    payload = {
        "image": str(image_path),
        "detections": [item.__dict__ for item in visible_items],
        "all_detections": [item.__dict__ for item in items],
        "anomaly_score": anomaly_score,
        **defect_summary,
        **judgment,
    }
    if settings.save_visualization or settings.save_json or settings.save_yolo_txt:
        save_outputs(image_path, result, payload, settings)
        save_review_sample(image_path, payload, settings, review_reasons(items, payload, settings))
    total_elapsed_ms = infer_elapsed_ms + ((time.perf_counter() - post_started) * 1000.0)
    return payload, infer_elapsed_ms, total_elapsed_ms


def main():
    args = parse_args()
    os.chdir(ROOT_DIR)
    settings = Settings()

    if args.image_source:
        object.__setattr__(settings, "image_source", Path(args.image_source))
    if args.model:
        object.__setattr__(settings, "yolo_model_path", Path(args.model))
    if args.device:
        object.__setattr__(settings, "device", args.device)
    if args.imgsz:
        object.__setattr__(settings, "image_size", args.imgsz)
    if args.no_save_output:
        object.__setattr__(settings, "save_visualization", False)
        object.__setattr__(settings, "save_json", False)
        object.__setattr__(settings, "save_yolo_txt", False)

    image_paths = list_images(settings.image_source)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {settings.image_source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "timing.csv"
    log_path = out_dir / "inference_time.log"
    summary_path = out_dir / "summary.json"
    model_path = settings.tensorrt_engine_path if settings.use_tensorrt else settings.yolo_model_path

    print(f"model={model_path} device={settings.device} imgsz={settings.image_size} images={len(image_paths)}")
    load_started = time.perf_counter()
    model = YOLO(str(model_path))
    load_elapsed_ms = (time.perf_counter() - load_started) * 1000.0

    for index in range(args.warmup):
        run_one(model, image_paths[0], settings)
        print(f"warmup={index + 1}/{args.warmup}")

    rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, log_path.open("w", encoding="utf-8") as log_file:
        fieldnames = ["time", "image", "repeat", "bytes", "infer_ms", "total_ms", "final_result", "detections"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        log_file.write(
            f"start={datetime.now().isoformat(timespec='seconds')} model={model_path} "
            f"device={settings.device} imgsz={settings.image_size} images={len(image_paths)} "
            f"repeat={args.repeat} warmup={args.warmup} model_load_ms={load_elapsed_ms:.3f}\n"
        )
        log_file.flush()
        for image_path in image_paths:
            image_bytes = image_path.stat().st_size
            for repeat_index in range(args.repeat):
                payload, infer_ms, total_ms = run_one(model, image_path, settings)
                row = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "image": str(image_path),
                    "repeat": repeat_index + 1,
                    "bytes": image_bytes,
                    "infer_ms": infer_ms,
                    "total_ms": total_ms,
                    "final_result": payload["final_result"],
                    "detections": len(payload["detections"]),
                }
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                log_line = (
                    f"{image_path.name} repeat={repeat_index + 1} "
                    f"infer={infer_ms:.2f}ms total={total_ms:.2f}ms result={payload['final_result']}"
                )
                log_file.write(f"{row['time']} {log_line}\n")
                log_file.flush()
                print(log_line, flush=True)

    infer_values = [row["infer_ms"] for row in rows]
    total_values = [row["total_ms"] for row in rows]
    summary = {
        "model": str(model_path),
        "device": settings.device,
        "image_size": settings.image_size,
        "image_count": len(image_paths),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "model_load_ms": load_elapsed_ms,
        "infer_ms": summarize_ms(infer_values),
        "total_ms": summarize_ms(total_values),
        "csv": str(csv_path),
        "log": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {log_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

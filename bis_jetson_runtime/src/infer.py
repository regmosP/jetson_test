from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

from anomaly import AnomalyChecker
from config import Settings
from rules import DetectionItem, judge_result


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_PRIORITY = {"fragment": 0, "broken": 1, "edge_damage": 2}
OUTPUT_CLASS_IDS = {"oreo_ok": 0, "edge_damage": 1, "broken": 2, "fragment": 3, "overlap": 4, "no_cookie": 5}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_items(result) -> list[DetectionItem]:
    items = []
    if result.boxes is None:
        return items

    names = result.names
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()

    for index, class_id in enumerate(classes):
        x1, y1, x2, y2 = boxes[index]
        items.append(
            DetectionItem(
                class_name=names[int(class_id)],
                confidence=float(confidences[index]),
                box=[float(value) for value in boxes[index]],
                box_area=float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
            )
        )
    return items


def filter_visible_items(items: list[DetectionItem], settings: Settings) -> list[DetectionItem]:
    defect_items = [item for item in items if item.class_name in settings.defect_classes]
    has_defect = bool(defect_items)
    if has_defect:
        visible_items = []
        for item in items:
            if item.class_name not in settings.hide_classes_in_output:
                visible_items.append(item)
            elif any(box_overlap_score(item.box, defect_item.box) >= settings.overlap_iou_threshold for defect_item in defect_items):
                visible_items.append(item)
        return visible_items
    return items


def filter_confident_items(items: list[DetectionItem], settings: Settings) -> list[DetectionItem]:
    return [item for item in items if item.confidence >= settings.output_conf_threshold]


def make_no_cookie_item(image) -> DetectionItem:
    height, width = image.shape[:2]
    return DetectionItem(
        class_name="no_cookie",
        confidence=1.0,
        box=[0.0, 0.0, float(width), float(height)],
        box_area=float(width * height),
    )


def box_overlap_score(first_box: list[float], second_box: list[float]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection_area = intersection_width * intersection_height
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    iou = intersection_area / union_area if union_area else 0.0
    smaller_area = min(first_area, second_area)
    smaller_overlap = intersection_area / smaller_area if smaller_area else 0.0
    return max(iou, smaller_overlap)


def deduplicate_overlapping_items(items: list[DetectionItem], settings: Settings) -> list[DetectionItem]:
    sorted_items = sorted(
        items,
        key=lambda item: (CLASS_PRIORITY.get(item.class_name, 100), -item.confidence),
    )
    kept_items = []
    for item in sorted_items:
        if any(
            item.class_name not in settings.ok_classes
            and kept_item.class_name not in settings.ok_classes
            and box_overlap_score(item.box, kept_item.box) >= settings.overlap_iou_threshold
            for kept_item in kept_items
        ):
            continue
        kept_items.append(item)
    return kept_items


def summarize_defects(items: list[DetectionItem], settings: Settings) -> dict:
    defects = []
    counts = {}
    for item in items:
        if item.class_name not in settings.defect_classes:
            continue
        display_name = settings.class_display_names.get(item.class_name, item.class_name)
        counts[display_name] = counts.get(display_name, 0) + 1
        defects.append(
            {
                "class_name": item.class_name,
                "display_name": display_name,
                "confidence": item.confidence,
                "box": item.box,
                "box_area": item.box_area,
            }
        )
    return {
        "defect_count": len(defects),
        "defect_types": sorted(counts),
        "defect_summary": counts,
        "defects": defects,
    }


def save_outputs(image_path: Path, result, payload: dict, settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    if settings.save_visualization:
        visualized = cv2.imread(str(image_path))
        for item in payload["detections"]:
            x1, y1, x2, y2 = [int(value) for value in item["box"]]
            label = f"{item['class_name']} {item['confidence']:.2f}"
            cv2.rectangle(visualized, (x1, y1), (x2, y2), (0, 0, 255), settings.visualization_line_width)
            cv2.putText(
                visualized,
                label,
                (x1, max(40, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                settings.visualization_font_scale,
                (0, 0, 255),
                settings.visualization_font_thickness,
            )
        cv2.imwrite(str(settings.output_dir / f"{stem}_vis.jpg"), visualized)

    if settings.save_json:
        with (settings.output_dir / f"{stem}.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    if settings.save_yolo_txt:
        with (settings.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as file:
            height, width = result.orig_shape
            class_ids = {name: class_id for class_id, name in result.names.items()}
            for item in payload["detections"]:
                x1, y1, x2, y2 = item["box"]
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                class_id = class_ids.get(item["class_name"], OUTPUT_CLASS_IDS[item["class_name"]])
                confidence = item["confidence"]
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                if settings.save_yolo_conf:
                    line = f"{line} {confidence:.6f}"
                file.write(f"{line}\n")


def review_reasons(items: list[DetectionItem], payload: dict, settings: Settings) -> list[str]:
    reasons = []
    if settings.review_save_ng and payload["final_result"] == "NG":
        reasons.append("ng")

    low_conf_items = [item for item in items if item.confidence < settings.review_low_conf_threshold]
    if settings.review_save_low_conf and low_conf_items:
        reasons.append("low_conf")

    anomaly_score = payload.get("anomaly_score")
    if settings.review_save_anomaly and anomaly_score is not None and anomaly_score >= settings.anomaly_score_threshold:
        reasons.append("anomaly")

    return reasons


def save_review_sample(image_path: Path, payload: dict, settings: Settings, reasons: list[str]) -> None:
    if not settings.enable_review_save or not reasons:
        return

    reason_dir = "__".join(sorted(set(reasons)))
    target_dir = settings.review_dir / reason_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    image_target = target_dir / image_path.name
    json_target = target_dir / f"{image_path.stem}.json"
    shutil.copy2(image_path, image_target)
    with json_target.open("w", encoding="utf-8") as file:
        json.dump({**payload, "review_reasons": reasons}, file, ensure_ascii=False, indent=2)


def main() -> None:
    settings = Settings()
    model_path = settings.tensorrt_engine_path if settings.use_tensorrt else settings.yolo_model_path
    model = YOLO(str(model_path))
    anomaly_checker = AnomalyChecker(settings.anomaly_model_path) if settings.use_anomaly_check else None

    image_paths = list_images(settings.image_source)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {settings.image_source}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        result = model.predict(
            source=image,
            imgsz=settings.image_size,
            conf=settings.conf_threshold,
            iou=settings.iou_threshold,
            device=settings.device,
            verbose=False,
        )[0]
        items = parse_items(result)
        output_items = filter_confident_items(items, settings)
        if not output_items:
            output_items = [make_no_cookie_item(image)]
        else:
            output_items = deduplicate_overlapping_items(output_items, settings)
        anomaly_score = anomaly_checker.score(image) if anomaly_checker is not None else None
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
        save_outputs(image_path, result, payload, settings)
        save_review_sample(image_path, payload, settings, review_reasons(items, payload, settings))
        print(f"{image_path.name}: {payload['final_result']} {payload['reasons']}")


if __name__ == "__main__":
    main()

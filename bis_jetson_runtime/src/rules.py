from __future__ import annotations

from dataclasses import dataclass

from config import Settings


@dataclass
class DetectionItem:
    class_name: str
    confidence: float
    box: list[float]
    box_area: float


def judge_result(items: list[DetectionItem], anomaly_score: float | None, settings: Settings) -> dict:
    reasons = []

    for item in items:
        if item.class_name in settings.defect_classes - settings.slot_classes and item.box_area >= settings.min_defect_box_area:
            reasons.append(f"{item.class_name} detected")

    cookie_items = [item for item in items if item.class_name in settings.ok_classes]
    no_cookie_items = [item for item in items if item.class_name in settings.slot_classes]
    has_defect = any(item.class_name in settings.defect_classes - settings.slot_classes for item in items)
    if no_cookie_items:
        reasons.append("no_cookie detected")
    elif not has_defect and len(cookie_items) < settings.expected_cookie_min_count:
        reasons.append("no cookie detected")
    elif len(cookie_items) > settings.expected_cookie_max_count:
        reasons.append(f"too many cookies detected: {len(cookie_items)}")

    for item in cookie_items:
        if item.box_area < settings.min_cookie_box_area:
            reasons.append(f"cookie box area too small {item.box_area:.1f}")

    if anomaly_score is not None and anomaly_score >= settings.anomaly_score_threshold:
        reasons.append(f"anomaly score {anomaly_score:.3f}")

    return {
        "final_result": "NG" if reasons else "OK",
        "reasons": reasons,
    }

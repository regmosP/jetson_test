from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str) -> set[str]:
    value = os.getenv(name, "")
    return {item.strip() for item in value.split(",") if item.strip()}


def _get_mapping(name: str) -> dict[str, str]:
    value = os.getenv(name, "")
    mapping = {}
    for item in value.split(","):
        if ":" not in item:
            continue
        key, label = item.split(":", 1)
        mapping[key.strip()] = label.strip()
    return mapping


def _resolve_device(default: str = "0") -> str:
    device = os.getenv("DEVICE", default).strip()
    if device.lower() == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    return device if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Settings:
    yolo_model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", "models/yolo11m.pt"))
    anomaly_model_path: Path = Path(os.getenv("ANOMALY_MODEL_PATH", "models/efficientad.onnx"))
    image_source: Path = Path(os.getenv("IMAGE_SOURCE", "data/input"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "runs/inference"))
    device: str = _resolve_device("0")
    image_size: int = int(os.getenv("IMAGE_SIZE", "2048"))
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.35"))
    output_conf_threshold: float = float(os.getenv("OUTPUT_CONF_THRESHOLD", os.getenv("CONF_THRESHOLD", "0.35")))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.50"))
    use_tensorrt: bool = _get_bool("USE_TENSORRT", False)
    tensorrt_engine_path: Path = Path(os.getenv("TENSORRT_ENGINE_PATH", "models/yolo11m.engine"))
    use_anomaly_check: bool = _get_bool("USE_ANOMALY_CHECK", False)
    anomaly_score_threshold: float = float(os.getenv("ANOMALY_SCORE_THRESHOLD", "0.60"))
    save_visualization: bool = _get_bool("ENABLE_SAVE_VISUALIZATION", True)
    save_json: bool = _get_bool("ENABLE_SAVE_JSON", True)
    save_yolo_txt: bool = _get_bool("ENABLE_SAVE_YOLO_TXT", True)
    save_yolo_conf: bool = _get_bool("ENABLE_SAVE_YOLO_CONF", True)
    visualization_line_width: int = int(os.getenv("VISUALIZATION_LINE_WIDTH", "5"))
    visualization_font_scale: float = float(os.getenv("VISUALIZATION_FONT_SCALE", "2.0"))
    visualization_font_thickness: int = int(os.getenv("VISUALIZATION_FONT_THICKNESS", "4"))
    overlap_iou_threshold: float = float(os.getenv("OVERLAP_IOU_THRESHOLD", "0.30"))
    enable_review_save: bool = _get_bool("ENABLE_REVIEW_SAVE", True)
    review_dir: Path = Path(os.getenv("REVIEW_DIR", "data/review"))
    review_save_ng: bool = _get_bool("REVIEW_SAVE_NG", True)
    review_save_low_conf: bool = _get_bool("REVIEW_SAVE_LOW_CONF", True)
    review_low_conf_threshold: float = float(os.getenv("REVIEW_LOW_CONF_THRESHOLD", "0.45"))
    review_save_anomaly: bool = _get_bool("REVIEW_SAVE_ANOMALY", True)
    ok_classes: set[str] = None
    defect_classes: set[str] = None
    slot_classes: set[str] = None
    hide_classes_in_output: set[str] = None
    class_display_names: dict[str, str] = None
    expected_cookie_min_count: int = int(os.getenv("EXPECTED_COOKIE_MIN_COUNT", "1"))
    expected_cookie_max_count: int = int(os.getenv("EXPECTED_COOKIE_MAX_COUNT", "2"))
    min_cookie_box_area: float = float(os.getenv("MIN_COOKIE_BOX_AREA", "1000"))
    min_defect_box_area: float = float(os.getenv("MIN_DEFECT_BOX_AREA", "10"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "ok_classes", _get_list("OK_CLASSES"))
        object.__setattr__(self, "defect_classes", _get_list("DEFECT_CLASSES"))
        object.__setattr__(self, "slot_classes", _get_list("SLOT_CLASSES"))
        object.__setattr__(self, "hide_classes_in_output", _get_list("HIDE_CLASSES_IN_OUTPUT"))
        object.__setattr__(self, "class_display_names", _get_mapping("CLASS_DISPLAY_NAMES"))


@dataclass(frozen=True)
class TrainSettings:
    model_path: str = os.getenv("YOLO_MODEL_PATH", "models/yolo11m.pt")
    data_yaml: str = os.getenv("TRAIN_DATA_YAML", "data/dataset.yaml")
    image_size: int = int(os.getenv("IMAGE_SIZE", "2048"))
    epochs: int = int(os.getenv("TRAIN_EPOCHS", "80"))
    batch: int = int(os.getenv("TRAIN_BATCH", "8"))
    workers: int = int(os.getenv("TRAIN_WORKERS", "8"))
    device: str = _resolve_device("0")
    project: str = os.getenv("TRAIN_PROJECT", "runs/train")
    name: str = os.getenv("TRAIN_NAME", "biscuit_yolo11m_det")


@dataclass(frozen=True)
class ExportSettings:
    model_path: str = os.getenv("YOLO_MODEL_PATH", "models/yolo11m.pt")
    export_format: str = os.getenv("EXPORT_FORMAT", "onnx")
    image_size: int = int(os.getenv("IMAGE_SIZE", "2048"))
    opset: int = int(os.getenv("EXPORT_OPSET", "12"))
    half: bool = _get_bool("EXPORT_HALF", True)
    device: str = _resolve_device("0")

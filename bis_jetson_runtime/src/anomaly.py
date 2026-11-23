from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class AnomalyChecker:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session = None
        if model_path.exists():
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def score(self, image: np.ndarray) -> float | None:
        if self.session is None:
            return None

        input_name = self.session.get_inputs()[0].name
        resized = cv2.resize(image, (256, 256))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {input_name: tensor})
        return float(np.max(outputs[0]))

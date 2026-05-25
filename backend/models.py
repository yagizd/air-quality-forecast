"""
models.py
---------
ONNX Runtime inference — torch bagimliligi yok.

load_model(city, models_dir) -> OnnxModel
OnnxModel.predict(array (1,48,18)) -> float
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

N_FEATURES = 18
WINDOW     = 48


class OnnxModel:
    def __init__(self, onnx_path: Path) -> None:
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, x: np.ndarray) -> float:
        """x: shape (1, 48, N_FEATURES), dtype float32"""
        result = self._session.run(["output"], {"input": x})
        return float(result[0].flat[0])


def load_model(city: str, models_dir: Path) -> OnnxModel:
    path = models_dir / f"{city}_pm25.onnx"
    return OnnxModel(path)

"""
06_export_onnx.py
-----------------
Her sehir icin PyTorch modelini ONNX'e export eder ve dogrular.

Calistir:
    python scripts/06_export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

N_FEATURES = 18


class PM25LSTM(nn.Module):
    """05_train.py ile birebir ayni mimari."""
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=N_FEATURES, hidden_size=64,
                             num_layers=1, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32,
                             num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(32, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out)
        out    = out[:, -1, :]
        out    = self.relu(self.fc1(out))
        out    = self.fc2(out)
        return out.squeeze(-1)

MODELS_DIR = ROOT / "models"
CITIES     = ["berlin", "hamburg", "munich", "cologne"]
WINDOW     = 48
OPSET      = 17
MAX_DIFF   = 1e-4


def export_city(city: str) -> None:
    pt_path   = MODELS_DIR / f"{city}_pm25.pt"
    onnx_path = MODELS_DIR / f"{city}_pm25.onnx"

    # 1) PyTorch modelini yukle
    model = PM25LSTM()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    # 2) Dummy input
    dummy = torch.randn(1, WINDOW, N_FEATURES)

    # 3) Torch inference (referans)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    # 4) ONNX export
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    # 5) Dogrulama: onnxruntime inference
    sess    = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {"input": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - ort_out).max())
    if max_diff > MAX_DIFF:
        raise RuntimeError(
            f"[{city}] output mismatch: max_diff={max_diff:.6f} > {MAX_DIFF}"
        )

    print(f"{city}: ✓ exported, max_diff={max_diff:.6f}")


def main() -> None:
    print(f"ONNX opset={OPSET}  window={WINDOW}  n_features={N_FEATURES}\n")
    for city in CITIES:
        export_city(city)
    print("\nTum sehirler basariyla export edildi.")


if __name__ == "__main__":
    main()

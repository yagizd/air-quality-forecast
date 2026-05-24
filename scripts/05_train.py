"""
05_train.py
-----------
Girdi : data/processed/{city}_features.parquet  (4 sehir)
        models/scalers/{city}_scaler.pkl
Cikti : models/{city}_pm25.pt                   -- state_dict
        models/training_report.json             -- per-city metrics

Model mimarisi:
    LSTM(18 -> 64)  + Dropout(0.2)
    LSTM(64 -> 32)  + Dropout(0.2)
    Linear(32 -> 16) + ReLU
    Linear(16 -> 1)

Training:
    Loss      : nn.L1Loss  (MAE)
    Optimizer : Adam, lr=1e-3
    Batch     : 64
    Max epoch : 50
    Early stop: 10 epoch patience, best weights restored

Calistir:
    python scripts/05_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CITIES       = ["berlin", "hamburg", "munich", "cologne"]
WINDOW       = 48
TRAIN_RATIO  = 0.80
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
PATIENCE     = 10
LR           = 1e-3
N_FEATURES   = 18

FEATURE_COLS = (
    [f"{poll}_lag_{h}h" for poll in ["pm25", "no2", "o3"] for h in [1, 6, 24, 48]]
    + ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]
)
TARGET_COL = "pm25"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PM25LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1   = nn.LSTM(input_size=N_FEATURES, hidden_size=64,
                               num_layers=1, batch_first=True)
        self.drop1   = nn.Dropout(0.2)
        self.lstm2   = nn.LSTM(input_size=64, hidden_size=32,
                               num_layers=1, batch_first=True)
        self.drop2   = nn.Dropout(0.2)
        self.fc1     = nn.Linear(32, 16)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq=48, features=18)
        out, _ = self.lstm1(x)           # (batch, 48, 64)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)         # (batch, 48, 32)
        out    = self.drop2(out)
        out    = out[:, -1, :]           # son adim: (batch, 32)
        out    = self.relu(self.fc1(out)) # (batch, 16)
        out    = self.fc2(out)            # (batch, 1)
        return out.squeeze(-1)            # (batch,)


# ---------------------------------------------------------------------------
# Veri hazirlama
# ---------------------------------------------------------------------------

def make_windows(X: np.ndarray, y: np.ndarray, window: int):
    """Sliding-window: X[i] = X_arr[i : i+window], y[i] = y_arr[i+window-1]"""
    n = len(X) - window + 1
    X_w = np.lib.stride_tricks.sliding_window_view(X, (window, X.shape[1]))
    X_w = X_w[:, 0, :, :]   # (n, window, features)
    y_w = y[window - 1:]     # (n,)
    return X_w[:n], y_w[:n]


def load_city(city: str):
    feat_path = PROCESSED_DIR / f"{city}_features.parquet"
    df = pd.read_parquet(feat_path)

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)

    # Zaman sirali train/val bolumu
    n      = len(X_all)
    n_tr   = int(n * TRAIN_RATIO)
    X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
    X_vl, y_vl = X_all[n_tr:], y_all[n_tr:]

    # Sliding windows
    X_tr_w, y_tr_w = make_windows(X_tr, y_tr, WINDOW)
    X_vl_w, y_vl_w = make_windows(X_vl, y_vl, WINDOW)

    to_tensor = lambda a: torch.from_numpy(a)
    tr_ds = TensorDataset(to_tensor(X_tr_w), to_tensor(y_tr_w))
    vl_ds = TensorDataset(to_tensor(X_vl_w), to_tensor(y_vl_w))

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False)
    vl_dl = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)

    return tr_dl, vl_dl


# ---------------------------------------------------------------------------
# Epoch fonksiyonlari
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_samples  = 0

    with torch.set_grad_enabled(training):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_b)
            n_samples  += len(y_b)

    return total_loss / n_samples


# ---------------------------------------------------------------------------
# Tek sehir egitim
# ---------------------------------------------------------------------------

def train_city(city: str) -> dict:
    print(f"\n[{city.upper()}] Veriler yukleniyor...", flush=True)
    tr_dl, vl_dl = load_city(city)

    model     = PM25LSTM().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val   = float("inf")
    best_state = None
    patience_c = 0
    stopped_ep = MAX_EPOCHS
    train_mae_final = None

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_mae = run_epoch(model, tr_dl, criterion, optimizer)
        vl_mae = run_epoch(model, vl_dl, criterion)

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train_mae={tr_mae:.4f} | val_mae={vl_mae:.4f}",
                  flush=True)

        train_mae_final = tr_mae

        if vl_mae < best_val - 1e-6:
            best_val   = vl_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                stopped_ep = epoch
                print(f"  Early stop @ epoch {epoch}  (best_val_mae={best_val:.4f})",
                      flush=True)
                break

    # En iyi agirliklari geri yukle ve kaydet
    model.load_state_dict(best_state)
    out_path = MODELS_DIR / f"{city}_pm25.pt"
    torch.save(model.state_dict(), out_path)
    print(f"  Model kaydedildi: {out_path}", flush=True)

    return {
        "best_val_mae":    round(float(best_val),         4),
        "stopped_epoch":   stopped_ep,
        "train_mae_final": round(float(train_mae_final),  4),
        "overfit_ratio":   round(float(best_val / (train_mae_final + 1e-9)), 4),
    }


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}", flush=True)
    report = {}
    for city in CITIES:
        report[city] = train_city(city)

    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\ntraining_report.json kaydedildi: {report_path}", flush=True)
    print("\n--- SONUC ---")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

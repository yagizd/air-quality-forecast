"""
04_features.py
--------------
Girdi : data/processed/{city}_hourly_agg.parquet  (4 sehir)
Cikti : data/processed/{city}_features.parquet    -- olcekli feature matrisi
        models/scalers/{city}_scaler.pkl           -- joblib MinMaxScaler

Feature listesi (n_features = 18):
  Lag  : pm25_lag_1h/6h/24h/48h  (4)
         no2_lag_1h/6h/24h/48h   (4)
         o3_lag_1h/6h/24h/48h    (4)
  Zaman: hour_sin, hour_cos       (2)
         month_sin, month_cos     (2)
         dayofweek_sin, dayofweek_cos  (2)

Target : pm25  (shift yok, ayni satir, olceklenmez)

NaN handling : lag uretiminden kaynaklanan ilk satirlar + Cologne 138 NaN
               drop et, interpolasyon yapma.

Scaler : sehir bazinda MinMaxScaler, sadece feature kolonlarina fit.

Calistir:
    python scripts/04_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SCALERS_DIR   = Path(__file__).parent.parent / "models" / "scalers"
SCALERS_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS    = ["pm25", "no2", "o3"]
LAG_HOURS     = [1, 6, 24, 48]
WINDOW        = 48        # LSTM sliding window boyutu
TRAIN_RATIO   = 0.80

# Cyclical encode: (kolon_adi, periyot)
CYCLICAL = [
    ("hour",       24),
    ("month",      12),
    ("dayofweek",   7),
]

# Feature kolon sirasi (target haric)
LAG_COLS = [
    f"{poll}_lag_{h}h"
    for poll in POLLUTANTS
    for h in LAG_HOURS
]  # 12 kolon

TIME_COLS = [f"{name}_{fn}" for name, _ in CYCLICAL for fn in ("sin", "cos")]  # 6 kolon

FEATURE_COLS = LAG_COLS + TIME_COLS   # 18 kolon
TARGET_COL   = "pm25"


# ---------------------------------------------------------------------------
# Yardimci: Lag feature uret
# ---------------------------------------------------------------------------

def make_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for poll in POLLUTANTS:
        for h in LAG_HOURS:
            df[f"{poll}_lag_{h}h"] = df[poll].shift(h)
    return df


# ---------------------------------------------------------------------------
# Yardimci: Cyclical zaman feature uret
# ---------------------------------------------------------------------------

def make_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"].dt
    df["hour"]       = dt.hour
    df["month"]      = dt.month
    df["dayofweek"]  = dt.dayofweek

    for name, period in CYCLICAL:
        df[f"{name}_sin"] = np.sin(2 * np.pi * df[name] / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * df[name] / period)
        df = df.drop(columns=[name])

    return df


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def process_city(src_path: Path) -> dict:
    city = src_path.stem.replace("_hourly_agg", "")

    df = pd.read_parquet(src_path)
    df = df.sort_values("datetime").reset_index(drop=True)

    n_raw = len(df)

    # Feature uretimi
    df = make_lags(df)
    df = make_cyclical(df)

    # NaN drop (lag bosluklar + Cologne O3 kalintilari)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    n_clean = len(df)
    n_dropped = n_raw - n_clean

    # Scaler: sadece feature kolonlarina fit
    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # Kaydet
    out_cols = ["datetime"] + FEATURE_COLS + [TARGET_COL]
    df[out_cols].to_parquet(PROCESSED_DIR / f"{city}_features.parquet", index=False)

    scaler_path = SCALERS_DIR / f"{city}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Shape hesabi
    n_rows    = n_clean
    n_samples = n_rows - WINDOW + 1          # sliding window, stride=1
    n_train   = int(n_samples * TRAIN_RATIO)
    n_val     = n_samples - n_train

    return {
        "city":       city,
        "n_raw":      n_raw,
        "n_dropped":  n_dropped,
        "n_rows":     n_rows,
        "n_features": len(FEATURE_COLS),
        "window":     WINDOW,
        "n_samples":  n_samples,
        "n_train":    n_train,
        "n_val":      n_val,
    }


def main() -> None:
    src_files = sorted(PROCESSED_DIR.glob("*_hourly_agg.parquet"))
    if not src_files:
        print("[HATA] *_hourly_agg.parquet bulunamadi. Once 03_aggregate.py calistir.")
        return

    results = [process_city(f) for f in src_files]

    # Sadece istenen cikti
    nf = results[0]["n_features"]
    w  = results[0]["window"]
    print(f"\nn_features={nf}  window={w}  train_ratio={TRAIN_RATIO:.0%}")
    print(f"\n{'Sehir':<10} {'Shape (samples,win,feat)':<26} {'Train':>8} {'Val':>8}")
    print("-" * 58)
    for r in results:
        shape_str = f"({r['n_samples']:,}, {w}, {nf})"
        print(f"{r['city']:<10} {shape_str:<26} {r['n_train']:>8,} {r['n_val']:>8,}")


if __name__ == "__main__":
    main()

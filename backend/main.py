"""
main.py
-------
FastAPI app:
    /health  -> servis durumu, cache yas bilgisi
    /predict -> sehir + saat parametresi ile pm25 tahmini
    /latest  -> son okumalar (harita icin)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from cache import CITIES, CacheManager, build_features
from models import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

BACKEND_DIR = Path(__file__).parent
MODELS_DIR  = BACKEND_DIR.parent / "models"

state: Dict = {"models": {}, "cache": None}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) Modelleri RAM'e al (ONNX Runtime)
    for c in CITIES:
        state["models"][c] = load_model(c, MODELS_DIR)
        logger.info(f"[{c}] model loaded from {c}_pm25.onnx")

    # 2) CacheManager — eager load + periodic background task
    cache = CacheManager()
    cache.start()
    state["cache"] = cache
    logger.info("Cache manager started (background refresh scheduled)")

    yield

    cache.stop()


app = FastAPI(title="Air Quality Forecast API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://air-quality-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    cache = state["cache"]
    now   = datetime.now(timezone.utc)

    last_updated: Dict[str, str | None] = {}
    cache_age:    Dict[str, float | None] = {}

    for c in CITIES:
        entry = cache.entries[c] if cache else None
        if entry and entry.last_updated:
            last_updated[c] = entry.last_updated.isoformat()
            cache_age[c]    = round((now - entry.last_updated).total_seconds() / 60.0, 2)
        else:
            last_updated[c] = None
            cache_age[c]    = None

    return {
        "status":            "ok",
        "models_loaded":     list(state["models"].keys()),
        "last_updated":      last_updated,
        "cache_age_minutes": cache_age,
    }


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

@app.get("/predict")
async def predict(city: str = Query(...), hours: int = Query(24)):
    if city not in CITIES:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_city", "city": city,
                    "allowed": list(CITIES.keys())},
        )
    if hours < 1 or hours > 72:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_hours", "hours": hours,
                    "allowed_range": [1, 72]},
        )

    cache = state["cache"]
    entry = cache.entries[city]
    if entry.raw_df is None or entry.scaler is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "city": city,
                    "reason": "cache_not_ready"},
        )

    model  = state["models"].get(city)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "city": city,
                    "reason": "model_not_loaded"},
        )

    df_running = entry.raw_df.copy()
    predictions = []

    for _ in range(hours):
        last_dt = pd.to_datetime(df_running["datetime"].iloc[-1])
        next_dt = last_dt + timedelta(hours=1)
        new_row = {
            "datetime": next_dt,
            "pm25":     np.nan,
            "no2":      df_running["no2"].iloc[-1],
            "o3":       df_running["o3"].iloc[-1],
        }
        df_ext = pd.concat([df_running, pd.DataFrame([new_row])], ignore_index=True)
        try:
            window, _ = build_features(df_ext, entry.scaler)
        except Exception as e:
            logger.error(f"[{city}] predict feature build failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={"error": "model_unavailable", "city": city,
                        "reason": "feature_build_failed"},
            )

        pred = model.predict(window)

        predictions.append({
            "datetime": next_dt.isoformat(),
            "pm25":     round(pred, 2),
        })

        df_ext.loc[df_ext.index[-1], "pm25"] = pred
        df_running = df_ext

    return {
        "city":         city,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions":  predictions,
        "unit":         "µg/m³",
    }


# ---------------------------------------------------------------------------
# /latest
# ---------------------------------------------------------------------------

@app.get("/latest")
async def latest():
    cache    = state["cache"]
    now      = datetime.now(timezone.utc)
    readings = []

    for c, info in CITIES.items():
        entry = cache.entries[c] if cache else None
        pm25_val = None
        if entry and entry.raw_df is not None and not entry.raw_df.empty:
            s = entry.raw_df["pm25"].dropna()
            if not s.empty:
                pm25_val = round(float(s.iloc[-1]), 2)
        readings.append({
            "city": c,
            "lat":  info["lat"],
            "lon":  info["lon"],
            "pm25": pm25_val,
        })

    return {
        "timestamp": now.isoformat(),
        "readings":  readings,
    }

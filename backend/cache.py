"""
cache.py
--------
1) fetch_latest(city) -> EEA API'den son ~7 gunluk PM2.5/NO2/O3 verisi cek,
                          merged raw DataFrame dondur.
2) build_features(raw_df, scaler) -> 04_features.py ile ayni feature
                                      engineering, scaler.transform uygula,
                                      (1, 48, 18) numpy array + processed df.
3) CacheManager -> 4 sehir icin eager load + saatlik arka plan refresh.
                    Refresh basarisiz olursa eski cache korunur, log'a yazilir.
"""

from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import httpx
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("cache")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parent
MODELS_DIR  = BACKEND_DIR.parent / "models"
SCALERS_DIR = MODELS_DIR / "scalers"

CITIES = {
    "berlin":  {"name": "Berlin",  "lat": 52.5200, "lon": 13.4050, "eea": "Berlin"},
    "hamburg": {"name": "Hamburg", "lat": 53.5511, "lon":  9.9937, "eea": "Hamburg"},
    "munich":  {"name": "Munich",  "lat": 48.1351, "lon": 11.5820, "eea": "München"},
    "cologne": {"name": "Cologne", "lat": 50.9375, "lon":  6.9603, "eea": "Köln"},
}

POLLUTANT_CODES = {"pm25": "PM2.5", "no2": "NO2", "o3": "O3"}

WINDOW       = 48
N_FEATURES   = 18
TARGET_COL   = "pm25"
LAG_COLS     = [f"{p}_lag_{h}h" for p in ["pm25", "no2", "o3"] for h in [1, 6, 24, 48]]
TIME_COLS    = ["hour_sin", "hour_cos", "month_sin", "month_cos",
                "dayofweek_sin", "dayofweek_cos"]
FEATURE_COLS = LAG_COLS + TIME_COLS

EEA_SUBMIT_URL = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/async"


# ---------------------------------------------------------------------------
# EEA fetch
# ---------------------------------------------------------------------------

async def _submit_request(city_name: str, pollutant_code: int,
                          start: datetime, end: datetime,
                          client: httpx.AsyncClient) -> str:
    body = {
        "countries":       ["DE"],
        "cities":          [city_name],
        "pollutants":      [pollutant_code],
        "dataset":         "2",
        "source":          "Api",
        "dateTimeStart":   start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateTimeEnd":     end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregationType": "hour",
    }
    r = await client.post(EEA_SUBMIT_URL, json=body, timeout=60.0)
    r.raise_for_status()
    return r.text.strip().strip('"')


async def _poll_blob(blob_url: str, client: httpx.AsyncClient,
                     max_wait_seconds: int = 180) -> bytes:
    elapsed = 0
    delay   = 5
    while elapsed < max_wait_seconds:
        r = await client.get(blob_url, timeout=60.0)
        if r.status_code == 200:
            return r.content
        if r.status_code != 404:
            r.raise_for_status()
        await asyncio.sleep(delay)
        elapsed += delay
    raise TimeoutError(f"Blob not ready after {max_wait_seconds}s: {blob_url}")


def _parse_zip(content: bytes, pollutant_key: str) -> pd.DataFrame:
    zf  = zipfile.ZipFile(io.BytesIO(content))
    dfs = []
    for name in zf.namelist():
        if not name.lower().endswith(".parquet"):
            continue
        with zf.open(name) as f:
            raw = f.read()
        try:
            df = pd.read_parquet(io.BytesIO(raw))
        except Exception as e:
            logger.warning(f"Parquet read failed for {name}: {e}")
            continue
        if "Value" not in df or "Start" not in df:
            continue
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.rename(columns={"Start": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime", "Value"])
        df[pollutant_key] = df["Value"]
        dfs.append(df[["datetime", pollutant_key]])

    if not dfs:
        return pd.DataFrame(columns=["datetime", pollutant_key])

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.groupby("datetime", as_index=False)[pollutant_key].mean()
    return merged


async def fetch_latest(city: str) -> pd.DataFrame:
    """
    EEA'dan son ~7 gunluk verileri cek, datetime + pm25/no2/o3 kolonlu
    DataFrame dondur. Tum kirleticiler basarisiz olursa istisna firlatir.
    """
    info  = CITIES[city]
    end   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=7)

    frames = []
    async with httpx.AsyncClient() as client:
        for poll_key, poll_code in POLLUTANT_CODES.items():
            try:
                blob = await _submit_request(info["eea"], poll_code,
                                             start, end, client)
                content = await _poll_blob(blob, client)
                df = _parse_zip(content, poll_key)
                logger.info(f"[{city}] {poll_key}: {len(df)} hourly rows")
                frames.append(df)
            except Exception as e:
                logger.warning(f"[{city}] {poll_key} fetch failed: {e}")
                frames.append(pd.DataFrame(columns=["datetime", poll_key]))

    if all(f.empty for f in frames):
        raise RuntimeError(f"No data fetched for {city}")

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="datetime", how="outer")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Feature engineering — 04_features.py ile ayni mantik
# ---------------------------------------------------------------------------

def build_features(raw_df: pd.DataFrame, scaler) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    raw_df: datetime + pm25 + no2 + o3 (hourly)
    scaler: joblib MinMaxScaler (sehir bazinda)
    return: ((1, 48, 18) scaled float32 array, processed feature df)
    """
    df = raw_df.copy().sort_values("datetime").reset_index(drop=True)

    for poll in ["pm25", "no2", "o3"]:
        for h in [1, 6, 24, 48]:
            df[f"{poll}_lag_{h}h"] = df[poll].shift(h)

    dt = df["datetime"].dt
    df["hour"]      = dt.hour
    df["month"]     = dt.month
    df["dayofweek"] = dt.dayofweek
    for name, period in [("hour", 24), ("month", 12), ("dayofweek", 7)]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * df[name] / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * df[name] / period)

    feats = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if len(feats) < WINDOW:
        raise ValueError(f"Not enough rows for window: {len(feats)} < {WINDOW}")

    last = feats.iloc[-WINDOW:][FEATURE_COLS].values.astype(np.float32)
    scaled = scaler.transform(last).astype(np.float32)
    return scaled.reshape(1, WINDOW, N_FEATURES), feats


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

class CacheEntry:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame]      = None
        self.window: Optional[np.ndarray]        = None
        self.last_updated: Optional[datetime]    = None
        self.scaler                              = None


class CacheManager:
    def __init__(self):
        self.entries = {c: CacheEntry() for c in CITIES}
        self._task: Optional[asyncio.Task] = None
        for c in CITIES:
            path = SCALERS_DIR / f"{c}_scaler.pkl"
            self.entries[c].scaler = joblib.load(path)
        logger.info("Scalers loaded for all cities")

    async def refresh(self, city: str) -> None:
        raw = None
        try:
            raw = await fetch_latest(city)
            if raw is None or raw.empty or len(raw["pm25"].dropna()) < 48:
                raise RuntimeError("EEA returned 0 usable rows")
            window, feats = build_features(raw, self.entries[city].scaler)
            self.entries[city].raw_df       = raw
            self.entries[city].window       = window
            self.entries[city].last_updated = datetime.now(timezone.utc)
            logger.info(f"[{city}] cache refreshed")
        except Exception as e:
            logger.error(f"[{city}] refresh failed: {e}")
            self._apply_local_fallback(city)

    def _apply_local_fallback(self, city: str) -> None:
        local_path = BACKEND_DIR / ".." / "data" / "processed" / f"{city}_hourly_agg.parquet"
        if not local_path.exists():
            logger.error(f"[{city}] local fallback not found: {local_path}")
            return
        try:
            df  = pd.read_parquet(local_path).tail(96).reset_index(drop=True)
            window, feats = build_features(df, self.entries[city].scaler)
            self.entries[city].raw_df       = df
            self.entries[city].window       = window
            self.entries[city].last_updated = datetime.now(timezone.utc)
            logger.info(f"[{city}] using local fallback for {city}")
        except Exception as ex:
            logger.error(f"[{city}] local fallback failed: {ex}")

    async def refresh_all(self) -> None:
        await asyncio.gather(*(self.refresh(c) for c in CITIES))

    async def _run_loop(self) -> None:
        # Eager load
        try:
            await self.refresh_all()
        except Exception as e:
            logger.error(f"Initial refresh crashed: {e}")
        # Saatlik dongu
        while True:
            try:
                await asyncio.sleep(3600)
                await self.refresh_all()
            except asyncio.CancelledError:
                logger.info("Cache loop cancelled")
                break
            except Exception as e:
                logger.error(f"Periodic refresh crashed (continuing): {e}")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

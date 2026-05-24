"""
02_validate.py
--------------
data/processed/ altındaki şehir parquet'lerini okur; her kirletici için
istatistik hesaplar ve terminale basar. Sonuçları JSON olarak kaydeder.

Çalıştır:
    python scripts/02_validate.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "validation_report.json"

POLLUTANTS = ["pm25", "no2", "o3"]
WARNING_THRESHOLD_PCT = 30.0  # Bu değerin üzerinde WARNING bas


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, pollutant: str) -> dict:
    """
    Tek bir kirletici için istatistik sözlüğü döner.
    """
    if pollutant not in df.columns:
        return {"error": "sütun bulunamadı"}

    series = df[pollutant]
    total = len(series)
    missing = int(series.isna().sum())
    valid = total - missing
    missing_pct = (missing / total * 100) if total > 0 else 0.0

    stats: dict = {
        "total_rows": total,
        "valid_rows": valid,
        "missing_rows": missing,
        "missing_pct": round(missing_pct, 2),
    }

    if valid > 0:
        stats.update({
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
        })
    else:
        stats.update({"min": None, "max": None, "mean": None})

    # Date range
    if "datetime" in df.columns:
        valid_mask = series.notna()
        if valid_mask.any():
            dt_valid = df.loc[valid_mask, "datetime"]
            stats["date_range"] = {
                "start": str(dt_valid.min()),
                "end": str(dt_valid.max()),
            }
        else:
            stats["date_range"] = {"start": None, "end": None}

    return stats


def validate_city(parquet_path: Path) -> dict:
    """
    Tek bir şehrin parquet dosyasını okur ve tüm kirleticiler için
    istatistik sözlüğü döner.
    """
    city_name = parquet_path.stem.replace("_hourly", "")
    print(f"\n{'─' * 50}")
    print(f"  Şehir: {city_name.upper()}")
    print(f"{'─' * 50}")

    df = pd.read_parquet(parquet_path)
    print(f"  Toplam satır: {len(df):,}  |  Sütunlar: {list(df.columns)}")

    city_report: dict = {"city": city_name, "pollutants": {}}

    for pollutant in POLLUTANTS:
        stats = compute_stats(df, pollutant)
        city_report["pollutants"][pollutant] = stats

        if "error" in stats:
            print(f"\n  [{pollutant.upper()}] ⚠ {stats['error']}")
            continue

        # Terminal çıktısı
        print(f"\n  [{pollutant.upper()}]")
        print(f"    Toplam satır    : {stats['total_rows']:,}")
        print(f"    Eksik değer     : {stats['missing_rows']:,}")
        print(f"    Eksik oran      : {stats['missing_pct']:.2f}%")
        if stats["min"] is not None:
            print(f"    Min / Max / Ort : "
                  f"{stats['min']} / {stats['max']} / {stats['mean']:.4f}")
        if "date_range" in stats and stats["date_range"]["start"]:
            print(f"    Tarih aralığı   : "
                  f"{stats['date_range']['start']}  →  "
                  f"{stats['date_range']['end']}")

        # Eksik oran uyarısı
        if stats["missing_pct"] > WARNING_THRESHOLD_PCT:
            msg = (
                f"[WARNING] {city_name.upper()} / {pollutant.upper()} eksik oranı "
                f"%{stats['missing_pct']:.1f} — eşik %{WARNING_THRESHOLD_PCT:.0f}"
            )
            print(f"\n  ⚠  {msg}")
            warnings.warn(msg, stacklevel=2)

    return city_report


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("EEA Hava Kalitesi Doğrulama — 02_validate.py")
    print("=" * 60)

    parquet_files = sorted(PROCESSED_DIR.glob("*_hourly.parquet"))

    if not parquet_files:
        print(
            "\n[HATA] data/processed/ altında *_hourly.parquet bulunamadı.\n"
            "Önce 01_download.py çalıştır."
        )
        return

    print(f"\n{len(parquet_files)} şehir dosyası bulundu:")
    for p in parquet_files:
        print(f"  {p.name}")

    full_report: list[dict] = []

    for parquet_path in parquet_files:
        city_report = validate_city(parquet_path)
        full_report.append(city_report)

    # JSON raporu kaydet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Doğrulama tamamlandı.")
    print(f"  JSON rapor : {REPORT_PATH}")
    print(f"{'=' * 60}")

    # Özet uyarı tarama
    warnings_found = []
    for city_report in full_report:
        city = city_report["city"]
        for pollutant, stats in city_report["pollutants"].items():
            if isinstance(stats, dict) and "missing_pct" in stats:
                if stats["missing_pct"] > WARNING_THRESHOLD_PCT:
                    warnings_found.append(
                        f"{city.upper()} / {pollutant.upper()} "
                        f"({stats['missing_pct']:.1f}%)"
                    )

    if warnings_found:
        print(f"\n⚠  {len(warnings_found)} yüksek eksik oran uyarısı:")
        for w in warnings_found:
            print(f"  • {w}")
    else:
        print("\n✓ Tüm kirleticiler %30 eksik oran eşiğinin altında.")


if __name__ == "__main__":
    main()

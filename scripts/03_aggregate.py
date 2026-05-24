"""
03_aggregate.py
---------------
data/processed/{city}_hourly.parquet dosyalarini (istasyon seviyesi) okur,
sehir bazinda saatlik ortalama alir ve fiziksel sinir temizligi uygular:

  PM2.5 > 500  µg/m³  → NaN   (WHO/EEA olcum limiti ustu)
  NO2   > 400  µg/m³  → NaN   (EEA raporlama maksimumu ustu)
  O3    < 0    µg/m³  → NaN   (fiziksel olarak imkansiz)

Cikti: data/processed/{city}_hourly_agg.parquet
Kolonlar: datetime, pm25, no2, o3

Calistir:
    python scripts/03_aggregate.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "validation_report.json"
WARNING_THRESHOLD_PCT = 30.0
POLLUTANTS = ["pm25", "no2", "o3"]


# ---------------------------------------------------------------------------
# Aggregasyon
# ---------------------------------------------------------------------------

def aggregate_city(src_path: Path) -> pd.DataFrame:
    """
    Istasyon-seviyesi parquet'i sehir bazinda saatlik ortalamaya donusturur.
    - Ayni datetime'da birden fazla istasyon degeri varsa mean alinir.
    - NaN'lar ortalamayi etkilemez (skipna=True varsayilan).
    - Hic gecerli degeri olmayan saat satirlari NaN olarak kalir.
    """
    df = pd.read_parquet(src_path)

    avail_polls = [p for p in POLLUTANTS if p in df.columns]
    df = df[["datetime"] + avail_polls].copy()

    agg = (
        df.groupby("datetime", sort=True)[avail_polls]
        .mean()
        .reset_index()
    )

    for p in POLLUTANTS:
        if p not in agg.columns:
            agg[p] = float("nan")

    agg = agg[["datetime"] + POLLUTANTS]
    return agg


# ---------------------------------------------------------------------------
# Temizlik — Sabit fiziksel sinirlar
# ---------------------------------------------------------------------------

# (sinir, yon): "upper" → deger > sinir ise NaN, "lower" → deger < sinir ise NaN
PHYSICAL_LIMITS: dict[str, tuple[float, str]] = {
    "pm25": (500.0, "upper"),   # µg/m³ — WHO/EEA olcum limiti ustu
    "no2":  (400.0, "upper"),   # µg/m³ — EEA raporlama maksimumu ustu
    "o3":   (0.0,   "lower"),   # µg/m³ — negatif O3 fiziksel olarak imkansiz
}


def apply_physical_limits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    PHYSICAL_LIMITS tanimina gore degerleri NaN ile replace eder.
    Her kirletici icin {limit, direction, clipped_count} doner.
    """
    df = df.copy()
    info: dict[str, dict] = {}
    for poll, (limit, direction) in PHYSICAL_LIMITS.items():
        if poll not in df.columns:
            continue
        if direction == "upper":
            mask = df[poll] > limit
        else:
            mask = df[poll] < limit
        clipped = int(mask.sum())
        df.loc[mask, poll] = float("nan")
        info[poll] = {
            "limit": limit,
            "direction": direction,
            "clipped_count": clipped,
        }
    return df, info


# ---------------------------------------------------------------------------
# Validation istatistigi
# ---------------------------------------------------------------------------

def compute_stats(series: pd.Series, city: str, pollutant: str) -> dict:
    total = len(series)
    missing = int(series.isna().sum())
    valid = total - missing
    missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0

    stats: dict = {
        "total_rows": total,
        "valid_rows": valid,
        "missing_rows": missing,
        "missing_pct": missing_pct,
    }
    if valid > 0:
        stats.update({
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
        })
    else:
        stats.update({"min": None, "max": None, "mean": None})

    if missing_pct > WARNING_THRESHOLD_PCT:
        msg = (
            f"[WARNING] {city.upper()} / {pollutant.upper()} eksik oran: "
            f"%{missing_pct:.1f} (esik %{WARNING_THRESHOLD_PCT:.0f})"
        )
        print(f"  {msg}")
        warnings.warn(msg, stacklevel=3)

    return stats


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main() -> None:
    src_files = sorted(PROCESSED_DIR.glob("*_hourly.parquet"))
    # *_hourly_agg.parquet'leri exclude et
    src_files = [f for f in src_files if "_agg" not in f.name]

    if not src_files:
        print("[HATA] data/processed/ altinda *_hourly.parquet bulunamadi.")
        print("Once 01_download.py calistir.")
        return

    print("=" * 60)
    print("Sehir bazli saatlik aggregasyon — 03_aggregate.py")
    print("=" * 60)
    print(f"\n{len(src_files)} kaynak dosya:")
    for f in src_files:
        print(f"  {f.name}")

    full_report: list[dict] = []

    for src in src_files:
        city = src.stem.replace("_hourly", "")
        print(f"\n{'─'*52}")
        print(f"  [{city.upper()}]  {src.name}")
        print(f"{'─'*52}")

        # Adim 1: Aggregasyon (istasyon → sehir saatlik ort)
        agg_df = aggregate_city(src)
        n_hours = len(agg_df)
        print(f"  Aggregasyon: {n_hours:,} saatlik satir")
        print(f"  Tarih: {agg_df['datetime'].min()} → {agg_df['datetime'].max()}")

        # NaN sayilari: temizlik oncesi
        nan_before = {p: int(agg_df[p].isna().sum()) for p in POLLUTANTS if p in agg_df.columns}

        # Temizlik: fiziksel sinirlar
        agg_df, phys_info = apply_physical_limits(agg_df)

        print(f"\n  [Temizlik] Fiziksel sinir uygulamasi:")
        for poll, info in phys_info.items():
            sign = ">" if info["direction"] == "upper" else "<"
            marker = f"  <- {info['clipped_count']} deger NaN" if info["clipped_count"] else "  (etkilenen yok)"
            print(f"    {poll}: {sign} {info['limit']}{marker}")

        # Before/after NaN karsilastirmasi
        nan_after = {p: int(agg_df[p].isna().sum()) for p in POLLUTANTS if p in agg_df.columns}
        print(f"\n  [NaN degisim ozeti]")
        print(f"    {'Kirletici':<8} | {'Once':>8} | {'Sonra':>8} | {'Fark':>8} | {'Sonra%':>8}")
        print(f"    {'-'*50}")
        for poll in POLLUTANTS:
            b = nan_before.get(poll, 0)
            a = nan_after.get(poll, 0)
            pct = a / n_hours * 100 if n_hours else 0
            diff_str = f"+{a-b}" if a > b else str(a - b)
            print(f"    {poll:<8} | {b:>8,} | {a:>8,} | {diff_str:>8} | {pct:>7.2f}%")

        # Interpolasyon: max 3 ardisik bosluk
        agg_df[POLLUTANTS] = agg_df[POLLUTANTS].interpolate(method="linear", limit=3)

        # Kaydet
        out_path = PROCESSED_DIR / f"{city}_hourly_agg.parquet"
        agg_df.to_parquet(out_path, index=False)
        print(f"\n  -> {out_path.name} kaydedildi ({out_path.stat().st_size//1024} KB)")

        # Validation istatistigi
        city_report: dict = {
            "city": city,
            "source": src.name,
            "cleaning": {"physical_limits": phys_info},
            "pollutants": {},
        }
        for poll in POLLUTANTS:
            stats = compute_stats(agg_df[poll], city, poll)
            city_report["pollutants"][poll] = stats

        full_report.append(city_report)

    # JSON guncelle
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Tamamlandi. Rapor: {REPORT_PATH}")
    print(f"{'='*60}")

    # Ozet tablo
    print("\n  Sehir       | PM2.5 eksik | NO2 eksik | O3 eksik")
    print("  " + "-"*50)
    for r in full_report:
        p = r["pollutants"]
        pm = p["pm25"]["missing_pct"]
        no = p["no2"]["missing_pct"]
        o3 = p["o3"]["missing_pct"]
        flag = lambda x: f"{x:5.1f}% {'⚠' if x > WARNING_THRESHOLD_PCT else ' '}"
        print(f"  {r['city']:<11} | {flag(pm)} | {flag(no)} | {flag(o3)}")


if __name__ == "__main__":
    main()

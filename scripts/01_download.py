"""
01_download.py
--------------
EEA Air Quality E-Reporting API'den Berlin, Hamburg, Muenchen ve Koeln icin
PM2.5, NO2 ve O3 saatlik verisi indirir (2020-2023, E1a verified).

Calistir:
    python scripts/01_download.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

BASE_URL = "https://eeadmz1-downloads-api-appservice.azurewebsites.net"
ASYNC_ENDPOINT = f"{BASE_URL}/ParquetFile/async"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL_SEC = 10
MAX_POLL_ATTEMPTS = 20

# ---------------------------------------------------------------------------
# API istek govdesi (gercek API davranimi: "hour" string, dataset string)
# ---------------------------------------------------------------------------

REQUEST_BODY = {
    "countries": ["DE"],
    "cities": ["Berlin", "Hamburg", "München", "Köln"],  # API umlaut bekliyor
    "pollutants": ["PM2.5", "NO2", "O3"],
    "dataset": "2",            # E1a verified — API string bekliyor
    "dateTimeStart": "2020-01-01T00:00:00Z",
    "dateTimeEnd":   "2023-12-31T23:00:00Z",
    "aggregationType": "hour", # API "hour"/"day"/"var" kabul eder (integer degil)
}

# ---------------------------------------------------------------------------
# Istasyon kodu → sehir eslesmesi
# EEA station ID format: DE_DEBE067 = DE(country) + BE(state) + E067(number)
# Gercek API filtrelemesi yaptigi icin DEBY=Munich, DENW=Cologne cikarsimi gecerli.
# ---------------------------------------------------------------------------

STATION_PREFIX_MAP: dict[str, str] = {
    "DEBE": "berlin",   # Berlin (city-state)
    "DEHH": "hamburg",  # Hamburg (city-state)
    "DEBY": "munich",   # Bayern — API Munich filtreledigi icin
    "DEMU": "munich",   # Alternatif Muenchen kodu (bazi istasyonlarda)
    "DENW": "cologne",  # NRW — API Cologne filtreledigi icin
    "DECK": "cologne",  # Alternatif Koeln kodu
}

# Dosya adindaki kirletici tanima: regex ile eslesir
POLLUTANT_FROM_FILENAME: list[tuple[re.Pattern, str]] = [
    # EEA dosya adinda PM2.5 → "_PM2_" veya "_PM2.5_" veya "_PM25_" seklinde gelebilir
    (re.compile(r"_PM2(?:[\._]?5)?_", re.IGNORECASE), "pm25"),
    # "_NO2_" — \b calismaz cunku _ de word-char sayilir
    (re.compile(r"_NO2_", re.IGNORECASE), "no2"),
    (re.compile(r"_O3_", re.IGNORECASE), "o3"),
]


# ---------------------------------------------------------------------------
# Adim 1: Async istek gonder
# ---------------------------------------------------------------------------

def submit_async_request() -> str:
    """
    EEA async endpoint'ine POST atar, blob ZIP URL'sini doner.

    Gercek API davranisi:
      - POST → 200 + text/plain → yanit govdesi dogrudan blob URL (duz metin)
      - 206 → istek 600 MB sinirini asiyor
      - 400 → yanlis parametre (aggregationType/dataset tip hatasi vb.)
      - 5xx → sunucu hatasi
    """
    print("-> EEA API'ye async istek gonderiliyor...")
    print(f"   Endpoint : {ASYNC_ENDPOINT}")
    print(f"   Sehirler : {REQUEST_BODY['cities']}")
    print(f"   Tarih    : {REQUEST_BODY['dateTimeStart']} - {REQUEST_BODY['dateTimeEnd']}")

    try:
        resp = requests.post(
            ASYNC_ENDPOINT,
            json=REQUEST_BODY,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
    except requests.exceptions.Timeout:
        sys.exit("\n[HATA] API istegi zaman asimina ugradi (60s). Tekrar dene.")
    except requests.exceptions.ConnectionError as e:
        sys.exit(f"\n[HATA] Baglanti kurulamadi: {e}")

    if resp.status_code == 206:
        sys.exit(
            "\n[HATA] API 206 Partial Content dondurdu.\n"
            "Istenen veri 600 MB sinirini asiyor olmali.\n"
            "Lutfen tarih araligini veya sehir/kirletici sayisini azalt."
        )

    if resp.status_code not in (200, 202):
        sys.exit(
            f"\n[HATA] API beklenmedik durum kodu: {resp.status_code}\n"
            f"Yanit: {resp.text[:600]}"
        )

    # API duz metin olarak blob URL doner; JSON da olabilir
    blob_url = resp.text.strip().strip('"')
    if blob_url.startswith("{"):
        try:
            data = json.loads(blob_url)
            blob_url = (
                data.get("url")
                or data.get("downloadUrl")
                or data.get("fileUrl")
                or ""
            )
        except Exception:
            pass

    if not blob_url or not blob_url.startswith("http"):
        sys.exit(
            f"\n[HATA] API yanitindan indirme URL'si cikarilamadi.\n"
            f"Ham yanit: {resp.text[:500]}"
        )

    print(f"   Blob URL : {blob_url}")
    return blob_url


# ---------------------------------------------------------------------------
# Adim 2: Polling — blob hazir olana kadar bekle
# ---------------------------------------------------------------------------

def poll_until_ready(blob_url: str) -> bytes:
    """
    blob_url'yi POLL_INTERVAL_SEC saniyede bir kontrol eder.

    Gercek API davranisi:
      - 404 → blob henuz olusturulmadi (hazir degil), bekle
      - 200 + icerik > 0 → ZIP hazir
      - 202 / 200+bos → bekle

    MAX_POLL_ATTEMPTS sonra timeout ile cikar.
    """
    print(f"\n-> Blob hazirlanıyor, {POLL_INTERVAL_SEC}s aralikla kontrol ediliyor...")

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        print(f"   Deneme {attempt}/{MAX_POLL_ATTEMPTS} ...", end=" ", flush=True)
        try:
            resp = requests.get(blob_url, timeout=180)
        except requests.exceptions.Timeout:
            print(f"zaman asimi! {POLL_INTERVAL_SEC}s bekleniyor.")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        if resp.status_code == 200 and len(resp.content) > 0:
            size_mb = len(resp.content) / (1024 ** 2)
            print(f"hazir! {size_mb:.1f} MB")
            return resp.content

        if resp.status_code == 404:
            print(f"404 (henuz hazir degil). {POLL_INTERVAL_SEC}s bekleniyor.")
        elif resp.status_code == 202:
            print(f"202 (isleniyor). {POLL_INTERVAL_SEC}s bekleniyor.")
        elif resp.status_code == 200 and len(resp.content) == 0:
            print(f"200 ama bos. {POLL_INTERVAL_SEC}s bekleniyor.")
        else:
            print(f"{resp.status_code} — beklenmedik durum. {POLL_INTERVAL_SEC}s bekleniyor.")

        time.sleep(POLL_INTERVAL_SEC)

    sys.exit(
        f"\n[HATA] {MAX_POLL_ATTEMPTS} deneme sonunda blob hazir olmadi.\n"
        f"API cok yavas olabilir — daha sonra tekrar dene."
    )


# ---------------------------------------------------------------------------
# Adim 3: ZIP'i kaydet ve ac
# ---------------------------------------------------------------------------

def save_and_extract_zip(zip_bytes: bytes) -> list[Path]:
    """
    ZIP'i data/raw/ altina yazar, icindeki .parquet dosyalarini cikarir.
    Cikarilan .parquet Path listesini doner.
    """
    zip_path = RAW_DIR / "eea_download.zip"
    zip_path.write_bytes(zip_bytes)
    print(f"\n-> ZIP kaydedildi: {zip_path}  ({len(zip_bytes) / 1024**2:.1f} MB)")

    parquet_paths: list[Path] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"   ZIP icindeki dosyalar ({len(names)} adet):")

        for name in names:
            print(f"     {name}")
            if name.lower().endswith(".parquet"):
                # Alt klasor yapısını korumadan RAW_DIR'a yaz
                flat_name = Path(name).name
                out_path = RAW_DIR / flat_name
                out_path.write_bytes(zf.read(name))
                parquet_paths.append(out_path)

    print(f"\n   {len(parquet_paths)} parquet dosyasi data/raw/ altina yazildi.")
    return parquet_paths


# ---------------------------------------------------------------------------
# Yardimci: Dosya adindan kirletici tespit et
# ---------------------------------------------------------------------------

def _pollutant_from_filename(filename: str) -> str | None:
    for pattern, name in POLLUTANT_FROM_FILENAME:
        if pattern.search(filename):
            return name
    return None


# ---------------------------------------------------------------------------
# Yardimci: Samplingpoint sutunundan istasyon kodunu cikar
# "DE/SPO.DE_DEBE067_NO2_dataGroup1" → "DEBE067"
# ---------------------------------------------------------------------------

def _station_code_from_samplingpoint(sp_value: str) -> str | None:
    """
    Samplingpoint'ten 4-harfli il/sehir on-ekini cikartir.
    Ornek: "DE/SPO.DE_DEBE067_NO2_dataGroup1" → "DEBE"
    """
    # DE_ ile baslayan bolumu bul
    m = re.search(r"DE_([A-Z]{4})", str(sp_value))
    if m:
        return m.group(1)  # "DEBE", "DEHH", "DEBY", "DENW" ...
    return None


# ---------------------------------------------------------------------------
# Yardimci: Parquet dosyasindan sehir adini cikart
# ---------------------------------------------------------------------------

def _city_from_file(df: pd.DataFrame, filename: str) -> str | None:
    """
    Oncelik: Samplingpoint sutunundan istasyon kodu → sehir
    Yedek   : Dosya adindaki sehir belirteci
    """
    # Samplingpoint sutunundan dene
    sp_col = next(
        (c for c in df.columns if c.lower() == "samplingpoint"), None
    )
    if sp_col and len(df) > 0:
        sample_sp = str(df[sp_col].iloc[0])
        prefix = _station_code_from_samplingpoint(sample_sp)
        if prefix and prefix in STATION_PREFIX_MAP:
            return STATION_PREFIX_MAP[prefix]

    # Dosya adindan dene (backup)
    fname_lower = filename.lower()
    for code, city in STATION_PREFIX_MAP.items():
        if code.lower() in fname_lower:
            return city

    return None


# ---------------------------------------------------------------------------
# Adim 4: Sehir bazinda birlestir ve temizle
# ---------------------------------------------------------------------------

def merge_and_clean(parquet_paths: list[Path]) -> None:
    """
    - Her dosyayi okur; Validity < 1 satirlari dusuruecek.
    - Sutunlari normalize eder: Start → datetime, Value → {pm25/no2/o3}
    - Dosyalari sehir bazinda gruplar.
    - Ayni sehrin farkli kirleticileri: kirletici bazinda wide pivot.
    - Sonucu data/processed/{city}_hourly.parquet olarak kayder.

    Cikti sutunlari:
        datetime, station, pm25, no2, o3
        (istasyonun olcmedigi kirleticide NaN)
    """
    print("\n-> Parquet dosyalari sehir bazinda birlestiriliyor...")

    # Her dosyayi oku, sehir/kirletici bilgisini ekle
    # Structure: city → pollutant → list of DataFrames
    city_poll_frames: dict[str, dict[str, list[pd.DataFrame]]] = {}
    ungrouped: list[str] = []

    for p in parquet_paths:
        df = pd.read_parquet(p)

        # --- Validity filtresi ---
        validity_col = next(
            (c for c in df.columns if c.lower() == "validity"), None
        )
        if validity_col:
            before = len(df)
            df = df[df[validity_col] >= 1].copy()
            dropped = before - len(df)
            if dropped:
                print(f"   {p.name}: Validity < 1 → {dropped} satir dusuruldu.")
        else:
            print(f"   [UYARI] 'Validity' sutunu bulunamadi: {p.name}")

        if df.empty:
            print(f"   [UYARI] {p.name} Validity filtresi sonrasi bos kaldi.")
            continue

        # --- Sehir tespiti ---
        city = _city_from_file(df, p.name)
        if city is None:
            ungrouped.append(p.name)
            print(f"   [UYARI] Sehir tespit edilemedi: {p.name}")
            continue

        # --- Kirletici tespiti (dosya adindan) ---
        pollutant = _pollutant_from_filename(p.name)
        if pollutant is None:
            print(f"   [UYARI] Kirletici tespit edilemedi: {p.name}")
            continue

        # --- Sutun normalizasyonu ---
        # Start → datetime  |  Value → kirletici adi  |  Samplingpoint → station
        rename_map: dict[str, str] = {}
        for col in df.columns:
            col_l = col.lower()
            if col_l in ("start", "datetimebegin", "datetime_begin", "starttime"):
                rename_map[col] = "datetime"
            elif col_l == "value":
                rename_map[col] = pollutant
            elif col_l == "samplingpoint":
                rename_map[col] = "station"

        df = df.rename(columns=rename_map)

        # datetime parse
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        # Value (simdi pollutant) string → float
        if pollutant in df.columns:
            df[pollutant] = pd.to_numeric(df[pollutant], errors="coerce")

        # Sadece gerekli sutunlari tut
        keep_cols = ["datetime", "station", pollutant]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Istasyon kodunu kisalt: "DE/SPO.DE_DEBE067_NO2_dataGroup1" → "DEBE067"
        if "station" in df.columns:
            def _shorten_station(s: str) -> str:
                m = re.search(r"DE_([A-Z]{2}[A-Z0-9]+?)(?:_|$)", str(s))
                return m.group(1) if m else str(s)
            df["station"] = df["station"].apply(_shorten_station)

        # Grupla
        city_poll_frames.setdefault(city, {}).setdefault(pollutant, []).append(df)

    if ungrouped:
        print(f"\n   [UYARI] Sehir tespit edilemeyen {len(ungrouped)} dosya atlandi:")
        for name in ungrouped:
            print(f"      {name}")

    # Her sehir icin birlestir ve pivot et
    for city, poll_dict in city_poll_frames.items():
        print(f"\n   [{city.upper()}] isleniyor ...")

        # Kirletici bazinda concat et
        poll_dfs: dict[str, pd.DataFrame] = {}
        for pollutant, frames in poll_dict.items():
            merged = pd.concat(frames, ignore_index=True)
            print(f"      {pollutant}: {len(frames)} dosya, {len(merged):,} satir")
            poll_dfs[pollutant] = merged

        if not poll_dfs:
            print(f"   [UYARI] {city} icin islenecek veri yok.")
            continue

        # Tum kirleticileri datetime+station uzerinde dis birlestir
        # (her kirleticinin kendi DataFrame'i var; bazi istasyonlar birden fazla olcer)
        all_polls = list(poll_dfs.values())
        city_df = all_polls[0]

        for df_right in all_polls[1:]:
            # datetime+station anahtarla disbirlesim — NaN kalsin
            city_df = pd.merge(
                city_df, df_right,
                on=["datetime", "station"],
                how="outer",
            )

        # Eksik kirletici sutunlari ekle (kirletici hicbir istasyonda yoksa)
        for poll in ("pm25", "no2", "o3"):
            if poll not in city_df.columns:
                city_df[poll] = float("nan")

        city_df = city_df.sort_values(["datetime", "station"]).reset_index(drop=True)

        # Kaydet
        out_path = PROCESSED_DIR / f"{city}_hourly.parquet"
        city_df.to_parquet(out_path, index=False)

        size_kb = out_path.stat().st_size / 1024
        print(f"      -> {out_path.name} kaydedildi: "
              f"{len(city_df):,} satir x {len(city_df.columns)} sutun "
              f"({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("=" * 62)
    print("EEA Hava Kalitesi Indirme — 01_download.py")
    print("=" * 62)

    blob_url = submit_async_request()
    zip_bytes = poll_until_ready(blob_url)
    parquet_paths = save_and_extract_zip(zip_bytes)

    if not parquet_paths:
        sys.exit(
            "\n[HATA] ZIP icinde hic .parquet dosyasi bulunamadi.\n"
            "API yanitini kontrol et: data/raw/eea_download.zip"
        )

    merge_and_clean(parquet_paths)

    processed = sorted(PROCESSED_DIR.glob("*_hourly.parquet"))
    print("\n" + "=" * 62)
    print("Indirme tamamlandi.")
    print(f"  Ham veriler  : {RAW_DIR}")
    print(f"  Islenenmis   : {PROCESSED_DIR}")
    print(f"  Olusturulan  : {[p.name for p in processed]}")
    print("  Simdi 02_validate.py calistirilabilir.")
    print("=" * 62)


if __name__ == "__main__":
    main()

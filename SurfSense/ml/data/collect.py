"""
Historical data collection pipeline.

Downloads hourly marine + weather data for five geographically diverse spots
(thesis Section 3.3.5) and merges them into ml/data/processed/historical.parquet.

Sources:
  Marine (wave height, period, direction): Open-Meteo Marine API (free, no key)
  Weather (wind, temperature):             Open-Meteo Archive API (free, no key)
  Tides: not available from Open-Meteo — tide_height_m left as NaN and
         imputed to spot preferred value during training (documented in thesis
         data-provenance note as a known limitation).

Usage:
    python -m ml.data.collect                   # full 2-year collection
    python -m ml.data.collect --years 1         # shorter run for testing
    python -m ml.data.collect --spot pipeline   # single spot
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

# ---------------------------------------------------------------------------
# Spot registry
# ---------------------------------------------------------------------------
SPOTS: dict[str, dict] = {
    "pipeline":     {"lat": 21.6650,  "lon": -158.0539, "tz": "Pacific/Honolulu"},
    "hossegor":     {"lat": 43.6647,  "lon":   -1.4240, "tz": "Europe/Paris"},
    "ericeira":     {"lat": 38.9625,  "lon":   -9.4167, "tz": "Europe/Lisbon"},
    "jeffreys_bay": {"lat": -34.0486, "lon":   24.9209, "tz": "Africa/Johannesburg"},
    "gold_coast":   {"lat": -28.0026, "lon":  153.4300, "tz": "Australia/Brisbane"},
}

RAW_DIR = Path("ml/data/raw")
PROCESSED_DIR = Path("ml/data/processed")

MARINE_VARS = [
    "wave_height", "wave_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "wind_wave_height", "wind_wave_period",
    "sea_surface_temperature",
]
WEATHER_VARS = [
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "temperature_2m",
]

MARINE_URL  = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

_RENAME_MARINE = {
    "wave_height":              "wave_height_m",
    "wave_period":              "wave_period_s",
    "wave_direction":           "wave_direction_deg",
    "swell_wave_height":        "swell_height_m",
    "swell_wave_period":        "swell_period_s",
    "swell_wave_direction":     "swell_direction_deg",
    "wind_wave_height":         "wind_wave_height_m",
    "wind_wave_period":         "wind_wave_period_s",
    "sea_surface_temperature":  "water_temp_c",
}
_RENAME_WEATHER = {
    "wind_speed_10m":       "wind_speed_kph",
    "wind_direction_10m":   "wind_direction_deg",
    "wind_gusts_10m":       "wind_gust_kph",
    "temperature_2m":       "air_temp_c",
}


def _fetch(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = httpx.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def _hourly_to_df(data: dict, rename: dict) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    df = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
    for api_key, col_name in rename.items():
        if api_key in hourly:
            df[col_name] = hourly[api_key]
    return df


def fetch_spot_year(spot_id: str, meta: dict, start: date, end: date) -> pd.DataFrame:
    """Download marine + weather for one spot over [start, end] and return merged DataFrame."""
    params_base = {
        "latitude":   meta["lat"],
        "longitude":  meta["lon"],
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "timezone":   "UTC",
    }

    marine_data  = _fetch(MARINE_URL,  {**params_base, "hourly": ",".join(MARINE_VARS)})
    time.sleep(0.2)  # pace requests to the same API provider
    weather_data = _fetch(WEATHER_URL, {**params_base, "hourly": ",".join(WEATHER_VARS)})

    df_m = _hourly_to_df(marine_data,  _RENAME_MARINE)
    df_w = _hourly_to_df(weather_data, _RENAME_WEATHER)

    df = df_m.merge(df_w, on="timestamp", how="outer").sort_values("timestamp")
    df["spot_id"]  = spot_id
    df["lat"]      = meta["lat"]
    df["lon"]      = meta["lon"]
    df["tide_height_m"] = float("nan")  # not available from Open-Meteo

    return df


def collect(years: int = 2, spot_filter: list[str] | None = None, force: bool = False) -> None:
    """Download and merge historical data for all (or selected) spots.

    Args:
        years: Number of years of history to collect (ending yesterday).
        spot_filter: Optional list of spot IDs to collect; defaults to all.
        force: Re-download even if a raw cache file already exists.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    spots = {k: v for k, v in SPOTS.items() if not spot_filter or k in spot_filter}
    all_frames: list[pd.DataFrame] = []

    for spot_id, meta in spots.items():
        raw_path = RAW_DIR / f"{spot_id}.parquet"
        if raw_path.exists() and not force:
            print(f"  [{spot_id}] loading cached raw data from {raw_path}")
            df = pd.read_parquet(raw_path)
        else:
            print(f"  [{spot_id}] downloading {start_date} → {end_date} …")
            df = fetch_spot_year(spot_id, meta, start_date, end_date)
            df.to_parquet(raw_path, index=False)
            print(f"  [{spot_id}] saved {len(df):,} rows → {raw_path}")
        all_frames.append(df)

    historical = pd.concat(all_frames, ignore_index=True)
    historical = historical.drop_duplicates(subset=["spot_id", "timestamp"])
    historical = historical.sort_values(["spot_id", "timestamp"]).reset_index(drop=True)

    out = PROCESSED_DIR / "historical.parquet"
    historical.to_parquet(out, index=False)

    print(f"\nMerged dataset: {len(historical):,} rows → {out}")
    print(f"Rows per spot:  {historical['spot_id'].value_counts().to_dict()}")
    print(f"Date range:     {historical['timestamp'].min()} → {historical['timestamp'].max()}")

    # Missing-value rates (required for EDA acceptance gate)
    null_rates = (historical.isnull().mean() * 100).round(1)
    cols_with_nulls = null_rates[null_rates > 0]
    if cols_with_nulls.empty:
        print("Missing values: none")
    else:
        print("Missing value rates (%):")
        for col, rate in cols_with_nulls.items():
            print(f"  {col}: {rate}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--spot",  type=str, default=None, help="Single spot ID to collect")
    parser.add_argument("--force", action="store_true", help="Re-download even if raw cache exists")
    args = parser.parse_args()
    collect(years=args.years, spot_filter=[args.spot] if args.spot else None, force=args.force)

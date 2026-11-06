"""
Feature extractor for the surf condition ML model.

FEATURE_NAMES is the single source of truth for the feature list.  Both
training-time extraction (from a historical DataFrame row) and inference-time
extraction (from a live ForecastPoint) use the same code paths, so drift
between train and serve is caught by tests/test_feature_extractor.py.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from app.forecasting.models import ForecastPoint, TideState

# ---------------------------------------------------------------------------
# Feature list – never modify one path without updating the other
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "wave_height_min",
    "wave_height_max",
    "wave_height_avg",
    "swell_height",
    "swell_period",
    "swell_dir_sin",
    "swell_dir_cos",
    "wind_speed",
    "wind_gust",
    "wind_dir_sin",
    "wind_dir_cos",
    "is_offshore",
    "is_light_wind",
    "tide_height",
    "tide_is_rising",
    "wave_energy_proxy",       # swell_height² × swell_period
    "wind_wave_interaction",   # wind_speed × cos(wind_dir − swell_dir)
    "swell_wind_ratio",        # swell_period / max(wind_speed, 1)
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
    "water_temp",
    "air_temp",
    "skill_level_encoded",     # 0=beginner 1=intermediate 2=advanced 3=expert
]

_SKILL_ENCODING: dict[str, int] = {
    "beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3,
}

# Cardinal direction → degrees (used when reconstructing from dict)
WIND_DIR_TO_DEG: dict[str, float] = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}
SWELL_DIR_TO_DEG: dict[str, float] = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sincos(degrees: float) -> tuple[float, float]:
    r = math.radians(degrees)
    return math.sin(r), math.cos(r)


def _temporal(ts: datetime) -> tuple[float, float, float, float, int]:
    hs = math.sin(2 * math.pi * ts.hour / 24)
    hc = math.cos(2 * math.pi * ts.hour / 24)
    ms = math.sin(2 * math.pi * ts.month / 12)
    mc = math.cos(2 * math.pi * ts.month / 12)
    return hs, hc, ms, mc, ts.weekday()


def _wind_wave_interaction(wind_speed: float, wind_deg: float, swell_deg: float) -> float:
    return wind_speed * math.cos(math.radians(wind_deg - swell_deg))


def _is_offshore_approx(wind_deg: float, swell_deg: float) -> float:
    """Approximate offshore check: wind blowing away from swell source."""
    diff = (wind_deg - (swell_deg + 180.0)) % 360.0
    return float(diff < 90.0 or diff > 270.0)


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class ForecastPointFeatureExtractor:
    """Produce a fixed-length float32 feature vector from different input types.

    Entry points:
      transform_point(ForecastPoint, skill_level)  – inference path
      transform_row(pd.Series, skill_level)         – training path
      transform_batch(pd.DataFrame, skill_level)    – bulk training

    Both point and row paths produce a vector of length len(FEATURE_NAMES).
    """

    # -- Inference path ------------------------------------------------------

    def transform_point(
        self, fp: ForecastPoint, skill_level: str = "intermediate"
    ) -> np.ndarray:
        """Extract features from a live ForecastPoint."""
        sd = fp.swell.direction_degrees
        wd = fp.wind.direction_degrees
        sd_sin, sd_cos = _sincos(sd)
        wd_sin, wd_cos = _sincos(wd)
        hs, hc, ms, mc, dow = _temporal(fp.timestamp)

        tide_h = fp.tide.height if fp.tide else 0.0
        tide_r = float(fp.tide.state == TideState.RISING) if fp.tide else 0.0
        w_temp = (fp.weather.water_temperature or 0.0) if fp.weather else 0.0
        a_temp = (fp.weather.temperature or 0.0) if fp.weather else 0.0
        gust   = fp.wind.gust if fp.wind.gust is not None else fp.wind.speed

        sh, st, ws = fp.swell.height, fp.swell.period, fp.wind.speed

        return np.array([
            fp.waves.height_min,
            fp.waves.height_max,
            fp.waves.height_avg,
            sh, st,
            sd_sin, sd_cos,
            ws, gust,
            wd_sin, wd_cos,
            float(fp.is_offshore_wind),
            float(fp.is_light_wind),
            tide_h, tide_r,
            sh ** 2 * st,
            _wind_wave_interaction(ws, wd, sd),
            st / max(ws, 1.0),
            hs, hc, ms, mc, float(dow),
            w_temp, a_temp,
            float(_SKILL_ENCODING.get(skill_level.lower(), 1)),
        ], dtype=np.float32)

    # -- Training path -------------------------------------------------------

    def transform_row(
        self, row, skill_level: str = "intermediate"
    ) -> np.ndarray:
        """Extract features from a historical DataFrame row.

        Expected columns (NaN-tolerant; missing → 0.0):
            wave_height_m, swell_height_m, swell_period_s, swell_direction_deg,
            wind_speed_kph, wind_gust_kph, wind_direction_deg,
            tide_height_m, water_temp_c, air_temp_c, timestamp.
        """
        def _f(key: str, default: float = 0.0) -> float:
            val = row.get(key) if hasattr(row, "get") else getattr(row, key, default)
            if val is None:
                return default
            try:
                f = float(val)
                return default if math.isnan(f) else f
            except (TypeError, ValueError):
                return default

        raw_ts = row.get("timestamp") if hasattr(row, "get") else getattr(row, "timestamp", None)
        if isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts)
        elif isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            ts = datetime(2000, 1, 1)

        wave_avg = _f("wave_height_m")
        wave_min = wave_avg * 0.85
        wave_max = wave_avg * 1.15
        sh  = _f("swell_height_m")
        st  = _f("swell_period_s")
        sd  = _f("swell_direction_deg")
        ws  = _f("wind_speed_kph")
        wg  = _f("wind_gust_kph") or ws
        wd  = _f("wind_direction_deg")
        th  = _f("tide_height_m")
        wt  = _f("water_temp_c")
        at  = _f("air_temp_c")

        sd_sin, sd_cos = _sincos(sd)
        wd_sin, wd_cos = _sincos(wd)
        hs, hc, ms, mc, dow = _temporal(ts)

        return np.array([
            wave_min, wave_max, wave_avg,
            sh, st,
            sd_sin, sd_cos,
            ws, wg,
            wd_sin, wd_cos,
            _is_offshore_approx(wd, sd),
            float(ws < 15.0),
            th, 0.0,  # tide_is_rising not derivable from single snapshot
            sh ** 2 * st,
            _wind_wave_interaction(ws, wd, sd),
            st / max(ws, 1.0),
            hs, hc, ms, mc, float(dow),
            wt, at,
            float(_SKILL_ENCODING.get(skill_level.lower(), 1)),
        ], dtype=np.float32)

    def transform_batch(
        self, df, skill_level: str = "intermediate"
    ) -> np.ndarray:
        """Apply transform_row to every row of a DataFrame.

        Returns array of shape (N, len(FEATURE_NAMES)).
        """
        return np.stack([self.transform_row(row, skill_level) for _, row in df.iterrows()])

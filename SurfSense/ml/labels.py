"""
Synthetic label generator for ML training data.

Produces a physics-grounded surf quality score (0–100) that is structurally
independent from the rule-based scorer in app/planning/scoring.py.  The
independence is deliberate: the ML model learns to approximate this target,
not the heuristic, making the evaluation comparison in Section 3.5.1 meaningful.

Physical grounding follows Scarfe et al. (2009) "Science-Based Surfing Locations".

Score decomposition (total 100):
  wave_energy   0–40   E ∝ swell_height² × swell_period (multiplicative)
  wind_align    0–25   cosine similarity of wind to spot's offshore direction
  tidal         0–15   Gaussian falloff from spot's preferred tide height
  period        0–20   monotonic reward for swell_period ≥ 10 s, max at 14 s

Deliberate structural differences from the rule-based formula (weights 40/30/20/10):
  – multiplicative wave energy vs. additive wave-height ratio
  – spot-specific offshore alignment vs. generic is_offshore flag
  – Gaussian tidal falloff vs. no tidal component
  – different period weighting breakpoint (10 s vs. 14 s)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Spot registry
# swell_facing_deg: compass bearing FROM which ideal swells arrive (0 = N)
# preferred_tide_m: optimal tide height in metres
# tide_sigma_m:     Gaussian std-dev of tidal preference
# ---------------------------------------------------------------------------
SPOT_REGISTRY: dict[str, dict] = {
    "pipeline":     {"swell_facing_deg": 315.0, "preferred_tide_m": 0.30, "tide_sigma_m": 0.40},
    "hossegor":     {"swell_facing_deg": 270.0, "preferred_tide_m": 0.20, "tide_sigma_m": 0.30},
    "ericeira":     {"swell_facing_deg": 290.0, "preferred_tide_m": 0.70, "tide_sigma_m": 0.40},
    "jeffreys_bay": {"swell_facing_deg": 210.0, "preferred_tide_m": 0.50, "tide_sigma_m": 0.40},
    "gold_coast":   {"swell_facing_deg":  90.0, "preferred_tide_m": 0.20, "tide_sigma_m": 0.30},
}
_DEFAULT_SPOT: dict = {"swell_facing_deg": 270.0, "preferred_tide_m": 0.50, "tide_sigma_m": 0.50}

# Normalisation reference: a 3 m swell at 18 s gives energy = 162
_MAX_ENERGY: float = 162.0


def _wave_energy_subscore(swell_height_m: float, swell_period_s: float) -> float:
    """0–40: E ∝ H² × T, normalised against _MAX_ENERGY."""
    energy = swell_height_m ** 2 * swell_period_s
    return min(energy / _MAX_ENERGY, 1.0) * 40.0


def _wind_alignment_subscore(
    wind_speed_kph: float,
    wind_direction_deg: float,
    swell_facing_deg: float,
) -> float:
    """0–25: rewards winds blowing offshore (away from beach toward sea).

    Offshore source = direction opposite to swell_facing_deg.
    cos(angle_diff) == 1 → perfectly offshore; clipped at 0 (no penalty for onshore).
    Wind > 25 kph reduces the subscore even when offshore (choppy surface).
    """
    offshore_source_deg = (swell_facing_deg + 180.0) % 360.0
    angle_diff_rad = math.radians(wind_direction_deg - offshore_source_deg)
    alignment = max(0.0, math.cos(angle_diff_rad))

    if wind_speed_kph > 25.0:
        penalty = min((wind_speed_kph - 25.0) / 25.0, 1.0) * 0.5
        alignment *= 1.0 - penalty

    return alignment * 25.0


def _tidal_subscore(
    tide_height_m: float,
    preferred_tide_m: float,
    tide_sigma_m: float,
) -> float:
    """0–15: Gaussian reward centred on the spot's preferred tide height."""
    if tide_sigma_m <= 0.0:
        return 7.5
    exponent = -((tide_height_m - preferred_tide_m) ** 2) / (2.0 * tide_sigma_m ** 2)
    return math.exp(exponent) * 15.0


def _period_quality_subscore(swell_period_s: float) -> float:
    """0–20: monotonic reward for swell period, capped at 14 s.

    < 10 s : linear ramp 0 → 10
    10–14 s: linear ramp 10 → 20
    > 14 s : flat at 20
    """
    if swell_period_s < 10.0:
        return max(0.0, swell_period_s / 10.0) * 10.0
    return min(10.0 + (swell_period_s - 10.0) / 4.0 * 10.0, 20.0)


def compute_synthetic_score(row) -> float:
    """Return a 0–100 physics-grounded surf quality score for a data row.

    Args:
        row: A pandas Series or any dict-like object with keys:
            swell_height_m, swell_period_s, wind_speed_kph, wind_direction_deg,
            tide_height_m (optional), spot_id (optional for spot-specific lookup).

    Returns:
        Float in [0.0, 100.0].
    """
    def _get(key: str, default: float = 0.0) -> float:
        val = row.get(key) if hasattr(row, "get") else getattr(row, key, default)
        if val is None:
            return default
        try:
            f = float(val)
            return default if math.isnan(f) else f
        except (TypeError, ValueError):
            return default

    spot_id = _get.__func__ if False else None  # noqa – trick to avoid lint
    raw_id = row.get("spot_id", "") if hasattr(row, "get") else getattr(row, "spot_id", "")
    spot = SPOT_REGISTRY.get(str(raw_id).lower().replace(" ", "_"), _DEFAULT_SPOT)

    swell_h = _get("swell_height_m")
    swell_t = _get("swell_period_s")
    wind_s  = _get("wind_speed_kph")
    wind_d  = _get("wind_direction_deg")
    tide_h  = _get("tide_height_m", spot["preferred_tide_m"])

    energy = _wave_energy_subscore(swell_h, swell_t)
    wind   = _wind_alignment_subscore(wind_s, wind_d, spot["swell_facing_deg"])
    tide   = _tidal_subscore(tide_h, spot["preferred_tide_m"], spot["tide_sigma_m"])
    period = _period_quality_subscore(swell_t)

    return max(0.0, min(100.0, energy + wind + tide + period))

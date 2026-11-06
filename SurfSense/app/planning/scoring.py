"""
Rule-based surf condition scoring formula.

Extracted from ConditionAssessmentAgent so it can be imported standalone
(e.g. by evaluation notebooks) without pulling in the full agent stack,
and compared directly against the ML model in Phase 5 evaluation.
"""

from __future__ import annotations


def rule_based_score(
    wave_avg: float,
    wind_speed: float,
    swell_period: float,
    is_offshore: bool,
    thresholds: dict,
) -> float:
    """Compute a 0–100 surf quality score using the deterministic rule-based formula.

    Weights: wave 40 / period 30 / wind-penalty 20 / offshore 10.

    Args:
        wave_avg: Average wave height in metres.
        wind_speed: Wind speed in km/h.
        swell_period: Swell period in seconds.
        is_offshore: True if wind direction is offshore.
        thresholds: Dict with keys 'max_wave_height' and 'max_wind_speed'.

    Returns:
        Float in [0.0, 100.0].
    """
    max_wave = thresholds["max_wave_height"]
    max_wind = thresholds["max_wind_speed"]

    wave_score = min(wave_avg / max_wave, 1.0) * 40.0 if max_wave > 0 else 0.0
    period_score = min(swell_period / 14.0, 1.0) * 30.0
    wind_penalty = max(0.0, (wind_speed - max_wind) / 10.0) * 20.0
    offshore_bonus = 10.0 if is_offshore else 0.0

    return max(0.0, min(100.0, wave_score + period_score - wind_penalty + offshore_bonus))


def derive_rating(
    score: float,
    wave_avg: float,
    wind_speed: float,
    thresholds: dict,
) -> str:
    """Map a 0–100 score to a four-category rating.

    The 1.5× unsafe gate runs before the score thresholds and is
    intentionally deterministic regardless of whether the score came from
    the rule-based formula or an ML model (thesis Section 3.3.5).

    Args:
        score: Quality score in [0, 100].
        wave_avg: Average wave height in metres.
        wind_speed: Wind speed in km/h.
        thresholds: Dict with keys 'max_wave_height' and 'max_wind_speed'.

    Returns:
        One of: 'unsafe', 'ideal', 'suitable', 'challenging'.
    """
    max_wave = thresholds["max_wave_height"]
    max_wind = thresholds["max_wind_speed"]

    if wind_speed > max_wind * 1.5 or wave_avg > max_wave * 1.5:
        return "unsafe"
    if score >= 70:
        return "ideal"
    if score >= 45:
        return "suitable"
    return "challenging"

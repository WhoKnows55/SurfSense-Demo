"""
SurfSense Condition Assessment Agent

Deterministic sub-agent that evaluates environmental conditions against
user skill levels to produce safety-aware quality assessments.

Wraps the existing ConditionAssessor from app/planning/condition_assessor.py.
"""

from datetime import datetime
from typing import Any, Optional

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import ForecastPoint, WaveData, SwellData, WindData
from app.planning.condition_assessor import ConditionAssessor, ConditionRating
from app.planning.scoring import derive_rating, rule_based_score

logger = get_logger(__name__)


class ConditionAssessmentAgent(LoggerMixin):
    """Deterministic sub-agent for condition evaluation.

    Exposes tools for the orchestrator to:
    - assess_conditions: Score forecast points against skill thresholds
    - check_safety: Combine hazard data with condition assessment
    - get_skill_thresholds: Return configured thresholds

    When settings.scoring.scoring_mode == 'ml', the 0–100 score is
    computed by the XGBoost model; all safety logic, rating derivation,
    and threshold enforcement remain deterministic regardless of mode.
    """

    def __init__(self, settings):
        self._settings = settings
        self._thresholds = settings.skill_thresholds
        self._assessor = ConditionAssessor()
        self._mode: str = settings.scoring.scoring_mode  # "rule" or "ml"
        self._ml_model = None  # lazy-loaded on first use
        self._ml_model_path: str = settings.scoring.ml_model_path

        # Downgrade to rule-based if the model file hasn't been trained yet
        if self._mode == "ml":
            from pathlib import Path
            if not Path(self._ml_model_path).exists():
                logger.warning(
                    "ML model not found at '%s'; falling back to rule-based scoring. "
                    "Run `python -m ml.train` to enable ML scoring.",
                    self._ml_model_path,
                )
                self._mode = "rule"

    def _get_ml_model(self):
        """Lazy-load the ML model on first ML-mode assessment."""
        if self._ml_model is None:
            from app.ml.surf_model import SurfConditionModel
            try:
                self._ml_model = SurfConditionModel(self._ml_model_path)
            except (FileNotFoundError, ImportError) as e:
                logger.warning("ML model load failed (%s); switching to rule mode.", e)
                self._mode = "rule"
                return None
        return self._ml_model

    @staticmethod
    def _forecast_point_from_dict(fc: dict) -> Optional[ForecastPoint]:
        """Reconstruct a minimal ForecastPoint from a normalised forecast dict.

        Requires swell.direction_deg and wind.direction_deg to be present
        (added by ForecastDataAgent._forecast_to_dict in Section 3.3.3 update).
        Returns None if any required field is missing.
        """
        try:
            waves_d = fc.get("waves", {})
            swell_d = fc.get("swell", {})
            wind_d  = fc.get("wind", {})

            min_m = float(waves_d.get("min_m") or 0.0)
            max_m = float(waves_d.get("max_m") or min_m)

            swell_dir = float(swell_d.get("direction_deg") or 0.0)
            wind_dir  = float(wind_d.get("direction_deg") or 0.0)

            ts_str = fc.get("timestamp", "")
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()

            return ForecastPoint(
                timestamp=ts,
                waves=WaveData(height_min=min_m, height_max=max(min_m, max_m)),
                swell=SwellData(
                    height=float(swell_d.get("height_m") or 0.0),
                    period=float(swell_d.get("period_s") or 0.0),
                    direction_degrees=swell_dir,
                ),
                wind=WindData(
                    speed=float(wind_d.get("speed_kph") or 0.0),
                    direction_degrees=wind_dir,
                ),
            )
        except Exception:
            return None

    def assess_conditions(
        self, forecast_data: dict, skill_level: str = "intermediate"
    ) -> list[dict]:
        """Score each forecast point against skill-level thresholds.

        Args:
            forecast_data: Unified forecast dict (from ForecastDataAgent).
            skill_level: User's surfing skill level.

        Returns:
            List of per-hour assessment dicts with rating, score, reasoning.
        """
        if "error" in forecast_data:
            return [{"error": forecast_data["error"]}]

        forecasts = forecast_data.get("forecasts", [])
        if not forecasts:
            return [{"error": "No forecast data available"}]

        thresholds = self._thresholds.get_thresholds(skill_level)
        results = []

        for fc in forecasts:
            waves = fc.get("waves", {})
            wind = fc.get("wind", {})
            swell = fc.get("swell", {})

            wave_avg = waves.get("avg_m") or waves.get("max_m") or 0
            wind_speed = wind.get("speed_kph", 0)
            swell_period = swell.get("period_s") or 0
            is_offshore = wind.get("is_offshore", False)

            # --- Score (mode-dependent) ---
            contributions = None
            if self._mode == "ml":
                fp = self._forecast_point_from_dict(fc)
                model = self._get_ml_model() if fp is not None else None
                if model is not None and fp is not None:
                    score = model.predict(fp, skill_level)
                    contributions = model.get_feature_contributions(fp, skill_level)
                else:
                    score = rule_based_score(wave_avg, wind_speed, swell_period, is_offshore, thresholds)
            else:
                score = rule_based_score(wave_avg, wind_speed, swell_period, is_offshore, thresholds)

            # --- Rating (always deterministic) ---
            rating = derive_rating(score, wave_avg, wind_speed, thresholds)

            record = {
                "timestamp": fc.get("timestamp", ""),
                "rating": rating,
                "score": round(score, 1),
                "reasoning": self._build_reasoning(
                    wave_avg, wind_speed, swell_period, thresholds, rating
                ),
                "wave_height_m": wave_avg,
                "wind_speed_kph": wind_speed,
                "swell_period_s": swell_period,
            }
            if contributions is not None:
                record["feature_contributions"] = contributions
            results.append(record)

        return results

    def check_safety(
        self,
        spot_name: str,
        forecast_data: dict,
        skill_level: str,
        safety_info: dict | None = None,
    ) -> dict:
        """Combine hazard data with condition assessment into a safety verdict.

        Args:
            spot_name: Name of the surf spot.
            forecast_data: Unified forecast dict.
            skill_level: User's surfing skill level.
            safety_info: Safety info from contextual layer (optional).

        Returns:
            Safety verdict with warnings.
        """
        if safety_info is None:
            safety_info = {}

        assessments = self.assess_conditions(forecast_data, skill_level)
        if assessments and "error" in assessments[0]:
            return {
                "spot_name": spot_name,
                "safe_overall": False,
                "error": assessments[0]["error"],
            }

        unsafe_count = sum(1 for a in assessments if a.get("rating") == "unsafe")
        warnings = list(safety_info.get("warnings", []))

        recommended = safety_info.get("recommended_skill_level")
        if recommended:
            skill_order = ["beginner", "intermediate", "advanced", "expert"]
            user_idx = (
                skill_order.index(skill_level)
                if skill_level in skill_order
                else 1
            )
            rec_idx = (
                skill_order.index(recommended)
                if recommended in skill_order
                else 1
            )
            if user_idx < rec_idx:
                warnings.append(
                    f"This spot is recommended for {recommended} surfers. "
                    f"Your level ({skill_level}) may be insufficient."
                )

        return {
            "spot_name": spot_name,
            "safe_overall": unsafe_count == 0 and len(warnings) == 0,
            "unsafe_hours": unsafe_count,
            "total_hours": len(assessments),
            "warnings": warnings,
            "hazards": safety_info.get("hazards", []),
        }

    def get_skill_thresholds(self, skill_level: str) -> dict:
        """Return configured thresholds for a skill level.

        Args:
            skill_level: beginner, intermediate, or advanced.

        Returns:
            Dict with max_wave_height and max_wind_speed.
        """
        return self._thresholds.get_thresholds(skill_level)

    @staticmethod
    def _build_reasoning(
        wave_avg: float,
        wind_speed: float,
        swell_period: float,
        thresholds: dict,
        rating: str,
    ) -> str:
        """Build human-readable reasoning for a condition rating."""
        parts = []

        max_wave = thresholds["max_wave_height"]
        max_wind = thresholds["max_wind_speed"]

        if wave_avg > max_wave:
            parts.append(
                f"Waves ({wave_avg:.1f}m) exceed limit ({max_wave}m)"
            )
        elif wave_avg > 0:
            parts.append(f"Waves {wave_avg:.1f}m within range")
        else:
            parts.append("No wave data")

        if wind_speed > max_wind:
            parts.append(
                f"Wind ({wind_speed:.0f}kph) too strong (max {max_wind}kph)"
            )
        elif wind_speed > 0:
            parts.append(f"Wind {wind_speed:.0f}kph acceptable")

        if swell_period >= 12:
            parts.append(f"Good swell period ({swell_period:.0f}s)")
        elif swell_period >= 8:
            parts.append(f"Decent swell period ({swell_period:.0f}s)")
        elif swell_period > 0:
            parts.append(f"Short swell period ({swell_period:.0f}s)")

        return "; ".join(parts) + f" → {rating}"

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return OpenAI function-calling schemas for all tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "assess_conditions",
                    "description": (
                        "Evaluate surf conditions against a user's skill level. "
                        "Returns per-hour ratings (ideal/suitable/challenging/unsafe) "
                        "with scores and reasoning."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_name": {
                                "type": "string",
                                "description": "Spot name (forecast must have been fetched first)",
                            },
                            "skill_level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "User's surfing skill level",
                            },
                        },
                        "required": ["spot_name", "skill_level"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_safety",
                    "description": (
                        "Get a safety assessment for a spot, combining hazard data "
                        "with current conditions and user skill level."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_name": {"type": "string"},
                            "skill_level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                            },
                        },
                        "required": ["spot_name", "skill_level"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_skill_thresholds",
                    "description": (
                        "Return the wave height and wind speed thresholds "
                        "for a given skill level."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                            },
                        },
                        "required": ["skill_level"],
                    },
                },
            },
        ]

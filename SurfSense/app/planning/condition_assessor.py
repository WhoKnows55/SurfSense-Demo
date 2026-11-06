"""
SurfSense Condition Assessment

Evaluates surf conditions against skill level requirements to determine
suitability and safety. Provides ratings, reasoning, and warnings.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import ForecastPoint, WindDirection
from config.settings import get_settings

logger = get_logger(__name__)


class ConditionRating(str, Enum):
    """Rating for surf conditions relative to skill level."""
    
    IDEAL = "ideal"           # Perfect conditions for skill level
    SUITABLE = "suitable"     # Good conditions, within comfort zone
    CHALLENGING = "challenging"  # Possible but pushing limits
    UNSAFE = "unsafe"         # Beyond skill level, not recommended
    UNKNOWN = "unknown"       # Insufficient data to assess


class SkillLevel(str, Enum):
    """Surfer skill levels."""
    
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class WindCondition(str, Enum):
    """Wind condition categories."""
    
    OFFSHORE = "offshore"     # Wind blowing from land to sea (clean)
    CROSS_OFFSHORE = "cross_offshore"  # Mostly offshore (decent)
    CROSS_SHORE = "cross_shore"  # Side wind (bumpy)
    CROSS_ONSHORE = "cross_onshore"  # Mostly onshore (choppy)
    ONSHORE = "onshore"       # Wind blowing from sea to land (messy)


@dataclass
class ConditionAssessment:
    """
    Complete assessment of surf conditions for a skill level.
    
    Attributes:
        rating: Overall suitability rating
        skill_level: The skill level this assessment is for
        wave_assessment: Description of wave conditions
        wind_assessment: Description of wind conditions
        overall_summary: Brief summary for the user
        safety_warnings: List of safety concerns
        positive_factors: List of favorable conditions
        negative_factors: List of unfavorable conditions
        score: Numeric score (0-100) for ranking
        timestamp: When conditions occur
    """
    
    rating: ConditionRating
    skill_level: str
    wave_assessment: str
    wind_assessment: str
    overall_summary: str
    safety_warnings: list[str] = field(default_factory=list)
    positive_factors: list[str] = field(default_factory=list)
    negative_factors: list[str] = field(default_factory=list)
    score: float = 0.0
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rating": self.rating.value,
            "skill_level": self.skill_level,
            "wave_assessment": self.wave_assessment,
            "wind_assessment": self.wind_assessment,
            "overall_summary": self.overall_summary,
            "safety_warnings": self.safety_warnings,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "score": round(self.score, 1),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        emoji_map = {
            ConditionRating.IDEAL: "🟢",
            ConditionRating.SUITABLE: "🟡",
            ConditionRating.CHALLENGING: "🟠",
            ConditionRating.UNSAFE: "🔴",
            ConditionRating.UNKNOWN: "⚪",
        }
        
        lines = [
            f"{emoji_map[self.rating]} {self.rating.value.upper()} for {self.skill_level}",
            f"   {self.overall_summary}",
            f"   Waves: {self.wave_assessment}",
            f"   Wind: {self.wind_assessment}",
        ]
        
        if self.safety_warnings:
            lines.append("   ⚠️  Warnings:")
            for warning in self.safety_warnings:
                lines.append(f"      • {warning}")
        
        return "\n".join(lines)


class ConditionAssessor(LoggerMixin):
    """
    Assesses surf conditions for different skill levels.
    
    Uses configurable thresholds to determine if conditions are
    safe and suitable for surfers of various abilities.
    """
    
    # Ideal swell period thresholds (seconds)
    MIN_GOOD_SWELL_PERIOD = 8.0    # Minimum for quality waves
    IDEAL_SWELL_PERIOD = 12.0      # Ideal swell period
    
    # Wave height sweet spots by skill level (meters)
    SKILL_WAVE_SWEET_SPOTS = {
        "beginner": (0.3, 1.0),       # 0.3m - 1.0m ideal
        "intermediate": (0.8, 2.0),   # 0.8m - 2.0m ideal
        "advanced": (1.5, 4.0),       # 1.5m - 4.0m ideal
        "expert": (2.5, 10.0),        # 2.5m+ for experts
    }
    
    # Wind speed thresholds (km/h) - lighter is generally better
    IDEAL_WIND_SPEED = 10.0  # Ideal max wind for clean conditions
    
    def __init__(self):
        """Initialize the condition assessor with settings."""
        self._settings = get_settings()
        self._thresholds = self._settings.skill_thresholds
        self.log_info("ConditionAssessor initialized")
    
    def assess(
        self,
        forecast: ForecastPoint,
        skill_level: str,
        wave_direction: Optional[str] = None,
    ) -> ConditionAssessment:
        """
        Assess forecast conditions for a given skill level.
        
        Args:
            forecast: The forecast point to assess.
            skill_level: Skill level to assess for (beginner/intermediate/advanced/expert).
            wave_direction: Expected wave direction at the spot (for wind analysis).
            
        Returns:
            ConditionAssessment with rating, explanations, and warnings.
        """
        skill = skill_level.lower()
        if skill not in ["beginner", "intermediate", "advanced", "expert"]:
            skill = "intermediate"  # Default
        
        # Get thresholds
        if skill == "expert":
            thresholds = self._thresholds.get_thresholds("advanced")
            # Experts have higher tolerances
            thresholds["max_wave_height"] = 15.0
            thresholds["max_wind_speed"] = 40.0
        else:
            thresholds = self._thresholds.get_thresholds(skill)
        
        # Collect factors
        positive_factors: list[str] = []
        negative_factors: list[str] = []
        safety_warnings: list[str] = []
        score = 50.0  # Start at neutral
        
        # --- Wave Assessment ---
        wave_height = self._get_wave_height(forecast)
        wave_assessment, wave_score, wave_warnings = self._assess_waves(
            wave_height, skill, thresholds
        )
        score += wave_score
        safety_warnings.extend(wave_warnings)
        
        if wave_score > 0:
            positive_factors.append(f"Wave height ({wave_height:.1f}m) in good range")
        elif wave_score < -10:
            negative_factors.append(f"Wave height ({wave_height:.1f}m) outside comfort zone")
        
        # --- Swell Assessment ---
        swell_period = forecast.swell.period if forecast.swell else None
        if swell_period:
            swell_score = self._assess_swell_period(swell_period)
            score += swell_score
            if swell_period >= self.IDEAL_SWELL_PERIOD:
                positive_factors.append(f"Good swell period ({swell_period:.0f}s)")
            elif swell_period < self.MIN_GOOD_SWELL_PERIOD:
                negative_factors.append(f"Short swell period ({swell_period:.0f}s) - choppy waves")
        
        # --- Wind Assessment ---
        wind_speed = forecast.wind.speed if forecast.wind else 0.0
        wind_direction = forecast.wind.direction if forecast.wind else None
        wind_assessment, wind_score, wind_condition = self._assess_wind(
            wind_speed, wind_direction, wave_direction, thresholds
        )
        score += wind_score
        
        if wind_score > 10:
            positive_factors.append(wind_assessment)
        elif wind_score < -10:
            negative_factors.append(wind_assessment)
        
        # Strong wind warning
        if wind_speed > thresholds["max_wind_speed"]:
            safety_warnings.append(f"Strong wind ({wind_speed:.0f} km/h) - challenging paddle")
        
        # --- Determine Rating ---
        rating = self._calculate_rating(score, safety_warnings, skill)
        
        # --- Build Summary ---
        overall_summary = self._build_summary(
            rating, wave_height, wind_speed, wind_condition, skill
        )
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        return ConditionAssessment(
            rating=rating,
            skill_level=skill,
            wave_assessment=wave_assessment,
            wind_assessment=wind_assessment,
            overall_summary=overall_summary,
            safety_warnings=safety_warnings,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            score=score,
            timestamp=forecast.timestamp,
        )
    
    def _get_wave_height(self, forecast: ForecastPoint) -> float:
        """Extract best wave height estimate from forecast."""
        if forecast.waves:
            if forecast.waves.height_max is not None:
                return forecast.waves.height_max
            if forecast.waves.height_min is not None:
                return forecast.waves.height_min
        if forecast.swell and forecast.swell.height is not None:
            return forecast.swell.height
        return 0.0
    
    def _assess_waves(
        self,
        wave_height: float,
        skill: str,
        thresholds: dict[str, float],
    ) -> tuple[str, float, list[str]]:
        """
        Assess wave conditions.
        
        Returns:
            Tuple of (assessment text, score adjustment, warnings)
        """
        max_height = thresholds["max_wave_height"]
        sweet_spot = self.SKILL_WAVE_SWEET_SPOTS.get(skill, (0.5, 2.0))
        
        warnings: list[str] = []
        
        if wave_height == 0:
            return "No wave data available", 0.0, warnings
        
        if wave_height < 0.3:
            return f"Flat ({wave_height:.1f}m) - minimal surf", -20.0, warnings
        
        if wave_height > max_height:
            warnings.append(f"Waves ({wave_height:.1f}m) exceed {skill} limit ({max_height}m)")
            over_ratio = wave_height / max_height
            if over_ratio > 1.5:
                warnings.append("Significantly larger than skill level - HIGH RISK")
            return f"Too large ({wave_height:.1f}m) for {skill}", -30.0, warnings
        
        # Check if in sweet spot
        if sweet_spot[0] <= wave_height <= sweet_spot[1]:
            return f"Ideal size ({wave_height:.1f}m) for {skill}", 25.0, warnings
        elif wave_height < sweet_spot[0]:
            return f"Small ({wave_height:.1f}m) but surfable", 10.0, warnings
        else:
            # Above sweet spot but below max
            return f"Larger ({wave_height:.1f}m) - stay alert", 5.0, warnings
    
    def _assess_swell_period(self, period: float) -> float:
        """Score swell period quality."""
        if period >= self.IDEAL_SWELL_PERIOD:
            return 15.0  # Great groundswell
        elif period >= self.MIN_GOOD_SWELL_PERIOD:
            return 5.0   # Decent period
        elif period >= 5.0:
            return -5.0  # Short period, wind swell
        else:
            return -15.0  # Very short, messy
    
    def _assess_wind(
        self,
        wind_speed: float,
        wind_direction: Optional[WindDirection],
        wave_direction: Optional[str],
        thresholds: dict[str, float],
    ) -> tuple[str, float, WindCondition]:
        """
        Assess wind conditions.
        
        Returns:
            Tuple of (assessment text, score adjustment, wind condition)
        """
        max_wind = thresholds["max_wind_speed"]
        
        # Determine wind condition relative to waves
        # This is simplified - real implementation would compare angles
        wind_condition = WindCondition.CROSS_SHORE  # Default
        
        # For now, use a simple heuristic based on direction
        if wind_direction:
            dir_str = wind_direction.value if hasattr(wind_direction, 'value') else str(wind_direction)
            # Simplified: N/NE/NW often offshore for many spots
            if dir_str in ["N", "NE", "NW", "E"]:
                wind_condition = WindCondition.OFFSHORE
            elif dir_str in ["S", "SW", "SE", "W"]:
                wind_condition = WindCondition.ONSHORE
        
        # Score based on speed and condition
        if wind_speed <= 5.0:
            # Light wind - almost always good
            return f"Light wind ({wind_speed:.0f} km/h) - glassy", 20.0, WindCondition.OFFSHORE
        
        if wind_speed <= self.IDEAL_WIND_SPEED:
            if wind_condition == WindCondition.OFFSHORE:
                return f"Light offshore ({wind_speed:.0f} km/h) - clean", 15.0, wind_condition
            else:
                return f"Light wind ({wind_speed:.0f} km/h)", 10.0, wind_condition
        
        if wind_speed <= max_wind:
            if wind_condition == WindCondition.OFFSHORE:
                return f"Offshore ({wind_speed:.0f} km/h) - holding shape", 5.0, wind_condition
            elif wind_condition == WindCondition.ONSHORE:
                return f"Onshore ({wind_speed:.0f} km/h) - bumpy", -10.0, wind_condition
            else:
                return f"Cross-shore ({wind_speed:.0f} km/h)", 0.0, wind_condition
        
        # Wind exceeds threshold
        if wind_condition == WindCondition.OFFSHORE:
            return f"Strong offshore ({wind_speed:.0f} km/h) - difficult paddle out", -15.0, wind_condition
        else:
            return f"Strong wind ({wind_speed:.0f} km/h) - messy conditions", -25.0, wind_condition
    
    def _calculate_rating(
        self,
        score: float,
        warnings: list[str],
        skill: str,
    ) -> ConditionRating:
        """Determine overall rating from score and warnings."""
        # Safety warnings can override score
        has_severe_warning = any(
            "HIGH RISK" in w or "exceed" in w.lower() 
            for w in warnings
        )
        
        if has_severe_warning:
            return ConditionRating.UNSAFE
        
        if score >= 70:
            return ConditionRating.IDEAL
        elif score >= 50:
            return ConditionRating.SUITABLE
        elif score >= 30:
            return ConditionRating.CHALLENGING
        else:
            return ConditionRating.UNSAFE
    
    def _build_summary(
        self,
        rating: ConditionRating,
        wave_height: float,
        wind_speed: float,
        wind_condition: WindCondition,
        skill: str,
    ) -> str:
        """Build a concise summary for the user."""
        summaries = {
            ConditionRating.IDEAL: f"Excellent conditions for {skill} surfers!",
            ConditionRating.SUITABLE: f"Good surfable conditions for {skill} level.",
            ConditionRating.CHALLENGING: "Challenging conditions - proceed with caution.",
            ConditionRating.UNSAFE: f"Not recommended for {skill} surfers - conditions too demanding.",
            ConditionRating.UNKNOWN: "Insufficient data to assess conditions.",
        }
        return summaries.get(rating, "Conditions unknown.")
    
    def assess_forecast_range(
        self,
        forecasts: list[ForecastPoint],
        skill_level: str,
        wave_direction: Optional[str] = None,
    ) -> list[ConditionAssessment]:
        """
        Assess multiple forecast points.
        
        Args:
            forecasts: List of forecast points to assess.
            skill_level: Skill level to assess for.
            wave_direction: Expected wave direction at spot.
            
        Returns:
            List of assessments, one per forecast point.
        """
        return [
            self.assess(fc, skill_level, wave_direction)
            for fc in forecasts
        ]
    
    def find_best_conditions(
        self,
        forecasts: list[ForecastPoint],
        skill_level: str,
        min_rating: ConditionRating = ConditionRating.SUITABLE,
    ) -> list[ConditionAssessment]:
        """
        Find the best conditions from a forecast range.
        
        Args:
            forecasts: List of forecast points.
            skill_level: Skill level to assess for.
            min_rating: Minimum acceptable rating.
            
        Returns:
            List of assessments meeting criteria, sorted by score.
        """
        assessments = self.assess_forecast_range(forecasts, skill_level)
        
        # Filter by minimum rating
        rating_order = [
            ConditionRating.IDEAL,
            ConditionRating.SUITABLE,
            ConditionRating.CHALLENGING,
            ConditionRating.UNSAFE,
        ]
        min_idx = rating_order.index(min_rating)
        acceptable_ratings = set(rating_order[:min_idx + 1])
        
        filtered = [a for a in assessments if a.rating in acceptable_ratings]
        
        # Sort by score descending
        return sorted(filtered, key=lambda a: a.score, reverse=True)
    
    def get_daily_summary(
        self,
        forecasts: list[ForecastPoint],
        skill_level: str,
    ) -> dict[str, Any]:
        """
        Generate a daily summary of conditions.
        
        Groups forecasts by date and provides best/worst times.
        
        Args:
            forecasts: List of forecast points (ideally hourly for a day).
            skill_level: Skill level to assess for.
            
        Returns:
            Dictionary with daily summary information.
        """
        if not forecasts:
            return {"error": "No forecast data"}
        
        assessments = self.assess_forecast_range(forecasts, skill_level)
        
        # Find best and worst
        best = max(assessments, key=lambda a: a.score)
        worst = min(assessments, key=lambda a: a.score)
        
        # Count ratings
        rating_counts = {}
        for a in assessments:
            rating_counts[a.rating.value] = rating_counts.get(a.rating.value, 0) + 1
        
        # Overall recommendation
        avg_score = sum(a.score for a in assessments) / len(assessments)
        if avg_score >= 60:
            recommendation = "Great day for surfing!"
        elif avg_score >= 45:
            recommendation = "Decent conditions - pick your windows"
        elif avg_score >= 30:
            recommendation = "Marginal conditions - for experienced surfers"
        else:
            recommendation = "Poor conditions - consider alternatives"
        
        return {
            "skill_level": skill_level,
            "total_hours": len(assessments),
            "average_score": round(avg_score, 1),
            "rating_breakdown": rating_counts,
            "best_time": {
                "time": best.timestamp.strftime("%H:%M") if best.timestamp else "Unknown",
                "rating": best.rating.value,
                "score": round(best.score, 1),
                "summary": best.overall_summary,
            },
            "worst_time": {
                "time": worst.timestamp.strftime("%H:%M") if worst.timestamp else "Unknown",
                "rating": worst.rating.value,
                "score": round(worst.score, 1),
            },
            "recommendation": recommendation,
            "all_warnings": list(set(
                warning 
                for a in assessments 
                for warning in a.safety_warnings
            )),
        }

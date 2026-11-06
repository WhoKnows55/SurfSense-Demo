"""
SurfSense Surf Window Finder

Identifies optimal surfing windows within a forecast by analyzing
conditions over time and grouping consecutive favorable hours.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import ForecastPoint, WindDirection
from app.planning.condition_assessor import (
    ConditionAssessor,
    ConditionAssessment,
    ConditionRating,
)

logger = get_logger(__name__)


class WindowQuality(str, Enum):
    """Quality rating for a surf window."""
    
    EPIC = "epic"           # 90+ score - exceptional conditions
    EXCELLENT = "excellent" # 75-90 score - great surfing
    GOOD = "good"           # 60-75 score - solid session
    FAIR = "fair"           # 45-60 score - surfable but not ideal
    POOR = "poor"           # Below 45 - marginal at best


@dataclass
class SurfWindow:
    """
    A contiguous window of favorable surf conditions.
    
    Represents a period of time when conditions are suitable
    for surfing at a particular skill level.
    """
    
    start_time: datetime
    end_time: datetime
    skill_level: str
    quality: WindowQuality
    average_score: float
    peak_score: float
    peak_time: datetime
    duration_hours: float
    assessments: list[ConditionAssessment] = field(default_factory=list)
    
    # Summary stats
    wave_height_range: tuple[float, float] = (0.0, 0.0)
    wind_speed_range: tuple[float, float] = (0.0, 0.0)
    dominant_wind: Optional[str] = None
    
    # Factors
    positive_factors: list[str] = field(default_factory=list)
    negative_factors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "skill_level": self.skill_level,
            "quality": self.quality.value,
            "average_score": round(self.average_score, 1),
            "peak_score": round(self.peak_score, 1),
            "peak_time": self.peak_time.isoformat(),
            "duration_hours": self.duration_hours,
            "wave_height_range": {
                "min_m": round(self.wave_height_range[0], 1),
                "max_m": round(self.wave_height_range[1], 1),
            },
            "wind_speed_range": {
                "min_kph": round(self.wind_speed_range[0], 1),
                "max_kph": round(self.wind_speed_range[1], 1),
            },
            "dominant_wind": self.dominant_wind,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "warnings": self.warnings,
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        quality_emoji = {
            WindowQuality.EPIC: "🌟",
            WindowQuality.EXCELLENT: "🔥",
            WindowQuality.GOOD: "👍",
            WindowQuality.FAIR: "👌",
            WindowQuality.POOR: "😐",
        }
        
        start_str = self.start_time.strftime("%a %H:%M")
        end_str = self.end_time.strftime("%H:%M")
        peak_str = self.peak_time.strftime("%H:%M")
        
        lines = [
            f"{quality_emoji.get(self.quality, '•')} {self.quality.value.upper()} Window: {start_str} - {end_str} ({self.duration_hours:.1f}h)",
            f"   Score: {self.average_score:.0f}/100 (peak {self.peak_score:.0f} at {peak_str})",
            f"   Waves: {self.wave_height_range[0]:.1f}-{self.wave_height_range[1]:.1f}m",
            f"   Wind: {self.wind_speed_range[0]:.0f}-{self.wind_speed_range[1]:.0f} km/h",
        ]
        
        if self.positive_factors:
            lines.append(f"   ✓ {', '.join(self.positive_factors[:3])}")
        
        if self.warnings:
            lines.append(f"   ⚠️ {self.warnings[0]}")
        
        return "\n".join(lines)


@dataclass
class WindowFinderResult:
    """Result from the window finder analysis."""
    
    spot_name: str
    skill_level: str
    forecast_start: datetime
    forecast_end: datetime
    windows: list[SurfWindow] = field(default_factory=list)
    total_surfable_hours: float = 0.0
    best_window: Optional[SurfWindow] = None
    recommendation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spot_name": self.spot_name,
            "skill_level": self.skill_level,
            "forecast_period": {
                "start": self.forecast_start.isoformat(),
                "end": self.forecast_end.isoformat(),
            },
            "total_surfable_hours": round(self.total_surfable_hours, 1),
            "window_count": len(self.windows),
            "best_window": self.best_window.to_dict() if self.best_window else None,
            "all_windows": [w.to_dict() for w in self.windows],
            "recommendation": self.recommendation,
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        lines = [
            f"🏄 Surf Windows for {self.spot_name}",
            f"   Skill Level: {self.skill_level}",
            f"   Period: {self.forecast_start.strftime('%a %d %b')} - {self.forecast_end.strftime('%a %d %b')}",
            f"   Total Surfable: {self.total_surfable_hours:.1f} hours in {len(self.windows)} window(s)",
            "",
        ]
        
        if self.best_window:
            lines.append("🏆 BEST WINDOW:")
            lines.append(self.best_window.format_for_display())
            lines.append("")
        
        if len(self.windows) > 1:
            lines.append("📅 ALL WINDOWS:")
            for i, window in enumerate(self.windows[:5], 1):
                if window != self.best_window:
                    lines.append(f"\n{i}. {window.format_for_display()}")
        
        lines.append(f"\n💡 {self.recommendation}")
        
        return "\n".join(lines)


class SurfWindowFinder(LoggerMixin):
    """
    Finds optimal surfing windows within a forecast.
    
    Analyzes conditions over time to identify contiguous periods
    when surfing is favorable for a given skill level.
    """
    
    # Minimum window duration (hours)
    MIN_WINDOW_HOURS = 1.5
    
    # Minimum score to be considered surfable
    MIN_SURFABLE_SCORE = 40.0
    
    # Score thresholds for quality ratings
    QUALITY_THRESHOLDS = {
        WindowQuality.EPIC: 90.0,
        WindowQuality.EXCELLENT: 75.0,
        WindowQuality.GOOD: 60.0,
        WindowQuality.FAIR: 45.0,
    }
    
    # Weights for different factors in window scoring
    SCORING_WEIGHTS = {
        "wave_quality": 0.35,      # Wave height in sweet spot
        "wind_quality": 0.30,      # Offshore/light wind
        "swell_period": 0.15,      # Longer period = better
        "consistency": 0.10,       # Stable conditions
        "time_of_day": 0.10,       # Dawn/dusk patrol bonus
    }
    
    def __init__(self, condition_assessor: Optional[ConditionAssessor] = None):
        """
        Initialize the window finder.
        
        Args:
            condition_assessor: Optional assessor instance. Creates one if not provided.
        """
        self._assessor = condition_assessor or ConditionAssessor()
        self.log_info("SurfWindowFinder initialized")
    
    def find_windows(
        self,
        forecasts: list[ForecastPoint],
        skill_level: str,
        spot_name: str = "Unknown",
        min_duration_hours: float = MIN_WINDOW_HOURS,
        min_score: float = MIN_SURFABLE_SCORE,
    ) -> WindowFinderResult:
        """
        Find all surfable windows in a forecast.
        
        Args:
            forecasts: List of forecast points (ideally hourly).
            skill_level: Skill level to assess for.
            spot_name: Name of the spot for the result.
            min_duration_hours: Minimum window duration.
            min_score: Minimum score to be considered surfable.
            
        Returns:
            WindowFinderResult with all identified windows.
        """
        if not forecasts:
            return WindowFinderResult(
                spot_name=spot_name,
                skill_level=skill_level,
                forecast_start=datetime.utcnow(),
                forecast_end=datetime.utcnow(),
                recommendation="No forecast data available.",
            )
        
        self.log_info(f"Finding windows for {spot_name}, skill={skill_level}")
        
        # Get assessments for all forecast points
        assessments = self._assessor.assess_forecast_range(forecasts, skill_level)
        
        # Pair assessments with forecasts
        forecast_assessments = list(zip(forecasts, assessments))
        
        # Group into windows
        windows = self._group_into_windows(
            forecast_assessments,
            skill_level,
            min_duration_hours,
            min_score,
        )
        
        # Calculate total surfable hours
        total_hours = sum(w.duration_hours for w in windows)
        
        # Find best window
        best_window = max(windows, key=lambda w: w.average_score) if windows else None
        
        # Generate recommendation
        recommendation = self._generate_recommendation(windows, skill_level, total_hours)
        
        return WindowFinderResult(
            spot_name=spot_name,
            skill_level=skill_level,
            forecast_start=forecasts[0].timestamp,
            forecast_end=forecasts[-1].timestamp,
            windows=windows,
            total_surfable_hours=total_hours,
            best_window=best_window,
            recommendation=recommendation,
        )
    
    def _group_into_windows(
        self,
        forecast_assessments: list[tuple[ForecastPoint, ConditionAssessment]],
        skill_level: str,
        min_duration: float,
        min_score: float,
    ) -> list[SurfWindow]:
        """
        Group consecutive surfable hours into windows.
        
        Args:
            forecast_assessments: Paired forecasts and assessments.
            skill_level: Skill level being assessed.
            min_duration: Minimum window duration in hours.
            min_score: Minimum score to be surfable.
            
        Returns:
            List of SurfWindow objects.
        """
        windows = []
        current_window_data: list[tuple[ForecastPoint, ConditionAssessment]] = []
        
        for fc, assessment in forecast_assessments:
            is_surfable = (
                assessment.score >= min_score and
                assessment.rating != ConditionRating.UNSAFE
            )
            
            if is_surfable:
                current_window_data.append((fc, assessment))
            else:
                # End of current window
                if current_window_data:
                    window = self._create_window(current_window_data, skill_level)
                    if window and window.duration_hours >= min_duration:
                        windows.append(window)
                    current_window_data = []
        
        # Don't forget the last window
        if current_window_data:
            window = self._create_window(current_window_data, skill_level)
            if window and window.duration_hours >= min_duration:
                windows.append(window)
        
        # Sort by average score descending
        windows.sort(key=lambda w: w.average_score, reverse=True)
        
        return windows
    
    def _create_window(
        self,
        window_data: list[tuple[ForecastPoint, ConditionAssessment]],
        skill_level: str,
    ) -> Optional[SurfWindow]:
        """
        Create a SurfWindow from a list of consecutive forecast points.
        
        Args:
            window_data: List of (forecast, assessment) tuples.
            skill_level: Skill level for the window.
            
        Returns:
            SurfWindow object or None if invalid.
        """
        if not window_data:
            return None
        
        forecasts = [fc for fc, _ in window_data]
        assessments = [a for _, a in window_data]
        
        # Calculate timing
        start_time = forecasts[0].timestamp
        end_time = forecasts[-1].timestamp
        
        # Estimate duration (assume hourly forecasts, add 1 for end hour)
        if len(forecasts) > 1:
            time_diff = (forecasts[1].timestamp - forecasts[0].timestamp).total_seconds() / 3600
        else:
            time_diff = 1.0
        duration_hours = len(forecasts) * time_diff
        
        # Calculate scores
        scores = [a.score for a in assessments]
        average_score = sum(scores) / len(scores)
        peak_score = max(scores)
        peak_idx = scores.index(peak_score)
        peak_time = forecasts[peak_idx].timestamp
        
        # Determine quality
        quality = self._score_to_quality(average_score)
        
        # Extract wave heights
        wave_heights = []
        for fc in forecasts:
            if fc.waves and fc.waves.height_max:
                wave_heights.append(fc.waves.height_max)
            elif fc.swell and fc.swell.height:
                wave_heights.append(fc.swell.height)
        
        wave_range = (
            (min(wave_heights), max(wave_heights)) 
            if wave_heights else (0.0, 0.0)
        )
        
        # Extract wind speeds
        wind_speeds = [fc.wind.speed for fc in forecasts if fc.wind]
        wind_range = (
            (min(wind_speeds), max(wind_speeds))
            if wind_speeds else (0.0, 0.0)
        )
        
        # Determine dominant wind direction
        wind_dirs = [
            fc.wind.direction.value 
            for fc in forecasts 
            if fc.wind and fc.wind.direction
        ]
        dominant_wind = max(set(wind_dirs), key=wind_dirs.count) if wind_dirs else None
        
        # Collect factors and warnings
        positive_factors = self._collect_positive_factors(assessments, forecasts)
        negative_factors = self._collect_negative_factors(assessments)
        warnings = list(set(w for a in assessments for w in a.safety_warnings))
        
        return SurfWindow(
            start_time=start_time,
            end_time=end_time,
            skill_level=skill_level,
            quality=quality,
            average_score=average_score,
            peak_score=peak_score,
            peak_time=peak_time,
            duration_hours=duration_hours,
            assessments=assessments,
            wave_height_range=wave_range,
            wind_speed_range=wind_range,
            dominant_wind=dominant_wind,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            warnings=warnings,
        )
    
    def _score_to_quality(self, score: float) -> WindowQuality:
        """Convert score to quality rating."""
        for quality, threshold in self.QUALITY_THRESHOLDS.items():
            if score >= threshold:
                return quality
        return WindowQuality.POOR
    
    def _collect_positive_factors(
        self,
        assessments: list[ConditionAssessment],
        forecasts: list[ForecastPoint],
    ) -> list[str]:
        """Extract common positive factors from assessments."""
        factors = []
        
        # Check for offshore wind
        offshore_count = sum(
            1 for fc in forecasts 
            if fc.wind and fc.is_offshore_wind
        )
        if offshore_count > len(forecasts) * 0.5:
            factors.append("Offshore winds")
        
        # Check for light wind
        light_wind_count = sum(
            1 for fc in forecasts
            if fc.wind and fc.is_light_wind
        )
        if light_wind_count > len(forecasts) * 0.5:
            factors.append("Light winds")
        
        # Check for good swell period
        good_period_count = sum(
            1 for fc in forecasts
            if fc.swell and fc.swell.period and fc.swell.period >= 10
        )
        if good_period_count > len(forecasts) * 0.5:
            factors.append("Good swell period")
        
        # Dawn patrol bonus (5-8 AM)
        dawn_count = sum(
            1 for fc in forecasts
            if 5 <= fc.timestamp.hour <= 8
        )
        if dawn_count > 0:
            factors.append("Dawn patrol window")
        
        # Aggregate from assessments
        all_positives = [f for a in assessments for f in a.positive_factors]
        common_positives = set(f for f in all_positives if all_positives.count(f) > len(assessments) * 0.3)
        factors.extend(list(common_positives)[:2])
        
        return list(set(factors))[:5]
    
    def _collect_negative_factors(
        self,
        assessments: list[ConditionAssessment],
    ) -> list[str]:
        """Extract common negative factors from assessments."""
        all_negatives = [f for a in assessments for f in a.negative_factors]
        common_negatives = set(f for f in all_negatives if all_negatives.count(f) > len(assessments) * 0.3)
        return list(common_negatives)[:3]
    
    def _generate_recommendation(
        self,
        windows: list[SurfWindow],
        skill_level: str,
        total_hours: float,
    ) -> str:
        """Generate a recommendation based on the windows found."""
        if not windows:
            return f"No suitable surf windows found for {skill_level} level. Consider checking back later or trying a different spot."
        
        best = windows[0]
        
        if best.quality == WindowQuality.EPIC:
            return f"Epic conditions! Don't miss the {best.duration_hours:.1f}-hour window starting {best.start_time.strftime('%A at %H:%M')}."
        elif best.quality == WindowQuality.EXCELLENT:
            return f"Excellent session ahead! Best window: {best.start_time.strftime('%A %H:%M')} for {best.duration_hours:.1f} hours."
        elif best.quality == WindowQuality.GOOD:
            return f"Good conditions for a solid session. Hit the water around {best.start_time.strftime('%H:%M')} for best results."
        elif best.quality == WindowQuality.FAIR:
            if len(windows) > 1:
                return f"Conditions are fair - {total_hours:.1f} hours total. Best bet is {best.start_time.strftime('%H:%M')}."
            return f"Fair conditions. Manage expectations but {best.duration_hours:.1f} hours of surfable waves."
        else:
            return f"Marginal conditions. Only {total_hours:.1f} hours of surfable time. Consider waiting for improvement."
    
    def find_best_window(
        self,
        forecasts: list[ForecastPoint],
        skill_level: str,
        spot_name: str = "Unknown",
    ) -> Optional[SurfWindow]:
        """
        Find the single best surf window.
        
        Convenience method that returns just the best window.
        
        Args:
            forecasts: List of forecast points.
            skill_level: Skill level to assess for.
            spot_name: Name of the spot.
            
        Returns:
            Best SurfWindow or None if no surfable windows.
        """
        result = self.find_windows(forecasts, skill_level, spot_name)
        return result.best_window
    
    def find_windows_by_day(
        self,
        forecasts: list[ForecastPoint],
        skill_level: str,
        spot_name: str = "Unknown",
    ) -> dict[str, WindowFinderResult]:
        """
        Find windows grouped by day.
        
        Args:
            forecasts: List of forecast points (multi-day).
            skill_level: Skill level to assess for.
            spot_name: Name of the spot.
            
        Returns:
            Dictionary mapping date strings to WindowFinderResult.
        """
        # Group forecasts by date
        by_date: dict[str, list[ForecastPoint]] = {}
        for fc in forecasts:
            date_key = fc.timestamp.strftime("%Y-%m-%d")
            if date_key not in by_date:
                by_date[date_key] = []
            by_date[date_key].append(fc)
        
        # Find windows for each day
        results = {}
        for date_key, day_forecasts in sorted(by_date.items()):
            results[date_key] = self.find_windows(
                day_forecasts,
                skill_level,
                spot_name,
            )
        
        return results

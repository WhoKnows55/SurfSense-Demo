"""
SurfSense Forecast Preview

Generates condensed, visually appealing forecast previews
for trip planning confirmation flow.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import ForecastPoint, ForecastResponse
from app.planning.condition_assessor import ConditionAssessor, ConditionRating

logger = get_logger(__name__)


# Emoji for condition ratings
RATING_EMOJI = {
    ConditionRating.IDEAL: "🌟",
    ConditionRating.SUITABLE: "🏄",
    ConditionRating.CHALLENGING: "👌",
    ConditionRating.UNSAFE: "⚠️",
    ConditionRating.UNKNOWN: "❓",
}

# Text descriptions for ratings
RATING_TEXT = {
    ConditionRating.IDEAL: "IDEAL!",
    ConditionRating.SUITABLE: "Suitable",
    ConditionRating.CHALLENGING: "Challenging",
    ConditionRating.UNSAFE: "Unsafe",
    ConditionRating.UNKNOWN: "Unknown",
}

# Wind direction emojis
WIND_DIRECTION_EMOJI = {
    "N": "⬇️",
    "NE": "↙️",
    "E": "⬅️",
    "SE": "↖️",
    "S": "⬆️",
    "SW": "↗️",
    "W": "➡️",
    "NW": "↘️",
}


@dataclass
class DayForecastSummary:
    """Summary of conditions for a single day."""
    date: date
    rating: ConditionRating
    wave_height_min: float  # meters
    wave_height_max: float  # meters
    swell_period: float  # seconds
    swell_direction: str
    wind_speed: float  # km/h
    wind_direction: str
    water_temp: Optional[float] = None  # Celsius
    best_time: Optional[str] = None  # e.g., "6am-10am"
    notes: Optional[str] = None


class ForecastPreviewGenerator(LoggerMixin):
    """
    Generates visual forecast previews for trip planning.
    
    Creates condensed, easy-to-read summaries that help users
    decide whether to proceed with their planned trip.
    """
    
    def __init__(self, condition_assessor: Optional[ConditionAssessor] = None):
        """
        Initialize the preview generator.
        
        Args:
            condition_assessor: Assessor for rating conditions.
        """
        self._assessor = condition_assessor or ConditionAssessor()
    
    def generate_preview(
        self,
        forecast: ForecastResponse,
        target_dates: list[date],
        skill_level: str = "intermediate",
        spot_name: Optional[str] = None,
    ) -> str:
        """
        Generate a formatted forecast preview.
        
        Args:
            forecast: Full forecast response.
            target_dates: Dates the user is interested in.
            skill_level: User's skill level.
            spot_name: Name of the surf spot.
            
        Returns:
            Formatted string preview.
        """
        # Summarize each day
        daily_summaries = self._summarize_by_day(
            forecast.points,
            target_dates,
            skill_level
        )
        
        if not daily_summaries:
            return self._no_data_preview(spot_name)
        
        # Build the preview
        lines = []
        
        # Header
        header = "📊 FORECAST PREVIEW"
        if spot_name:
            header = f"📊 FORECAST PREVIEW - {spot_name.upper()}"
        lines.append(header)
        lines.append("═" * len(header))
        
        # Overall rating
        overall = self._calculate_overall_rating(daily_summaries)
        overall_emoji = RATING_EMOJI.get(overall, "❓")
        overall_text = RATING_TEXT.get(overall, "Unknown")
        lines.append(f"\n{overall_emoji} Overall: {overall_text} for {skill_level} surfers\n")
        
        # Daily breakdown
        lines.append("📅 Daily Conditions:")
        lines.append("-" * 40)
        
        for summary in daily_summaries:
            day_line = self._format_day_summary(summary)
            lines.append(day_line)
        
        lines.append("-" * 40)
        
        # Best window recommendation
        best_day = self._find_best_day(daily_summaries)
        if best_day:
            best_emoji = RATING_EMOJI.get(best_day.rating, "🏄")
            best_date = best_day.date.strftime("%A, %b %d")
            lines.append(f"\n✨ Best Day: {best_date} {best_emoji}")
            if best_day.best_time:
                lines.append(f"   Recommended Time: {best_day.best_time}")
        
        # Add notes/tips
        tips = self._generate_tips(daily_summaries, skill_level)
        if tips:
            lines.append(f"\n💡 Tips: {tips}")
        
        return "\n".join(lines)
    
    def _summarize_by_day(
        self,
        points: list[ForecastPoint],
        target_dates: list[date],
        skill_level: str
    ) -> list[DayForecastSummary]:
        """Summarize forecast points by day."""
        summaries = []
        
        # Group points by date
        points_by_date: dict[date, list[ForecastPoint]] = {}
        for point in points:
            d = point.timestamp.date()
            if target_dates and d not in target_dates:
                continue
            if d not in points_by_date:
                points_by_date[d] = []
            points_by_date[d].append(point)
        
        # Create summary for each day
        for d in sorted(points_by_date.keys()):
            day_points = points_by_date[d]
            summary = self._summarize_day(d, day_points, skill_level)
            summaries.append(summary)
        
        return summaries
    
    def _summarize_day(
        self,
        d: date,
        points: list[ForecastPoint],
        skill_level: str
    ) -> DayForecastSummary:
        """Create a summary for a single day's forecast."""
        # Find min/max wave heights
        wave_heights = [p.wave_height for p in points if p.wave_height is not None]
        wave_min = min(wave_heights) if wave_heights else 0.0
        wave_max = max(wave_heights) if wave_heights else 0.0
        
        # Average swell period (weight toward higher values)
        swell_periods = [p.swell_period for p in points if p.swell_period is not None]
        swell_period = max(swell_periods) if swell_periods else 0.0
        
        # Most common swell direction
        swell_dirs = [p.swell_direction for p in points if p.swell_direction is not None]
        swell_dir = max(set(swell_dirs), key=swell_dirs.count) if swell_dirs else "N/A"
        
        # Average wind speed
        wind_speeds = [p.wind_speed for p in points if p.wind_speed is not None]
        wind_speed = sum(wind_speeds) / len(wind_speeds) if wind_speeds else 0.0
        
        # Most common wind direction
        wind_dirs = [p.wind_direction for p in points if p.wind_direction is not None]
        wind_dir = max(set(wind_dirs), key=wind_dirs.count) if wind_dirs else "N/A"
        
        # Water temperature (if available)
        water_temps = [p.water_temperature for p in points if p.water_temperature is not None]
        water_temp = sum(water_temps) / len(water_temps) if water_temps else None
        
        # Rate conditions for each point and find best time window
        ratings = []
        best_rating = ConditionRating.FLAT
        best_time_start = None
        best_time_end = None
        
        for point in points:
            rating = self._assessor.rate_conditions(point, skill_level)
            ratings.append((point.timestamp, rating))
            
            if rating.value > best_rating.value:
                best_rating = rating
                best_time_start = point.timestamp
                best_time_end = point.timestamp
            elif rating.value == best_rating.value and best_time_end:
                # Extend window if same rating
                if (point.timestamp - best_time_end).seconds <= 7200:  # 2 hours
                    best_time_end = point.timestamp
        
        # Format best time window
        best_time = None
        if best_time_start:
            start_str = best_time_start.strftime("%I%p").lower().lstrip("0")
            if best_time_end and best_time_end != best_time_start:
                end_str = best_time_end.strftime("%I%p").lower().lstrip("0")
                best_time = f"{start_str}-{end_str}"
            else:
                best_time = f"around {start_str}"
        
        # Overall day rating (weighted toward best conditions)
        day_rating = self._calculate_day_rating(ratings)
        
        return DayForecastSummary(
            date=d,
            rating=day_rating,
            wave_height_min=wave_min,
            wave_height_max=wave_max,
            swell_period=swell_period,
            swell_direction=swell_dir,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            water_temp=water_temp,
            best_time=best_time,
        )
    
    def _calculate_day_rating(
        self,
        ratings: list[tuple[datetime, ConditionRating]]
    ) -> ConditionRating:
        """Calculate overall day rating from hourly ratings."""
        if not ratings:
            return ConditionRating.FAIR
        
        # Weight morning/midday ratings higher (typically surf windows)
        weighted_values = []
        for timestamp, rating in ratings:
            hour = timestamp.hour
            # Higher weight for 5am-11am
            if 5 <= hour <= 11:
                weight = 1.5
            # Medium weight for 3pm-6pm
            elif 15 <= hour <= 18:
                weight = 1.2
            else:
                weight = 1.0
            weighted_values.append(rating.value * weight)
        
        avg_value = sum(weighted_values) / len(weighted_values)
        
        # Map back to rating
        if avg_value >= 4.5:
            return ConditionRating.IDEAL
        elif avg_value >= 3.5:
            return ConditionRating.SUITABLE
        elif avg_value >= 2.5:
            return ConditionRating.CHALLENGING
        elif avg_value >= 1.5:
            return ConditionRating.UNSAFE
        else:
            return ConditionRating.UNKNOWN
    
    def _calculate_overall_rating(
        self,
        summaries: list[DayForecastSummary]
    ) -> ConditionRating:
        """Calculate overall trip rating from daily summaries."""
        if not summaries:
            return ConditionRating.FAIR
        
        # Use the best day's rating as overall
        best = max(summaries, key=lambda s: s.rating.value)
        return best.rating
    
    def _find_best_day(
        self,
        summaries: list[DayForecastSummary]
    ) -> Optional[DayForecastSummary]:
        """Find the best day from the summaries."""
        if not summaries:
            return None
        return max(summaries, key=lambda s: s.rating.value)
    
    def _format_day_summary(self, summary: DayForecastSummary) -> str:
        """Format a single day summary as a string."""
        emoji = RATING_EMOJI.get(summary.rating, "❓")
        rating_text = RATING_TEXT.get(summary.rating, "?")
        
        # Format date
        day_name = summary.date.strftime("%a %b %d")
        
        # Format waves
        if summary.wave_height_min == summary.wave_height_max:
            waves = f"{summary.wave_height_min:.1f}m"
        else:
            waves = f"{summary.wave_height_min:.1f}-{summary.wave_height_max:.1f}m"
        
        # Format wind
        wind_emoji = WIND_DIRECTION_EMOJI.get(summary.wind_direction, "🌬️")
        wind = f"{summary.wind_speed:.0f}km/h {wind_emoji}"
        
        # Format swell
        swell = f"{summary.swell_period:.0f}s"
        
        # Build line
        line = f"  {day_name} | {emoji} {rating_text:6} | {waves:10} | Swell: {swell} | Wind: {wind}"
        
        if summary.best_time:
            line += f" | Best: {summary.best_time}"
        
        return line
    
    def _generate_tips(
        self,
        summaries: list[DayForecastSummary],
        skill_level: str
    ) -> Optional[str]:
        """Generate helpful tips based on conditions."""
        tips = []
        
        # Check for dangerous days
        dangerous_days = [s for s in summaries if s.rating == ConditionRating.DANGEROUS]
        if dangerous_days:
            dates = ", ".join(s.date.strftime("%a") for s in dangerous_days)
            tips.append(f"Conditions may be too challenging on {dates}")
        
        # Check wind patterns
        avg_wind = sum(s.wind_speed for s in summaries) / len(summaries) if summaries else 0
        if avg_wind > 30:
            tips.append("Expect choppy conditions due to wind")
        
        # Skill-specific tips
        if skill_level == "beginner":
            high_waves = [s for s in summaries if s.wave_height_max > 1.5]
            if high_waves:
                tips.append("Waves may be large - stay in whitewash")
        
        return "; ".join(tips) if tips else None
    
    def _no_data_preview(self, spot_name: Optional[str] = None) -> str:
        """Generate a preview when no data is available."""
        spot_text = f" for {spot_name}" if spot_name else ""
        return f"""
📊 FORECAST PREVIEW
═══════════════════

⚠️ Unable to retrieve forecast data{spot_text}.

This could be due to:
- The spot location not being recognized
- API service temporarily unavailable
- Invalid date range

Would you like to:
- Try a different surf spot?
- Adjust your travel dates?
"""


def format_compact_preview(
    spot_name: str,
    dates: list[date],
    wave_range: tuple[float, float],
    overall_rating: str,
    best_day: Optional[date] = None,
) -> str:
    """
    Create a compact one-line preview for quick display.
    
    Args:
        spot_name: Name of the surf spot.
        dates: Trip dates.
        wave_range: (min, max) wave height in meters.
        overall_rating: Rating string (e.g., "Good").
        best_day: Best day of the trip.
        
    Returns:
        Compact preview string.
    """
    date_range = f"{dates[0].strftime('%b %d')}"
    if len(dates) > 1:
        date_range += f"-{dates[-1].strftime('%d')}"
    
    waves = f"{wave_range[0]:.1f}-{wave_range[1]:.1f}m"
    best = f" | Best: {best_day.strftime('%a')}" if best_day else ""
    
    return f"🏄 {spot_name} ({date_range}): {waves} waves, {overall_rating}{best}"

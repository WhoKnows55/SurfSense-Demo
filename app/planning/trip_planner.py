"""
SurfSense Trip Planner

Plans multi-day surf trip itineraries by:
- Finding optimal windows at multiple spots
- Optimizing travel time between spots
- Considering weather, crowds, and logistics
- Creating day-by-day schedules
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from math import radians, cos, sin, sqrt, atan2
from typing import Any, Optional

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import ForecastPoint
from app.planning.condition_assessor import ConditionAssessor, ConditionAssessment
from app.planning.window_finder import (
    SurfWindowFinder,
    SurfWindow,
    WindowQuality,
    WindowFinderResult,
)

logger = get_logger(__name__)


class SessionPriority(str, Enum):
    """Priority level for a surf session."""
    
    MUST_SURF = "must_surf"    # Can't miss this window
    PREFERRED = "preferred"     # Good option
    OPTIONAL = "optional"       # Nice to have
    BACKUP = "backup"           # Fallback if others don't work


@dataclass
class TripSpot:
    """
    A spot being considered for a trip.
    
    Contains the spot info, location, and forecast windows.
    """
    
    spot_id: str
    spot_name: str
    latitude: float
    longitude: float
    
    # Planning info
    windows: list[SurfWindow] = field(default_factory=list)
    best_window: Optional[SurfWindow] = None
    total_surfable_hours: float = 0.0
    
    # Contextual factors (0-100 scale)
    parking_score: float = 50.0
    crowd_score: float = 50.0    # Higher = less crowded
    safety_score: float = 50.0
    
    # Distance from other spots (km)
    distances: dict[str, float] = field(default_factory=dict)
    
    def overall_score(self) -> float:
        """Calculate overall attractiveness score."""
        if not self.best_window:
            return 0.0
        
        # Weights
        SURF_WEIGHT = 0.6
        CONTEXT_WEIGHT = 0.4
        
        surf_score = self.best_window.average_score
        context_score = (
            self.parking_score * 0.25 +
            self.crowd_score * 0.50 +
            self.safety_score * 0.25
        )
        
        return surf_score * SURF_WEIGHT + context_score * CONTEXT_WEIGHT


@dataclass
class SurfSession:
    """
    A planned surf session within the trip.
    """
    
    spot: TripSpot
    window: SurfWindow
    priority: SessionPriority
    
    # Timing
    start_time: datetime
    end_time: datetime
    duration_hours: float
    
    # Context
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spot": self.spot.spot_name,
            "spot_id": self.spot.spot_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": self.duration_hours,
            "priority": self.priority.value,
            "quality": self.window.quality.value,
            "score": round(self.window.average_score, 1),
            "notes": self.notes,
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
        
        emoji = quality_emoji.get(self.window.quality, "•")
        start_str = self.start_time.strftime("%H:%M")
        end_str = self.end_time.strftime("%H:%M")
        
        lines = [
            f"   {emoji} {self.spot.spot_name}: {start_str}-{end_str} ({self.duration_hours:.1f}h)",
            f"      Quality: {self.window.quality.value.upper()} | Score: {self.window.average_score:.0f}/100",
        ]
        
        if self.notes:
            lines.append(f"      💡 {self.notes[0]}")
        
        if self.warnings:
            lines.append(f"      ⚠️  {self.warnings[0]}")
        
        return "\n".join(lines)


@dataclass
class TripDay:
    """
    A single day in the trip itinerary.
    """
    
    day_date: date
    day_number: int
    sessions: list[SurfSession] = field(default_factory=list)
    travel_time_hours: float = 0.0
    rest_day: bool = False
    notes: list[str] = field(default_factory=list)
    
    @property
    def total_surf_hours(self) -> float:
        """Total hours of surfing planned."""
        return sum(s.duration_hours for s in self.sessions)
    
    @property
    def session_count(self) -> int:
        """Number of sessions planned."""
        return len(self.sessions)
    
    @property
    def best_quality(self) -> Optional[WindowQuality]:
        """Best quality session of the day."""
        if not self.sessions:
            return None
        return max(s.window.quality for s in self.sessions)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.day_date.isoformat(),
            "day_number": self.day_number,
            "sessions": [s.to_dict() for s in self.sessions],
            "total_surf_hours": round(self.total_surf_hours, 1),
            "travel_time_hours": round(self.travel_time_hours, 1),
            "rest_day": self.rest_day,
            "notes": self.notes,
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        date_str = self.day_date.strftime("%A, %b %d")
        
        if self.rest_day:
            return f"\n📅 Day {self.day_number} - {date_str}\n   🛌 REST DAY\n   {self.notes[0] if self.notes else 'Recovery and exploration'}"
        
        lines = [
            f"\n📅 Day {self.day_number} - {date_str}",
            f"   📊 {self.session_count} session(s), {self.total_surf_hours:.1f}h total surf time",
        ]
        
        if self.travel_time_hours > 0:
            lines.append(f"   🚗 ~{self.travel_time_hours:.1f}h travel time")
        
        for session in self.sessions:
            lines.append(session.format_for_display())
        
        for note in self.notes:
            lines.append(f"\n   📝 {note}")
        
        return "\n".join(lines)


@dataclass
class TripItinerary:
    """
    Complete trip itinerary spanning multiple days.
    """
    
    skill_level: str
    start_date: date
    end_date: date
    
    days: list[TripDay] = field(default_factory=list)
    spots_visited: list[str] = field(default_factory=list)
    
    # Summary stats
    total_surf_hours: float = 0.0
    total_travel_hours: float = 0.0
    best_day: Optional[TripDay] = None
    highlight_session: Optional[SurfSession] = None
    
    # Planning notes
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "skill_level": self.skill_level,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "trip_length_days": len(self.days),
            "spots_visited": self.spots_visited,
            "total_surf_hours": round(self.total_surf_hours, 1),
            "total_travel_hours": round(self.total_travel_hours, 1),
            "days": [d.to_dict() for d in self.days],
            "highlight": self.highlight_session.to_dict() if self.highlight_session else None,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        lines = [
            "🏄 SURF TRIP ITINERARY",
            "=" * 50,
            f"📆 {self.start_date.strftime('%b %d')} - {self.end_date.strftime('%b %d, %Y')} ({len(self.days)} days)",
            f"🎯 Skill Level: {self.skill_level}",
            f"📍 Spots: {', '.join(self.spots_visited) if self.spots_visited else 'None'}",
            f"🏄 Total Surf Time: {self.total_surf_hours:.1f} hours",
        ]
        
        if self.total_travel_hours > 0:
            lines.append(f"🚗 Total Travel: ~{self.total_travel_hours:.1f} hours")
        
        if self.highlight_session:
            quality = self.highlight_session.window.quality.value.upper()
            lines.append(f"\n🌟 HIGHLIGHT: {quality} session at {self.highlight_session.spot.spot_name}")
        
        lines.append("\n" + "-" * 50)
        
        for day in self.days:
            lines.append(day.format_for_display())
        
        lines.append("\n" + "-" * 50)
        
        if self.recommendations:
            lines.append("\n💡 RECOMMENDATIONS:")
            for rec in self.recommendations[:5]:
                lines.append(f"   • {rec}")
        
        if self.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for warn in self.warnings[:3]:
                lines.append(f"   • {warn}")
        
        return "\n".join(lines)


class TripPlanner(LoggerMixin):
    """
    Plans multi-day surf trip itineraries.
    
    Optimizes for:
    - Best surf conditions (weather-aware)
    - Minimal travel time between spots
    - Contextual factors (parking, crowds, safety)
    - Appropriate rest days
    """
    
    # Planning constants
    MIN_SESSION_HOURS = 1.5
    MAX_SESSION_HOURS = 4.0
    MAX_DAILY_SURF_HOURS = 6.0
    IDEAL_DAILY_SURF_HOURS = 3.5
    
    # Travel assumptions (km/h average)
    AVERAGE_TRAVEL_SPEED_KPH = 40.0
    
    # Rest day triggers
    CONSECUTIVE_SURF_DAYS_BEFORE_REST = 3
    
    def __init__(
        self,
        condition_assessor: Optional[ConditionAssessor] = None,
        window_finder: Optional[SurfWindowFinder] = None,
    ):
        """
        Initialize the trip planner.
        
        Args:
            condition_assessor: Condition assessment instance.
            window_finder: Window finder instance.
        """
        self._assessor = condition_assessor or ConditionAssessor()
        self._window_finder = window_finder or SurfWindowFinder(self._assessor)
        
        self.log_debug("Trip planner initialized")
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Returns:
            Distance in kilometers.
        """
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _estimate_travel_time(self, distance_km: float) -> float:
        """
        Estimate travel time between spots.
        
        Returns:
            Time in hours.
        """
        return distance_km / self.AVERAGE_TRAVEL_SPEED_KPH
    
    def _build_trip_spots(
        self,
        spots_data: list[dict[str, Any]],
        forecasts_by_spot: dict[str, list[ForecastPoint]],
        skill_level: str,
    ) -> list[TripSpot]:
        """
        Build TripSpot objects with windows and distances.
        
        Args:
            spots_data: List of spot dictionaries with coordinates.
            forecasts_by_spot: Forecasts keyed by spot_id.
            skill_level: User's skill level.
            
        Returns:
            List of TripSpot objects with calculated windows.
        """
        trip_spots = []
        
        for spot_data in spots_data:
            spot_id = spot_data.get("id", spot_data.get("spot_id", ""))
            spot_name = spot_data.get("name", spot_data.get("spot_name", spot_id))
            lat = spot_data.get("latitude", spot_data.get("lat", 0.0))
            lon = spot_data.get("longitude", spot_data.get("lon", 0.0))
            
            trip_spot = TripSpot(
                spot_id=spot_id,
                spot_name=spot_name,
                latitude=lat,
                longitude=lon,
            )
            
            # Get contextual scores if provided
            trip_spot.parking_score = spot_data.get("parking_score", 50.0)
            trip_spot.crowd_score = spot_data.get("crowd_score", 50.0)
            trip_spot.safety_score = spot_data.get("safety_score", 50.0)
            
            # Find windows if we have forecasts
            if spot_id in forecasts_by_spot:
                forecasts = forecasts_by_spot[spot_id]
                if forecasts:
                    result = self._window_finder.find_windows(
                        forecasts=forecasts,
                        skill_level=skill_level,
                        spot_name=spot_name,
                    )
                    trip_spot.windows = result.windows
                    trip_spot.best_window = result.best_window
                    trip_spot.total_surfable_hours = result.total_surfable_hours
            
            trip_spots.append(trip_spot)
        
        # Calculate distances between all spots
        for i, spot_a in enumerate(trip_spots):
            for spot_b in trip_spots[i+1:]:
                distance = self._haversine_distance(
                    spot_a.latitude, spot_a.longitude,
                    spot_b.latitude, spot_b.longitude
                )
                spot_a.distances[spot_b.spot_id] = distance
                spot_b.distances[spot_a.spot_id] = distance
        
        return trip_spots
    
    def _select_daily_sessions(
        self,
        spots: list[TripSpot],
        target_date: date,
        max_travel_km: float = 100.0,
        base_spot: Optional[TripSpot] = None,
    ) -> list[SurfSession]:
        """
        Select best sessions for a specific day.

        Args:
            spots: Available spots with windows.
            target_date: Date to plan for.
            max_travel_km: Maximum travel distance to consider.
            base_spot: Starting location for travel calculations.

        Returns:
            List of sessions for the day.
        """
        sessions = []
        total_hours = 0.0
        current_spot = base_spot

        # Filter windows that overlap with this date
        candidates = []

        for spot in spots:
            for window in spot.windows:
                window_start_date = window.start_time.date()
                window_end_date = window.end_time.date()
                
                # Check if target_date falls within the window's span
                if window_start_date <= target_date <= window_end_date:
                    # Check travel feasibility
                    travel_ok = True
                    travel_time = 0.0
                    
                    if current_spot:
                        distance = current_spot.distances.get(spot.spot_id, 0)
                        if distance > max_travel_km:
                            travel_ok = False
                        travel_time = self._estimate_travel_time(distance)
                    
                    if travel_ok:
                        candidates.append({
                            "spot": spot,
                            "window": window,
                            "score": window.average_score + spot.crowd_score * 0.1,
                            "travel_time": travel_time,
                        })
        
        # Sort by score (best first)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Select sessions up to daily limit
        for candidate in candidates:
            if total_hours >= self.MAX_DAILY_SURF_HOURS:
                break
            
            window = candidate["window"]
            spot = candidate["spot"]
            
            # Determine session duration
            duration = min(
                window.duration_hours,
                self.MAX_SESSION_HOURS,
                self.MAX_DAILY_SURF_HOURS - total_hours
            )
            
            if duration < self.MIN_SESSION_HOURS:
                continue
            
            # Determine priority
            if window.quality in [WindowQuality.EPIC, WindowQuality.EXCELLENT]:
                priority = SessionPriority.MUST_SURF
            elif window.quality == WindowQuality.GOOD:
                priority = SessionPriority.PREFERRED
            else:
                priority = SessionPriority.OPTIONAL
            
            # Determine session start time
            # If window starts on an earlier day, use a reasonable morning time on target_date
            if window.start_time.date() < target_date:
                session_start = datetime.combine(target_date, datetime.min.time().replace(hour=7))
            else:
                session_start = window.start_time
            
            # Create session
            session = SurfSession(
                spot=spot,
                window=window,
                priority=priority,
                start_time=session_start,
                end_time=session_start + timedelta(hours=duration),
                duration_hours=duration,
                notes=window.positive_factors[:2] if window.positive_factors else [],
                warnings=window.warnings[:1] if window.warnings else [],
            )
            
            sessions.append(session)
            total_hours += duration
            current_spot = spot
            
            # Usually one good session per day is enough for planning
            if total_hours >= self.IDEAL_DAILY_SURF_HOURS:
                break
        
        return sessions
    
    def _should_rest(
        self,
        day_number: int,
        consecutive_surf_days: int,
        upcoming_epic_window: bool = False,
    ) -> bool:
        """
        Determine if a rest day is needed.
        
        Args:
            day_number: Current day in trip.
            consecutive_surf_days: Days surfed in a row.
            upcoming_epic_window: Is there an epic window coming?
            
        Returns:
            True if should rest.
        """
        # Rest every N days of consecutive surfing
        if consecutive_surf_days >= self.CONSECUTIVE_SURF_DAYS_BEFORE_REST:
            return True
        
        # Rest before an epic window to be fresh
        if upcoming_epic_window and consecutive_surf_days >= 2:
            return True
        
        return False
    
    def plan_trip(
        self,
        spots_data: list[dict[str, Any]],
        forecasts_by_spot: dict[str, list[ForecastPoint]],
        skill_level: str,
        trip_days: int = 3,
        start_date: Optional[date] = None,
    ) -> TripItinerary:
        """
        Plan a multi-day surf trip.
        
        Args:
            spots_data: List of spot dictionaries with coordinates.
            forecasts_by_spot: Forecasts keyed by spot_id.
            skill_level: User's skill level.
            trip_days: Number of days for the trip.
            start_date: Trip start date (defaults to tomorrow).
            
        Returns:
            TripItinerary with day-by-day plan.
        """
        self.log_info(
            f"Planning {trip_days}-day trip for {skill_level} surfer",
            spots=len(spots_data)
        )
        
        if start_date is None:
            start_date = date.today() + timedelta(days=1)
        
        end_date = start_date + timedelta(days=trip_days - 1)
        
        # Build trip spots with windows
        trip_spots = self._build_trip_spots(
            spots_data, forecasts_by_spot, skill_level
        )
        
        # Sort spots by overall score for initial selection
        trip_spots.sort(key=lambda s: s.overall_score(), reverse=True)
        
        # Check for epic windows in forecast
        epic_days = set()
        for spot in trip_spots:
            for window in spot.windows:
                if window.quality == WindowQuality.EPIC:
                    epic_days.add(window.start_time.date())
        
        # Build day-by-day itinerary
        days = []
        spots_visited = set()
        total_surf = 0.0
        total_travel = 0.0
        consecutive_surf = 0
        highlight_session = None
        best_day = None
        best_day_score = 0
        
        current_date = start_date
        last_spot = None
        
        for day_num in range(1, trip_days + 1):
            # Check if epic window coming in next 2 days
            upcoming_epic = any(
                (current_date + timedelta(days=d)) in epic_days
                for d in range(1, 3)
            )
            
            # Decide if rest day
            is_rest = self._should_rest(day_num, consecutive_surf, upcoming_epic)
            
            if is_rest:
                day = TripDay(
                    day_date=current_date,
                    day_number=day_num,
                    rest_day=True,
                    notes=["Rest and recover for upcoming sessions"]
                )
                consecutive_surf = 0
            else:
                # Select sessions for today
                sessions = self._select_daily_sessions(
                    spots=trip_spots,
                    target_date=current_date,
                    base_spot=last_spot,
                )
                
                # Calculate travel time
                day_travel = 0.0
                if sessions and last_spot:
                    first_session_spot = sessions[0].spot
                    distance = last_spot.distances.get(first_session_spot.spot_id, 0)
                    day_travel = self._estimate_travel_time(distance)
                
                day = TripDay(
                    day_date=current_date,
                    day_number=day_num,
                    sessions=sessions,
                    travel_time_hours=day_travel,
                )
                
                # Update tracking
                for session in sessions:
                    spots_visited.add(session.spot.spot_name)
                    total_surf += session.duration_hours
                    
                    # Track highlight
                    if (not highlight_session or 
                        session.window.quality.value < highlight_session.window.quality.value or
                        (session.window.quality == highlight_session.window.quality and
                         session.window.average_score > highlight_session.window.average_score)):
                        highlight_session = session
                
                total_travel += day_travel
                
                if sessions:
                    last_spot = sessions[-1].spot
                    consecutive_surf += 1
                    
                    # Track best day
                    day_score = sum(s.window.average_score for s in sessions)
                    if day_score > best_day_score:
                        best_day_score = day_score
                        best_day = day
                else:
                    day.rest_day = True
                    day.notes.append("No suitable windows - rest or explore")
                    consecutive_surf = 0
            
            days.append(day)
            current_date += timedelta(days=1)
        
        # Build recommendations
        recommendations = []
        warnings = []
        
        if highlight_session:
            recommendations.append(
                f"Don't miss the {highlight_session.window.quality.value.upper()} "
                f"session at {highlight_session.spot.spot_name}!"
            )
        
        if total_surf < trip_days * 2:
            warnings.append(
                "Limited surf windows available - conditions may be challenging"
            )
        
        if len(spots_visited) > 2 and total_travel > trip_days * 2:
            recommendations.append(
                "Consider staying closer to one region to reduce travel time"
            )
        
        if any(day.rest_day for day in days):
            recommendations.append(
                "Rest days scheduled to keep you fresh for the best conditions"
            )
        
        # Build final itinerary
        itinerary = TripItinerary(
            skill_level=skill_level,
            start_date=start_date,
            end_date=end_date,
            days=days,
            spots_visited=list(spots_visited),
            total_surf_hours=total_surf,
            total_travel_hours=total_travel,
            best_day=best_day,
            highlight_session=highlight_session,
            recommendations=recommendations,
            warnings=warnings,
        )
        
        self.log_info(
            f"Trip planned: {len(days)} days, {len(spots_visited)} spots, "
            f"{total_surf:.1f}h surfing"
        )
        
        return itinerary
    
    def plan_single_day(
        self,
        spots_data: list[dict[str, Any]],
        forecasts_by_spot: dict[str, list[ForecastPoint]],
        skill_level: str,
        target_date: Optional[date] = None,
    ) -> TripDay:
        """
        Plan optimal sessions for a single day.
        
        Args:
            spots_data: List of spot dictionaries.
            forecasts_by_spot: Forecasts keyed by spot_id.
            skill_level: User's skill level.
            target_date: Date to plan (defaults to tomorrow).
            
        Returns:
            TripDay with sessions.
        """
        if target_date is None:
            target_date = date.today() + timedelta(days=1)
        
        trip_spots = self._build_trip_spots(
            spots_data, forecasts_by_spot, skill_level
        )

        sessions = self._select_daily_sessions(
            spots=trip_spots,
            target_date=target_date,
        )
        
        return TripDay(
            day_date=target_date,
            day_number=1,
            sessions=sessions,
        )
    
    def suggest_best_spot(
        self,
        spots_data: list[dict[str, Any]],
        forecasts_by_spot: dict[str, list[ForecastPoint]],
        skill_level: str,
    ) -> Optional[dict[str, Any]]:
        """
        Suggest the single best spot for the current conditions.
        
        Args:
            spots_data: List of spot dictionaries.
            forecasts_by_spot: Forecasts keyed by spot_id.
            skill_level: User's skill level.
            
        Returns:
            Dictionary with spot recommendation.
        """
        trip_spots = self._build_trip_spots(
            spots_data, forecasts_by_spot, skill_level
        )
        
        # Find spot with best window
        best_spot = None
        best_score = 0
        
        for spot in trip_spots:
            score = spot.overall_score()
            if score > best_score:
                best_score = score
                best_spot = spot
        
        if not best_spot or not best_spot.best_window:
            return None
        
        return {
            "spot_id": best_spot.spot_id,
            "spot_name": best_spot.spot_name,
            "score": round(best_score, 1),
            "best_window": best_spot.best_window.to_dict() if best_spot.best_window else None,
            "total_surfable_hours": best_spot.total_surfable_hours,
            "reasoning": f"{best_spot.spot_name} has the best combination of "
                        f"wave quality ({best_spot.best_window.quality.value}) "
                        f"and favorable conditions for {skill_level} surfers.",
        }

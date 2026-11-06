"""
SurfSense Trip Planning Agent

Deterministic sub-agent that identifies optimal surf windows across
multiple spots and generates itineraries that maximise surfing quality.

Wraps existing SurfWindowFinder and TripPlanner from app/planning/.
"""

import math
from datetime import datetime, timedelta
from typing import Any

from app.core.logger import LoggerMixin, get_logger

logger = get_logger(__name__)


class TripPlanningAgent(LoggerMixin):
    """Deterministic sub-agent for itinerary optimisation.

    Exposes tools for the orchestrator to:
    - find_surf_windows: Group consecutive good-condition hours
    - plan_itinerary: Greedy multi-day itinerary optimisation
    - rank_spots: Compare spots by aggregate score
    """

    def find_surf_windows(
        self,
        assessments: list[dict],
        min_hours: int = 2,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[dict]:
        """Group consecutive good-condition hours into surf windows.

        Args:
            assessments: Output of ConditionAssessmentAgent.assess_conditions().
            min_hours: Minimum window length in hours.
            latitude: Spot latitude (decimal degrees). Used for daylight filtering.
            longitude: Spot longitude (decimal degrees). Used for daylight filtering.

        Returns:
            List of windows with start, end, avg_score, hours.
        """
        if not assessments:
            return []
        if assessments and "error" in assessments[0]:
            return []

        # Filter suitable hours
        suitable = [
            a for a in assessments if a.get("rating") in ("ideal", "suitable")
        ]

        # Filter nighttime hours when coordinates are available
        if latitude is not None and longitude is not None:
            suitable = [
                a for a in suitable
                if self._is_daylight(a.get("timestamp", ""), latitude, longitude)
            ]

        if not suitable:
            return []

        # Group consecutive timestamps into windows
        windows: list[list[dict]] = []
        current_window: list[dict] = [suitable[0]]

        for i in range(1, len(suitable)):
            prev_ts = suitable[i - 1].get("timestamp", "")
            curr_ts = suitable[i].get("timestamp", "")

            consecutive = self._is_consecutive(prev_ts, curr_ts)
            if consecutive:
                current_window.append(suitable[i])
            else:
                windows.append(current_window)
                current_window = [suitable[i]]

        if current_window:
            windows.append(current_window)

        # Filter by minimum hours and build result
        result = []
        for window in windows:
            if len(window) < min_hours:
                continue

            scores = [a["score"] for a in window]
            result.append(
                {
                    "start": window[0]["timestamp"],
                    "end": window[-1]["timestamp"],
                    "avg_score": round(sum(scores) / len(scores), 1),
                    "hours": len(window),
                    "peak_score": round(max(scores), 1),
                    "ratings": {
                        "ideal": sum(
                            1 for a in window if a["rating"] == "ideal"
                        ),
                        "suitable": sum(
                            1 for a in window if a["rating"] == "suitable"
                        ),
                    },
                }
            )

        return sorted(result, key=lambda w: w["avg_score"], reverse=True)

    def plan_itinerary(
        self,
        spots_data: dict,
        days: int,
        skill_level: str,
        base_coords: dict | None = None,
    ) -> dict:
        """Generate a multi-day itinerary using greedy optimisation.

        For each day:
        1. Compute best surf window per spot
        2. Penalise by travel time from previous day's spot (Haversine)
        3. Select spot with highest adjusted score
        4. Avoid repeating same spot on consecutive days (soft preference)

        Args:
            spots_data: Dict mapping spot_name -> {assessments, coordinates, windows}.
            days: Number of trip days.
            skill_level: User skill level.
            base_coords: Starting location {"lat": ..., "lon": ...}.

        Returns:
            Itinerary dict with day-by-day schedule.
        """
        if not spots_data:
            return {
                "days": [],
                "total_score": 0,
                "total_travel_km": 0,
                "message": "No spot data available for planning.",
            }

        itinerary: list[dict] = []
        prev_coords = base_coords
        used_spots: set[str] = set()

        for day in range(1, days + 1):
            best_spot = None
            best_adjusted_score = -1.0

            for spot_name, data in spots_data.items():
                windows = data.get("windows", [])
                if not windows:
                    continue

                # Find best window for this day
                day_windows = [
                    w for w in windows if self._is_day(w.get("start", ""), day)
                ]
                if not day_windows:
                    # If we can't match by day, use all windows
                    day_windows = windows

                top_window = max(day_windows, key=lambda w: w.get("avg_score", 0))
                score = top_window.get("avg_score", 0)

                # Travel penalty
                dist = 0.0
                spot_coords = data.get("coordinates")
                if prev_coords and spot_coords:
                    dist = self._haversine(prev_coords, spot_coords)
                    travel_penalty = min(dist / 100, 20)
                    score -= travel_penalty

                # Diversity bonus (avoid same spot on consecutive days)
                if spot_name in used_spots:
                    score -= 5

                if score > best_adjusted_score:
                    best_adjusted_score = score
                    best_spot = {
                        "day": day,
                        "spot": spot_name,
                        "window": top_window,
                        "travel_km": round(dist, 1) if dist else 0,
                    }

            if best_spot:
                itinerary.append(best_spot)
                spot_name = best_spot["spot"]
                prev_coords = spots_data[spot_name].get("coordinates")
                used_spots.add(spot_name)

        total_score = sum(
            d["window"].get("avg_score", 0) for d in itinerary
        )
        total_travel = sum(d["travel_km"] for d in itinerary)

        return {
            "days": itinerary,
            "total_score": round(total_score, 1),
            "total_travel_km": round(total_travel, 1),
        }

    def rank_spots(self, spot_assessments: dict[str, list[dict]]) -> list[dict]:
        """Rank spots by aggregate condition score.

        Args:
            spot_assessments: Dict mapping spot_name -> list of assessment dicts.

        Returns:
            Sorted list with avg_score, ideal/suitable/unsafe hour counts.
        """
        rankings = []
        for spot, assessments in spot_assessments.items():
            if not assessments or (assessments and "error" in assessments[0]):
                continue
            scores = [a.get("score", 0) for a in assessments]
            rankings.append(
                {
                    "spot": spot,
                    "avg_score": (
                        round(sum(scores) / len(scores), 1) if scores else 0
                    ),
                    "ideal_hours": sum(
                        1 for a in assessments if a.get("rating") == "ideal"
                    ),
                    "suitable_hours": sum(
                        1 for a in assessments if a.get("rating") == "suitable"
                    ),
                    "unsafe_hours": sum(
                        1 for a in assessments if a.get("rating") == "unsafe"
                    ),
                }
            )
        return sorted(rankings, key=lambda r: r["avg_score"], reverse=True)

    @staticmethod
    def _haversine(coord1: dict, coord2: dict) -> float:
        """Compute distance in km between two coordinate dicts."""
        R = 6371
        lat1 = math.radians(coord1.get("lat", 0))
        lon1 = math.radians(coord1.get("lon", 0))
        lat2 = math.radians(coord2.get("lat", 0))
        lon2 = math.radians(coord2.get("lon", 0))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _is_consecutive(ts1: str, ts2: str) -> bool:
        """Check if two ISO timestamps are within ~1.5 hours of each other."""
        try:
            # Handle both with and without timezone info
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    t1 = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(ts2.replace("Z", "+00:00"))
                    break
                except ValueError:
                    continue
            else:
                return False
            return abs((t2 - t1).total_seconds()) <= 5400  # 1.5 hours
        except Exception:
            return False

    @staticmethod
    def _is_day(timestamp_str: str, day_number: int) -> bool:
        """Check if a timestamp falls on the Nth day from today."""
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            target_date = datetime.utcnow().date() + timedelta(days=day_number - 1)
            return ts.date() == target_date
        except Exception:
            return False

    @staticmethod
    def _sunrise_sunset_utc(
        date: datetime, lat_deg: float, lon_deg: float
    ) -> tuple[float, float] | tuple[None, None]:
        """Return (sunrise_utc_h, sunset_utc_h) using Spencer (1971) formula, ±2 min.

        Returns (0.0, 24.0) for polar day, (None, None) for polar night.
        """
        doy = date.timetuple().tm_yday
        b_rad = math.radians((doy - 1) * 360 / 365)

        # Equation of time (minutes)
        eqt_min = 229.18 * (
            0.000075
            + 0.001868 * math.cos(b_rad)
            - 0.032077 * math.sin(b_rad)
            - 0.014615 * math.cos(2 * b_rad)
            - 0.04089 * math.sin(2 * b_rad)
        )

        # Solar declination (radians)
        decl = math.radians(-23.45 * math.cos(math.radians(360 / 365 * (doy + 10))))

        lat = math.radians(lat_deg)
        # -0.833° accounts for atmospheric refraction and solar disk radius
        cos_ha = (
            math.sin(math.radians(-0.833)) - math.sin(lat) * math.sin(decl)
        ) / (math.cos(lat) * math.cos(decl))

        if cos_ha <= -1.0:
            return 0.0, 24.0  # midnight sun
        if cos_ha >= 1.0:
            return None, None  # polar night

        ha_hours = math.degrees(math.acos(cos_ha)) / 15.0
        solar_noon_utc = 12.0 - lon_deg / 15.0 - eqt_min / 60.0
        return solar_noon_utc - ha_hours, solar_noon_utc + ha_hours

    @staticmethod
    def _is_daylight(timestamp_str: str, lat_deg: float, lon_deg: float) -> bool:
        """Return True if the UTC timestamp falls within daylight at the given location."""
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if ts.utcoffset() is not None:
                utc_tt = ts.utctimetuple()
                ts_date = datetime(utc_tt.tm_year, utc_tt.tm_mon, utc_tt.tm_mday)
                ts_hour_utc = utc_tt.tm_hour + utc_tt.tm_min / 60.0
            else:
                ts_date = ts.replace(hour=0, minute=0, second=0, microsecond=0)
                ts_hour_utc = ts.hour + ts.minute / 60.0

            sunrise, sunset = TripPlanningAgent._sunrise_sunset_utc(ts_date, lat_deg, lon_deg)
            if sunrise is None:
                return False  # polar night — no daylight
            return sunrise <= ts_hour_utc <= sunset
        except Exception:
            return True  # fail-open: don't filter if calculation fails

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return OpenAI function-calling schemas for all tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "find_surf_windows",
                    "description": (
                        "Find contiguous time windows with good surf conditions "
                        "at a spot. Requires that assess_conditions has been "
                        "called first."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_name": {"type": "string"},
                            "min_hours": {
                                "type": "integer",
                                "default": 2,
                                "description": "Minimum window duration in hours",
                            },
                        },
                        "required": ["spot_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_itinerary",
                    "description": (
                        "Generate a multi-day surf trip itinerary across "
                        "multiple spots, optimising for surf quality and "
                        "minimising travel. Requires forecasts and assessments "
                        "for all candidate spots."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of candidate spot names",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of trip days",
                            },
                            "skill_level": {
                                "type": "string",
                                "enum": [
                                    "beginner",
                                    "intermediate",
                                    "advanced",
                                ],
                            },
                        },
                        "required": ["spot_names", "days", "skill_level"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rank_spots",
                    "description": (
                        "Rank multiple surf spots by overall condition quality "
                        "for a given period. Requires that assess_conditions "
                        "has been called for each spot."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Spot names to rank",
                            },
                        },
                        "required": ["spot_names"],
                    },
                },
            },
        ]

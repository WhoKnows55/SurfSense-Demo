"""
Surf Spot Database

Provides access to comprehensive surf spot information.
Supports searching by name, location, skill level, and conditions.
"""

import json
from pathlib import Path
from typing import Any, Optional
from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field

from app.core.logger import LoggerMixin, get_logger

logger = get_logger(__name__)


# ============================================================
# Data Models
# ============================================================

class BreakType(str, Enum):
    """Type of wave break."""
    BEACH = "beach"
    REEF = "reef"
    POINT = "point"
    RIVERMOUTH = "rivermouth"


class WaveDirection(str, Enum):
    """Direction the wave breaks."""
    LEFT = "left"
    RIGHT = "right"
    RIGHT_AND_LEFT = "right_and_left"


class CrowdLevel(str, Enum):
    """How crowded the spot typically is."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SkillLevel(str, Enum):
    """Surfer skill level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    ALL = "all"


class SpotLocation(BaseModel):
    """Geographic location of a surf spot."""
    region: str
    country: str
    state: Optional[str] = None
    latitude: float
    longitude: float
    timezone: str


class SpotCharacteristics(BaseModel):
    """Physical characteristics of the break."""
    break_type: BreakType
    wave_direction: WaveDirection
    bottom: str
    crowd_level: CrowdLevel
    water_quality: str


class SkillLevels(BaseModel):
    """Skill level requirements for a spot."""
    minimum: SkillLevel
    recommended: SkillLevel
    beginner_friendly: bool


class BestConditions(BaseModel):
    """Optimal conditions for the spot."""
    swell_direction: list[str]
    swell_size_min_m: float
    swell_size_max_m: float
    swell_period_min: float
    wind_direction: list[str]
    tide: list[str]
    season: list[str]


class SpotInfo(BaseModel):
    """Complete information about a surf spot."""
    id: str
    name: str
    location: SpotLocation
    characteristics: SpotCharacteristics
    skill_levels: SkillLevels
    best_conditions: BestConditions
    hazards: list[str]
    facilities: list[str]
    description: str
    local_tips: str


# ============================================================
# Spot Database
# ============================================================

class SpotDatabase(LoggerMixin):
    """
    Database of surf spots with search and filtering capabilities.
    
    Loads spot data from JSON file and provides methods for:
    - Searching by name or keyword
    - Filtering by skill level
    - Finding spots by location/region
    - Getting spots matching specific conditions
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the spot database.
        
        Args:
            data_path: Path to spots.json file. If None, uses default location.
        """
        if data_path is None:
            # Default to data/spots.json relative to project root
            data_path = Path(__file__).parent.parent.parent / "data" / "spots.json"
        
        self._data_path = data_path
        self._spots: dict[str, SpotInfo] = {}
        self._load_spots()
    
    def _load_spots(self) -> None:
        """Load spots from JSON file."""
        try:
            if not self._data_path.exists():
                self.log_warning(f"Spots file not found: {self._data_path}")
                return
            
            with open(self._data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for spot_data in data.get("spots", []):
                try:
                    spot = SpotInfo(**spot_data)
                    self._spots[spot.id] = spot
                except Exception as e:
                    self.log_warning(f"Failed to load spot: {e}")
                    continue
            
            self.log_info(f"Loaded {len(self._spots)} surf spots")
            
        except Exception as e:
            self.log_error(f"Failed to load spots database: {e}")
    
    @property
    def spot_count(self) -> int:
        """Get total number of spots in database."""
        return len(self._spots)
    
    @property
    def all_spots(self) -> list[SpotInfo]:
        """Get all spots in database."""
        return list(self._spots.values())
    
    def get_spot(self, spot_id: str) -> Optional[SpotInfo]:
        """
        Get spot by ID.
        
        Args:
            spot_id: Spot identifier (e.g., "pipeline", "waikiki")
            
        Returns:
            SpotInfo if found, None otherwise.
        """
        return self._spots.get(spot_id.lower())
    
    def search_by_name(self, query: str) -> list[SpotInfo]:
        """
        Search spots by name (fuzzy matching).
        
        Args:
            query: Search term.
            
        Returns:
            List of matching spots, sorted by relevance.
        """
        query = query.lower().strip()
        results = []
        
        for spot in self._spots.values():
            name = spot.name.lower()
            spot_id = spot.id.lower()
            
            # Exact match (highest priority)
            if name == query or spot_id == query:
                results.insert(0, spot)
            # Starts with query
            elif name.startswith(query) or spot_id.startswith(query):
                results.append(spot)
            # Contains query
            elif query in name or query in spot_id:
                results.append(spot)
        
        return results
    
    def search_by_region(self, region: str) -> list[SpotInfo]:
        """
        Find spots in a region.
        
        Args:
            region: Region name (e.g., "North Shore", "Southern California")
            
        Returns:
            List of spots in that region.
        """
        region = region.lower()
        return [
            spot for spot in self._spots.values()
            if region in spot.location.region.lower()
        ]
    
    def search_by_country(self, country: str) -> list[SpotInfo]:
        """
        Find spots in a country.
        
        Args:
            country: Country name (e.g., "USA", "Australia")
            
        Returns:
            List of spots in that country.
        """
        country = country.lower()
        return [
            spot for spot in self._spots.values()
            if country in spot.location.country.lower()
        ]
    
    def filter_by_skill(
        self, 
        skill_level: SkillLevel,
        beginner_friendly_only: bool = False
    ) -> list[SpotInfo]:
        """
        Filter spots by skill level.
        
        Args:
            skill_level: Minimum skill level of the surfer.
            beginner_friendly_only: Only return beginner-friendly spots.
            
        Returns:
            List of appropriate spots.
        """
        skill_order = {
            SkillLevel.BEGINNER: 1,
            SkillLevel.INTERMEDIATE: 2,
            SkillLevel.ADVANCED: 3,
            SkillLevel.EXPERT: 4,
            SkillLevel.ALL: 1,
        }
        
        surfer_skill = skill_order.get(skill_level, 1)
        results = []
        
        for spot in self._spots.values():
            # Check if beginner-friendly filter is applied
            if beginner_friendly_only and not spot.skill_levels.beginner_friendly:
                continue
            
            # Check if surfer has minimum skill
            min_skill = skill_order.get(spot.skill_levels.minimum, 1)
            if surfer_skill >= min_skill:
                results.append(spot)
        
        return results
    
    def filter_by_break_type(self, break_type: BreakType) -> list[SpotInfo]:
        """
        Filter spots by break type.
        
        Args:
            break_type: Type of break (beach, reef, point).
            
        Returns:
            List of spots with that break type.
        """
        return [
            spot for spot in self._spots.values()
            if spot.characteristics.break_type == break_type
        ]
    
    def find_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100
    ) -> list[tuple[SpotInfo, float]]:
        """
        Find spots near a location.
        
        Args:
            latitude: Latitude of reference point.
            longitude: Longitude of reference point.
            radius_km: Search radius in kilometers.
            
        Returns:
            List of (spot, distance_km) tuples, sorted by distance.
        """
        from math import radians, cos, sin, sqrt, atan2
        
        results = []
        
        for spot in self._spots.values():
            # Haversine formula
            lat1, lon1 = radians(latitude), radians(longitude)
            lat2 = radians(spot.location.latitude)
            lon2 = radians(spot.location.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            distance = 6371 * c  # Earth's radius in km
            
            if distance <= radius_km:
                results.append((spot, round(distance, 1)))
        
        return sorted(results, key=lambda x: x[1])
    
    def get_spots_for_conditions(
        self,
        swell_direction: Optional[str] = None,
        swell_size_m: Optional[float] = None,
        wind_direction: Optional[str] = None,
        season: Optional[str] = None,
    ) -> list[SpotInfo]:
        """
        Find spots that work for given conditions.
        
        Args:
            swell_direction: Cardinal direction of swell (N, NW, etc.)
            swell_size_m: Swell size in meters.
            wind_direction: Cardinal direction of wind.
            season: Season (winter, summer, fall, spring).
            
        Returns:
            List of spots that work for those conditions.
        """
        results = []
        
        for spot in self._spots.values():
            conditions = spot.best_conditions
            
            # Check swell direction
            if swell_direction:
                if swell_direction.upper() not in [
                    d.upper() for d in conditions.swell_direction
                ]:
                    continue
            
            # Check swell size
            if swell_size_m:
                if not (conditions.swell_size_min_m <= swell_size_m 
                        <= conditions.swell_size_max_m):
                    continue
            
            # Check wind direction
            if wind_direction:
                if wind_direction.upper() not in [
                    d.upper() for d in conditions.wind_direction
                ]:
                    continue
            
            # Check season
            if season:
                if season.lower() not in [
                    s.lower() for s in conditions.season
                ]:
                    continue
            
            results.append(spot)
        
        return results
    
    def get_coordinates(self, spot_name: str) -> Optional[tuple[float, float]]:
        """
        Get coordinates for a spot by name.
        
        Args:
            spot_name: Name or ID of the spot.
            
        Returns:
            Tuple of (latitude, longitude) if found, None otherwise.
        """
        # Try direct lookup
        spot = self.get_spot(spot_name)
        
        # Try search
        if not spot:
            matches = self.search_by_name(spot_name)
            if matches:
                spot = matches[0]
        
        if spot:
            return (spot.location.latitude, spot.location.longitude)
        
        return None
    
    def to_dict(self, spot: SpotInfo) -> dict[str, Any]:
        """Convert spot to dictionary for API responses."""
        return {
            "id": spot.id,
            "name": spot.name,
            "location": {
                "region": spot.location.region,
                "country": spot.location.country,
                "state": spot.location.state,
                "coordinates": {
                    "lat": spot.location.latitude,
                    "lon": spot.location.longitude,
                },
                "timezone": spot.location.timezone,
            },
            "characteristics": {
                "break_type": spot.characteristics.break_type.value,
                "wave_direction": spot.characteristics.wave_direction.value,
                "bottom": spot.characteristics.bottom,
                "crowd_level": spot.characteristics.crowd_level.value,
            },
            "skill_levels": {
                "minimum": spot.skill_levels.minimum.value,
                "recommended": spot.skill_levels.recommended.value,
                "beginner_friendly": spot.skill_levels.beginner_friendly,
            },
            "best_conditions": {
                "swell_direction": spot.best_conditions.swell_direction,
                "swell_size_ft": f"{spot.best_conditions.swell_size_min_ft}-{spot.best_conditions.swell_size_max_ft}",
                "wind_direction": spot.best_conditions.wind_direction,
                "season": spot.best_conditions.season,
            },
            "hazards": spot.hazards,
            "facilities": spot.facilities,
            "description": spot.description,
            "local_tips": spot.local_tips,
        }


# ============================================================
# Singleton Access
# ============================================================

_spot_database: Optional[SpotDatabase] = None


def get_spot_database() -> SpotDatabase:
    """
    Get the singleton spot database instance.
    
    Returns:
        SpotDatabase instance.
    """
    global _spot_database
    if _spot_database is None:
        _spot_database = SpotDatabase()
    return _spot_database

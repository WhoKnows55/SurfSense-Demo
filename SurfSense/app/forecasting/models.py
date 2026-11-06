"""
SurfSense Forecast Data Models

Unified data schema for surf forecasts that works with multiple data sources.
All forecast data is normalized to these models regardless of the source API.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WindDirection(str, Enum):
    """Cardinal wind directions."""
    N = "N"
    NNE = "NNE"
    NE = "NE"
    ENE = "ENE"
    E = "E"
    ESE = "ESE"
    SE = "SE"
    SSE = "SSE"
    S = "S"
    SSW = "SSW"
    SW = "SW"
    WSW = "WSW"
    W = "W"
    WNW = "WNW"
    NW = "NW"
    NNW = "NNW"


class SwellDirection(str, Enum):
    """Swell direction categories for surfing."""
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


class TideState(str, Enum):
    """Tide state categories."""
    LOW = "low"
    RISING = "rising"
    HIGH = "high"
    FALLING = "falling"


class Coordinates(BaseModel):
    """Geographic coordinates."""
    
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )

    def __str__(self) -> str:
        return f"{self.latitude:.4f}, {self.longitude:.4f}"


class SpotMetadata(BaseModel):
    """Basic surf spot information included with forecasts."""
    
    name: str = Field(
        ...,
        min_length=1,
        description="Name of the surf spot",
    )
    coordinates: Coordinates = Field(
        ...,
        description="Geographic location of the spot",
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for the spot (e.g., 'America/Los_Angeles')",
    )


class WaveData(BaseModel):
    """Wave conditions at a specific time."""
    
    height_min: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Minimum wave height in meters",
    )
    height_max: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Maximum wave height in meters",
    )
    
    @field_validator("height_max")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Ensure max height is >= min height."""
        if "height_min" in info.data and v < info.data["height_min"]:
            raise ValueError("height_max must be >= height_min")
        return v

    @property
    def height_avg(self) -> float:
        """Average wave height."""
        return (self.height_min + self.height_max) / 2

    def height_range(self) -> tuple[float, float]:
        """Return wave height range in meters."""
        return (self.height_min, self.height_max)


class SwellData(BaseModel):
    """Primary swell information."""
    
    height: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Swell height in meters",
    )
    period: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Swell period in seconds",
    )
    direction_degrees: float = Field(
        ...,
        ge=0.0,
        lt=360.0,
        description="Swell direction in degrees (0-359, where 0 is North)",
    )
    direction: Optional[SwellDirection] = Field(
        default=None,
        description="Swell direction as cardinal direction",
    )

    @model_validator(mode="after")
    def compute_direction(self) -> "SwellData":
        """Compute cardinal direction from degrees if not provided."""
        if self.direction is None:
            self.direction = self._degrees_to_cardinal(self.direction_degrees)
        return self

    @staticmethod
    def _degrees_to_cardinal(degrees: float) -> SwellDirection:
        """Convert degrees to cardinal direction."""
        directions = [
            SwellDirection.N, SwellDirection.NE, SwellDirection.E, SwellDirection.SE,
            SwellDirection.S, SwellDirection.SW, SwellDirection.W, SwellDirection.NW,
        ]
        index = round(degrees / 45) % 8
        return directions[index]


class WindData(BaseModel):
    """Wind conditions."""
    
    speed: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Wind speed in km/h",
    )
    gust: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=300.0,
        description="Wind gust speed in km/h",
    )
    direction_degrees: float = Field(
        ...,
        ge=0.0,
        lt=360.0,
        description="Wind direction in degrees (0-359, where 0 is North)",
    )
    direction: Optional[WindDirection] = Field(
        default=None,
        description="Wind direction as cardinal direction",
    )

    @model_validator(mode="after")
    def compute_direction(self) -> "WindData":
        """Compute cardinal direction from degrees if not provided."""
        if self.direction is None:
            self.direction = self._degrees_to_cardinal(self.direction_degrees)
        return self

    @staticmethod
    def _degrees_to_cardinal(degrees: float) -> WindDirection:
        """Convert degrees to 16-point cardinal direction."""
        directions = [
            WindDirection.N, WindDirection.NNE, WindDirection.NE, WindDirection.ENE,
            WindDirection.E, WindDirection.ESE, WindDirection.SE, WindDirection.SSE,
            WindDirection.S, WindDirection.SSW, WindDirection.SW, WindDirection.WSW,
            WindDirection.W, WindDirection.WNW, WindDirection.NW, WindDirection.NNW,
        ]
        index = round(degrees / 22.5) % 16
        return directions[index]

    def speed_in_knots(self) -> float:
        """Convert wind speed to knots."""
        return self.speed * 0.54

    def speed_in_mph(self) -> float:
        """Convert wind speed to mph."""
        return self.speed * 0.621


class TideData(BaseModel):
    """Tide information."""
    
    height: float = Field(
        ...,
        ge=-5.0,
        le=15.0,
        description="Tide height in meters relative to mean sea level",
    )
    state: Optional[TideState] = Field(
        default=None,
        description="Current tide state (low, rising, high, falling)",
    )


class WeatherData(BaseModel):
    """General weather conditions."""
    
    temperature: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=60.0,
        description="Air temperature in Celsius",
    )
    water_temperature: Optional[float] = Field(
        default=None,
        ge=-5.0,
        le=40.0,
        description="Water temperature in Celsius",
    )
    description: Optional[str] = Field(
        default=None,
        description="Weather description (e.g., 'Partly cloudy')",
    )
    cloud_cover: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Cloud cover percentage",
    )
    precipitation: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Precipitation in mm/h",
    )
    visibility: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Visibility in kilometers",
    )


class ForecastPoint(BaseModel):
    """
    A single forecast data point for a specific time.
    
    This is the core model representing surf conditions at one moment in time.
    All forecast sources are normalized to this format.
    """
    
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp for this forecast point",
    )
    waves: WaveData = Field(
        ...,
        description="Wave conditions",
    )
    swell: SwellData = Field(
        ...,
        description="Primary swell data",
    )
    wind: WindData = Field(
        ...,
        description="Wind conditions",
    )
    tide: Optional[TideData] = Field(
        default=None,
        description="Tide information (may not be available from all sources)",
    )
    weather: Optional[WeatherData] = Field(
        default=None,
        description="General weather conditions",
    )

    @property
    def is_offshore_wind(self) -> bool:
        """
        Check if wind is offshore (generally good for surfing).
        
        Note: This is a simplified check. Real offshore detection
        depends on the specific beach orientation.
        """
        # Simplified: winds from land (E, NE, SE for west-facing beaches)
        offshore_directions = {
            WindDirection.E, WindDirection.ENE, WindDirection.ESE,
            WindDirection.NE, WindDirection.SE,
        }
        return self.wind.direction in offshore_directions

    @property
    def is_light_wind(self) -> bool:
        """Check if wind is light (< 15 km/h)."""
        return self.wind.speed < 15

    def summary(self) -> str:
        """Generate a human-readable summary of conditions."""
        wave_m = self.waves.height_range()
        return (
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} UTC: "
            f"Waves {wave_m[0]:.1f}-{wave_m[1]:.1f}m, "
            f"Swell {self.swell.period:.0f}s from {self.swell.direction.value if self.swell.direction else 'N/A'}, "
            f"Wind {self.wind.speed:.0f}km/h {self.wind.direction.value if self.wind.direction else 'N/A'}"
        )


class DataSource(str, Enum):
    """Forecast data source identifier."""
    STORMGLASS = "stormglass"
    OPEN_METEO = "open_meteo"
    SURFLINE = "surfline"
    NOAA = "noaa"
    LOCAL_MODEL = "local_model"
    UNKNOWN = "unknown"


class ForecastResponse(BaseModel):
    """
    Complete forecast response containing multiple forecast points.
    
    This is the main response model returned by the forecast service.
    """
    
    spot: SpotMetadata = Field(
        ...,
        description="Information about the surf spot",
    )
    forecasts: list[ForecastPoint] = Field(
        ...,
        min_length=1,
        description="List of forecast points ordered by timestamp",
    )
    source: DataSource = Field(
        default=DataSource.UNKNOWN,
        description="Data source for this forecast",
    )
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this forecast was retrieved",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When this forecast data expires (for caching)",
    )

    @property
    def start_time(self) -> datetime:
        """First forecast timestamp."""
        return self.forecasts[0].timestamp

    @property
    def end_time(self) -> datetime:
        """Last forecast timestamp."""
        return self.forecasts[-1].timestamp

    @property
    def duration_hours(self) -> float:
        """Total forecast duration in hours."""
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600

    def get_forecast_at(self, target_time: datetime) -> Optional[ForecastPoint]:
        """
        Get the forecast point closest to the target time.
        
        Args:
            target_time: The time to find forecast for.
            
        Returns:
            Closest ForecastPoint or None if no forecasts available.
        """
        if not self.forecasts:
            return None
        
        closest = min(
            self.forecasts,
            key=lambda f: abs((f.timestamp - target_time).total_seconds())
        )
        return closest

    def get_forecasts_for_date(self, date: datetime) -> list[ForecastPoint]:
        """
        Get all forecast points for a specific date.
        
        Args:
            date: The date to filter by.
            
        Returns:
            List of ForecastPoints for that date.
        """
        return [
            f for f in self.forecasts
            if f.timestamp.date() == date.date()
        ]

    def best_conditions(self, max_wind_speed: float = 20.0) -> list[ForecastPoint]:
        """
        Find forecast points with the best surfing conditions.
        
        Simple heuristic: larger waves + longer period + lighter wind = better.
        
        Args:
            max_wind_speed: Maximum acceptable wind speed in km/h.
            
        Returns:
            List of ForecastPoints sorted by quality (best first).
        """
        suitable = [f for f in self.forecasts if f.wind.speed <= max_wind_speed]
        
        def score(f: ForecastPoint) -> float:
            # Higher score = better conditions
            wave_score = f.waves.height_avg * 10
            period_score = f.swell.period * 2
            wind_penalty = f.wind.speed * 0.5
            return wave_score + period_score - wind_penalty
        
        return sorted(suitable, key=score, reverse=True)

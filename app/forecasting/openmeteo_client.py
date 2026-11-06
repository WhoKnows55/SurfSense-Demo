"""
Open-Meteo Marine Weather API Client

Fetches marine weather data from Open-Meteo's free API.
https://open-meteo.com/en/docs/marine-weather-api

FREE - No API key required!
Global coverage with 5km resolution.
Up to 16 days forecast.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import (
    Coordinates,
    DataSource,
    ForecastPoint,
    ForecastResponse,
    SpotMetadata,
    SwellData,
    TideData,
    WaveData,
    WeatherData,
    WindData,
)

logger = get_logger(__name__)


class OpenMeteoClient(LoggerMixin):
    """
    Client for Open-Meteo Marine Weather API.
    
    FREE - No API key required!
    
    Features:
    - Global coverage at 5km resolution
    - Up to 16 days forecast
    - Wave height, period, direction
    - Swell data (primary, secondary, tertiary)
    - Sea surface temperature
    - Ocean currents and tides
    """
    
    BASE_URL = "https://marine-api.open-meteo.com/v1/marine"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Marine parameters to request
    MARINE_PARAMS = [
        "wave_height",
        "wave_direction",
        "wave_period",
        "wave_peak_period",
        "swell_wave_height",
        "swell_wave_direction",
        "swell_wave_period",
        "swell_wave_peak_period",
        "wind_wave_height",
        "wind_wave_direction",
        "wind_wave_period",
        "sea_surface_temperature",
        "sea_level_height_msl",
        "ocean_current_velocity",
        "ocean_current_direction",
    ]
    
    # Weather parameters for wind data
    WEATHER_PARAMS = [
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "temperature_2m",
        "cloud_cover",
        "precipitation",
    ]
    
    def __init__(self, timeout: float = 15.0):
        """
        Initialize the Open-Meteo client.
        
        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
    
    @property
    def source_name(self) -> str:
        """Return the name of this data source."""
        return "open_meteo"
    
    @property
    def is_configured(self) -> bool:
        """Open-Meteo is always configured (no API key needed)."""
        return True
    
    @property
    def requires_api_key(self) -> bool:
        """Open-Meteo does not require an API key."""
        return False
    
    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        spot_name: str = "Unknown Spot",
        days: int = 7,
    ) -> ForecastResponse:
        """
        Fetch marine weather forecast from Open-Meteo.
        
        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            spot_name: Name of the surf spot.
            days: Number of days to forecast (1-16).
            
        Returns:
            ForecastResponse with forecast points.
        """
        days = min(max(days, 1), 16)  # Clamp to 1-16 days
        
        self.log_info(
            f"Fetching Open-Meteo forecast for {spot_name}",
            lat=latitude,
            lon=longitude,
            days=days,
        )
        
        try:
            # Fetch marine data and weather data with 200ms spacing between requests
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                marine_task = self._fetch_marine_data(client, latitude, longitude, days)

                # Add 200ms sleep before weather request to pace API calls
                await asyncio.sleep(0.2)
                weather_task = self._fetch_weather_data(client, latitude, longitude, days)

                marine_data, weather_data = await asyncio.gather(
                    marine_task, weather_task, return_exceptions=True
                )
                
                # Handle potential errors
                if isinstance(marine_data, Exception):
                    self.log_error(f"Marine data fetch failed: {marine_data}")
                    raise marine_data
                
                if isinstance(weather_data, Exception):
                    self.log_warning(f"Weather data fetch failed: {weather_data}")
                    weather_data = None  # Weather is optional, marine is required
            
            # Transform to our schema
            forecast_points = self._transform_response(marine_data, weather_data)
            
            self.log_info(
                f"Received {len(forecast_points)} forecast points from Open-Meteo",
                spot=spot_name,
            )
            
            # Determine timezone from response
            timezone = marine_data.get("timezone", "UTC")
            
            return ForecastResponse(
                spot=SpotMetadata(
                    name=spot_name,
                    coordinates=Coordinates(
                        latitude=latitude,
                        longitude=longitude,
                    ),
                    timezone=timezone,
                ),
                forecasts=forecast_points,
                source=DataSource.OPEN_METEO,
                fetched_at=datetime.utcnow(),
            )
            
        except httpx.TimeoutException:
            raise Exception(f"Open-Meteo request timed out after {self.timeout}s")
        except httpx.RequestError as e:
            raise Exception(f"Open-Meteo network error: {e}")
    
    async def _fetch_marine_data(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        days: int,
    ) -> dict[str, Any]:
        """Fetch marine data from Open-Meteo."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(self.MARINE_PARAMS),
            "forecast_days": days,
            "timezone": "auto",
        }
        
        response = await client.get(self.BASE_URL, params=params)
        
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            reason = error_data.get("reason", response.text)
            raise Exception(f"Open-Meteo API error: {response.status_code} - {reason}")
        
        return response.json()
    
    async def _fetch_weather_data(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        days: int,
    ) -> dict[str, Any]:
        """Fetch weather data (wind, temperature) from Open-Meteo."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(self.WEATHER_PARAMS),
            "forecast_days": days,
            "timezone": "auto",
            "wind_speed_unit": "kmh",
        }
        
        response = await client.get(self.WEATHER_URL, params=params)
        
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            reason = error_data.get("reason", response.text)
            raise Exception(f"Open-Meteo weather API error: {response.status_code} - {reason}")
        
        return response.json()
    
    def _transform_response(
        self,
        marine_data: dict[str, Any],
        weather_data: Optional[dict[str, Any]],
    ) -> list[ForecastPoint]:
        """
        Transform Open-Meteo response to our forecast models.
        """
        forecast_points = []
        
        hourly = marine_data.get("hourly", {})
        times = hourly.get("time", [])
        
        # Get weather hourly data if available
        weather_hourly = weather_data.get("hourly", {}) if weather_data else {}
        
        for i, time_str in enumerate(times):
            try:
                point = self._parse_hour(hourly, weather_hourly, i, time_str)
                if point:
                    forecast_points.append(point)
            except Exception as e:
                self.log_warning(f"Failed to parse hour {i}: {e}")
                continue
        
        return forecast_points
    
    def _parse_hour(
        self,
        marine: dict[str, Any],
        weather: dict[str, Any],
        idx: int,
        time_str: str,
    ) -> Optional[ForecastPoint]:
        """Parse a single hour of forecast data."""
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(time_str)
        except ValueError:
            return None
        
        # Helper to safely get value at index
        def get_val(data: dict, key: str, default: float = 0.0) -> float:
            values = data.get(key, [])
            if idx < len(values) and values[idx] is not None:
                return float(values[idx])
            return default
        
        # Wave data - Open-Meteo gives single wave_height, we estimate range
        wave_height = get_val(marine, "wave_height")
        waves = WaveData(
            height_min=round(wave_height * 0.8, 2),  # Estimate min as 80%
            height_max=round(wave_height * 1.2, 2),  # Estimate max as 120%
        )
        
        # Swell data - use swell-specific values
        swell_height = get_val(marine, "swell_wave_height", wave_height)
        swell_period = get_val(marine, "swell_wave_peak_period") or get_val(marine, "swell_wave_period")
        swell_direction = get_val(marine, "swell_wave_direction")
        
        swell = SwellData(
            height=max(0.0, min(swell_height, 20.0)),
            period=max(0.0, min(swell_period, 30.0)),
            direction_degrees=swell_direction % 360 if swell_direction else 0.0,
        )
        
        # Wind data - from weather API
        wind_speed = get_val(weather, "wind_speed_10m", 0.0)
        wind_direction = get_val(weather, "wind_direction_10m", 0.0)
        wind_gust = get_val(weather, "wind_gusts_10m")
        
        wind = WindData(
            speed=max(0.0, min(wind_speed, 200.0)),
            direction_degrees=wind_direction % 360,
            gust=wind_gust if wind_gust and wind_gust > wind_speed else None,
        )
        
        # Tide data from sea level height
        sea_level = get_val(marine, "sea_level_height_msl", 0.0)
        tide = TideData(height=sea_level)
        
        # Weather data
        temp = get_val(weather, "temperature_2m")
        water_temp = get_val(marine, "sea_surface_temperature")
        cloud_cover = get_val(weather, "cloud_cover", 0.0)
        
        # Generate description from cloud cover
        if cloud_cover > 80:
            description = "Overcast"
        elif cloud_cover > 50:
            description = "Cloudy"
        elif cloud_cover > 20:
            description = "Partly cloudy"
        else:
            description = "Clear"
        
        weather_info = WeatherData(
            description=description,
            temperature=temp if temp else None,
            water_temperature=water_temp if water_temp else None,
        )
        
        return ForecastPoint(
            timestamp=timestamp,
            waves=waves,
            swell=swell,
            wind=wind,
            tide=tide,
            weather=weather_info,
        )
    
    async def get_current_conditions(
        self,
        latitude: float,
        longitude: float,
    ) -> Optional[dict[str, Any]]:
        """
        Get current marine conditions.
        
        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            
        Returns:
            Current conditions dictionary.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": ",".join([
                        "wave_height",
                        "wave_direction", 
                        "wave_period",
                        "swell_wave_height",
                        "swell_wave_direction",
                        "swell_wave_period",
                        "sea_surface_temperature",
                    ]),
                }
                
                response = await client.get(self.BASE_URL, params=params)
                
                if response.status_code != 200:
                    return None
                
                data = response.json()
                return data.get("current", {})
                
        except Exception as e:
            self.log_warning(f"Failed to fetch current conditions: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if Open-Meteo API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Simple test request
                params = {
                    "latitude": 0,
                    "longitude": 0,
                    "hourly": "wave_height",
                    "forecast_days": 1,
                }
                response = await client.get(self.BASE_URL, params=params)
                return response.status_code == 200
        except Exception:
            return False


# Convenience function
async def fetch_openmeteo_forecast(
    latitude: float,
    longitude: float,
    spot_name: str = "Unknown",
    days: int = 7,
) -> ForecastResponse:
    """
    Quick helper to fetch Open-Meteo forecast.
    
    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        spot_name: Name of the surf spot.
        days: Number of days to forecast.
        
    Returns:
        ForecastResponse with forecast data.
    """
    client = OpenMeteoClient()
    return await client.get_forecast(latitude, longitude, spot_name, days)

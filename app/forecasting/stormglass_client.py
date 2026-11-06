"""
Stormglass API Client

Fetches real-time surf forecast data from the Stormglass API.
https://stormglass.io/

Free tier: 10 requests/day
API docs: https://docs.stormglass.io/
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from app.core.logger import LoggerMixin, get_logger
from app.forecasting.base_client import ForecastAPIClient
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
from config.settings import get_settings

logger = get_logger(__name__)


class StormglassAPIError(Exception):
    """Custom exception for Stormglass API errors."""
    pass


class StormglassClient(ForecastAPIClient, LoggerMixin):
    """
    Async client for the Stormglass marine weather API.
    
    Fetches wave, wind, tide, and weather data for surf forecasting.
    Implements ForecastAPIClient interface for consistent usage.
    """
    
    BASE_URL = "https://api.stormglass.io/v2"
    
    # Parameters to request from the API
    WAVE_PARAMS = [
        "waveHeight",
        "wavePeriod", 
        "waveDirection",
        "swellHeight",
        "swellPeriod",
        "swellDirection",
        "secondarySwellHeight",
        "secondarySwellPeriod",
        "secondarySwellDirection",
    ]
    
    WIND_PARAMS = [
        "windSpeed",
        "windDirection",
        "gust",
    ]
    
    WEATHER_PARAMS = [
        "airTemperature",
        "waterTemperature",
        "cloudCover",
        "precipitation",
    ]
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        """
        Initialize the Stormglass client.
        
        Args:
            api_key: Stormglass API key. If not provided, reads from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.api_key = api_key or settings.forecast.api_key
        self.timeout = timeout
        
        if not self.api_key:
            self.log_warning(
                "No Stormglass API key configured. "
                "Set FORECAST_API_KEY in .env file."
            )
    
    @property
    def source_name(self) -> str:
        """Return the name of this data source."""
        return "stormglass"
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    @property
    def requires_api_key(self) -> bool:
        """Stormglass requires an API key."""
        return True
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        return {
            "Authorization": self.api_key or "",
            "Content-Type": "application/json",
        }
    
    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        spot_name: str = "Unknown Spot",
        days: int = 3,
    ) -> ForecastResponse:
        """
        Fetch surf forecast for a location.
        
        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            spot_name: Name of the surf spot.
            days: Number of days to forecast (1-10).
            
        Returns:
            ForecastResponse with forecast points.
            
        Raises:
            StormglassAPIError: If API request fails.
        """
        if not self.is_configured:
            raise StormglassAPIError(
                "Stormglass API key not configured. "
                "Set STORMGLASS_API_KEY in your .env file."
            )
        
        # Calculate time range
        start = datetime.utcnow()
        end = start + timedelta(days=days)
        
        # Build parameters
        all_params = self.WAVE_PARAMS + self.WIND_PARAMS + self.WEATHER_PARAMS
        params = {
            "lat": latitude,
            "lng": longitude,
            "params": ",".join(all_params),
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        
        self.log_info(
            f"Fetching forecast for {spot_name}",
            lat=latitude,
            lon=longitude,
            days=days,
        )
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/weather/point",
                    params=params,
                    headers=self._get_headers(),
                )
                
                if response.status_code == 401:
                    raise StormglassAPIError("Invalid API key")
                elif response.status_code == 402:
                    raise StormglassAPIError(
                        "API quota exceeded. Free tier allows 10 requests/day."
                    )
                elif response.status_code == 429:
                    raise StormglassAPIError("Rate limited. Try again later.")
                elif response.status_code != 200:
                    raise StormglassAPIError(
                        f"API error: {response.status_code} - {response.text}"
                    )
                
                data = response.json()
                
        except httpx.TimeoutException:
            raise StormglassAPIError(
                f"Request timed out after {self.timeout}s"
            )
        except httpx.RequestError as e:
            raise StormglassAPIError(f"Network error: {e}")
        
        # Transform to our schema
        forecast_points = self._transform_response(data)
        
        self.log_info(
            f"Received {len(forecast_points)} forecast points",
            spot=spot_name,
        )
        
        return ForecastResponse(
            spot=SpotMetadata(
                name=spot_name,
                coordinates=Coordinates(
                    latitude=latitude,
                    longitude=longitude,
                ),
                timezone="UTC",
            ),
            forecasts=forecast_points,
            source=DataSource.STORMGLASS,
            fetched_at=datetime.utcnow(),
        )
    
    def _transform_response(
        self,
        data: dict[str, Any]
    ) -> list[ForecastPoint]:
        """
        Transform Stormglass API response to our forecast models.
        
        Stormglass returns hourly data with multiple sources.
        We take the 'sg' (Stormglass) values or first available.
        """
        forecast_points = []
        
        hours = data.get("hours", [])
        
        for hour_data in hours:
            try:
                point = self._parse_hour(hour_data)
                if point:
                    forecast_points.append(point)
            except Exception as e:
                self.log_warning(f"Failed to parse hour data: {e}")
                continue
        
        return forecast_points
    
    def _parse_hour(self, hour_data: dict[str, Any]) -> Optional[ForecastPoint]:
        """Parse a single hour of forecast data."""
        timestamp_str = hour_data.get("time")
        if not timestamp_str:
            return None
        
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        
        # Extract values, preferring 'sg' source
        def get_value(key: str) -> Optional[float]:
            if key not in hour_data:
                return None
            val = hour_data[key]
            if isinstance(val, dict):
                return val.get("sg") or val.get("noaa") or list(val.values())[0]
            return val
        
        # Wave data
        wave_height = get_value("waveHeight")
        waves = WaveData(
            height_min=wave_height or 0.0,
            height_max=(wave_height or 0.0) * 1.2,  # Estimate range
        ) if wave_height else WaveData(height_min=0.0, height_max=0.0)
        
        # Swell data
        swell_height = get_value("swellHeight") or get_value("waveHeight") or 0.0
        swell_period = get_value("swellPeriod") or get_value("wavePeriod") or 0.0
        swell_direction = get_value("swellDirection") or get_value("waveDirection") or 0.0
        
        swell = SwellData(
            height=max(0.0, min(swell_height, 20.0)),
            period=max(0.0, min(swell_period, 30.0)),
            direction_degrees=swell_direction % 360,
        )
        
        # Wind data
        wind_speed = get_value("windSpeed") or 0.0
        wind_direction = get_value("windDirection") or 0.0
        wind_gust = get_value("gust")
        
        # Convert m/s to km/h
        wind_speed_kph = wind_speed * 3.6
        wind_gust_kph = wind_gust * 3.6 if wind_gust else None
        
        wind = WindData(
            speed=max(0.0, min(wind_speed_kph, 200.0)),
            direction_degrees=wind_direction % 360,
            gust=wind_gust_kph if wind_gust_kph and wind_gust_kph < 300 else None,
        )
        
        # Tide data (Stormglass doesn't include tide in weather endpoint)
        tide = TideData(height=0.0)
        
        # Weather data
        air_temp = get_value("airTemperature")
        water_temp = get_value("waterTemperature")
        cloud_cover = get_value("cloudCover")
        
        weather_desc = "Clear"
        if cloud_cover:
            if cloud_cover > 80:
                weather_desc = "Overcast"
            elif cloud_cover > 50:
                weather_desc = "Cloudy"
            elif cloud_cover > 20:
                weather_desc = "Partly cloudy"
        
        weather = WeatherData(
            description=weather_desc,
            temperature=air_temp,
            water_temperature=water_temp,
        )
        
        return ForecastPoint(
            timestamp=timestamp,
            source=DataSource.STORMGLASS,
            waves=waves,
            swell=swell,
            wind=wind,
            tide=tide,
            weather=weather,
        )
    
    async def get_tide(
        self,
        lat: float,
        lon: float,
        date: Optional[datetime] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Fetch tide data for a location.
        
        Note: Tide endpoint requires separate API call.
        
        Args:
            lat: Location latitude.
            lon: Location longitude.
            date: Date for tide data (default: today).
            
        Returns:
            Tide data dictionary, or None if failed.
        """
        if not self.is_configured:
            return None
        
        start = date or datetime.utcnow()
        end = start + timedelta(days=1)
        
        params = {
            "lat": lat,
            "lng": lon,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/tide/extremes/point",
                    params=params,
                    headers=self._get_headers(),
                )
                
                if response.status_code != 200:
                    self.log_warning(f"Tide API error: {response.status_code}")
                    return None
                
                data = response.json()
                return {
                    "extremes": data.get("data", []),
                    "lat": lat,
                    "lon": lon,
                    "date": start.isoformat(),
                }
                
        except Exception as e:
            self.log_warning(f"Failed to fetch tide data: {e}")
            return None


# Convenience function for quick forecast fetching
async def fetch_stormglass_forecast(
    latitude: float,
    longitude: float,
    spot_name: str = "Unknown",
    days: int = 3,
) -> ForecastResponse:
    """
    Quick helper to fetch a forecast.
    
    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        spot_name: Name of the surf spot.
        days: Number of days to forecast.
        
    Returns:
        ForecastResponse with forecast data.
    """
    client = StormglassClient()
    return await client.get_forecast(latitude, longitude, spot_name, days)

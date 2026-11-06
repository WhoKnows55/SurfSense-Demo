"""
SurfSense Forecast Data Agent

Deterministic sub-agent that aggregates heterogeneous surf data
(wave forecasts, tide, wind, contextual info) into a unified
representation for the orchestrator.

Wraps existing forecasting clients and contextual providers.
Spot coordinates are resolved from research data injected by the orchestrator.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from app.contextual import (
    AccessibilityProvider,
    ParkingProvider,
    ReviewsProvider,
    SafetyProvider,
)
from app.core.logger import LoggerMixin, get_logger
from app.forecasting.models import Coordinates, ForecastResponse
from app.forecasting.openmeteo_client import OpenMeteoClient
from app.forecasting.stormglass_client import StormglassClient, StormglassAPIError

logger = get_logger(__name__)


class ForecastDataAgent(LoggerMixin):
    """Deterministic sub-agent for data aggregation.

    Exposes tools for the orchestrator to:
    - fetch_forecast: Get unified forecast data for a spot
    - fetch_contextual_info: Get parking, accessibility, reviews, safety
    - get_spot_metadata: Get static spot information
    """

    def __init__(self, settings):
        self._settings = settings
        self._forecast_cache: dict[str, dict] = {}
        self._cache_ttl = timedelta(hours=1)

        # Forecast API clients
        self._openmeteo_client = OpenMeteoClient()
        self._stormglass_client = StormglassClient()

        # Contextual providers
        self._parking = ParkingProvider()
        self._accessibility = AccessibilityProvider()
        self._reviews = ReviewsProvider()
        self._safety = SafetyProvider()

        # Research data injected by the orchestrator (spot_name -> research dict)
        self._research_data: dict[str, dict] = {}

    def set_research_data(self, spot_name: str, data: dict) -> None:
        """Inject researched spot data from the orchestrator.

        Called by the orchestrator after research_spot returns,
        so that fetch_forecast can resolve coordinates dynamically.
        """
        self._research_data[spot_name.lower().strip()] = data

    def _get_spot_coordinates(self, spot_name: str) -> Optional[Coordinates]:
        """Look up coordinates from researched data or orchestrator session."""
        normalized = spot_name.lower().strip()

        # Check research data injected by orchestrator
        for key, data in self._research_data.items():
            if key == normalized or data.get("name", "").lower().strip() == normalized:
                lat = data.get("latitude")
                lon = data.get("longitude")
                if lat is not None and lon is not None:
                    return Coordinates(latitude=lat, longitude=lon)

        return None

    def _get_cached_forecast(self, spot_name: str) -> Optional[dict]:
        """Return cached forecast if still valid."""
        key = spot_name.lower().strip()
        if key not in self._forecast_cache:
            return None
        entry = self._forecast_cache[key]
        age = datetime.utcnow() - entry.get("_cached_at", datetime.min)
        if age > self._cache_ttl:
            del self._forecast_cache[key]
            return None
        return entry

    def _cache_forecast(self, spot_name: str, data: dict) -> None:
        """Cache a forecast result."""
        key = spot_name.lower().strip()
        data["_cached_at"] = datetime.utcnow()
        self._forecast_cache[key] = data

    def _forecast_to_dict(self, forecast: ForecastResponse) -> dict:
        """Convert ForecastResponse to a dict the orchestrator can consume."""
        return {
            "spot": forecast.spot.name,
            "coordinates": {
                "lat": forecast.spot.coordinates.latitude,
                "lon": forecast.spot.coordinates.longitude,
            },
            "source": forecast.source.value,
            "fetched_at": forecast.fetched_at.isoformat(),
            "forecast_count": len(forecast.forecasts),
            "forecasts": [
                {
                    "timestamp": f.timestamp.isoformat(),
                    "summary": f.summary(),
                    "waves": {
                        "min_m": f.waves.height_min if f.waves else None,
                        "max_m": f.waves.height_max if f.waves else None,
                        "avg_m": f.waves.height_avg if f.waves else None,
                    },
                    "swell": {
                        "height_m": f.swell.height if f.swell else None,
                        "period_s": f.swell.period if f.swell else None,
                        "direction": (
                            f.swell.direction.value
                            if f.swell and f.swell.direction
                            else None
                        ),
                        "direction_deg": f.swell.direction_degrees if f.swell else None,
                    },
                    "wind": {
                        "speed_kph": f.wind.speed if f.wind else 0,
                        "direction": (
                            f.wind.direction.value
                            if f.wind and f.wind.direction
                            else None
                        ),
                        "direction_deg": f.wind.direction_degrees if f.wind else None,
                        "is_offshore": f.is_offshore_wind,
                        "is_light": f.is_light_wind,
                    },
                }
                for f in forecast.forecasts
            ],
        }

    async def fetch_forecast(
        self, spot_name: str, days: int = 3, snapshot_path: Optional[str] = None
    ) -> dict:
        """Fetch and unify forecast data for a spot.

        Args:
            spot_name: Name of the surf spot.
            days: Number of forecast days (1-7).
            snapshot_path: Optional path for deterministic scenario replay.
                If the file exists, read from it instead of calling APIs.
                If the file does not exist, write the result to it after fetching.

        Returns:
            Unified forecast dict with hourly data.
        """
        if snapshot_path:
            snap = Path(snapshot_path)
            if snap.exists():
                with open(snap) as f:
                    return json.load(f)

        cached = self._get_cached_forecast(spot_name)
        if cached:
            if snapshot_path:
                Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
                with open(snapshot_path, "w") as f:
                    json.dump(cached, f, default=str, indent=2)
            return cached

        coordinates = self._get_spot_coordinates(spot_name)
        if not coordinates:
            return {
                "error": (
                    f"Unknown spot: '{spot_name}'. "
                    "Call research_spot first to look up this location."
                ),
                "spot": spot_name,
            }

        lat = coordinates.latitude
        lon = coordinates.longitude

        def _save_and_return(data: dict) -> dict:
            self._cache_forecast(spot_name, data)
            if snapshot_path:
                Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
                with open(snapshot_path, "w") as f:
                    json.dump(data, f, default=str, indent=2)
            return data

        # Open-Meteo first: free, no API key required (thesis default per Section 3.3.3)
        try:
            result = await self._openmeteo_client.get_forecast(
                latitude=lat, longitude=lon, spot_name=spot_name, days=days
            )
            return _save_and_return(self._forecast_to_dict(result))
        except Exception as e:
            self.log_warning(f"Open-Meteo failed for {spot_name}: {e}")

        # Fallback: Stormglass (requires API key)
        if self._stormglass_client.is_configured:
            try:
                result = await self._stormglass_client.get_forecast(
                    latitude=lat, longitude=lon, spot_name=spot_name, days=days
                )
                return _save_and_return(self._forecast_to_dict(result))
            except StormglassAPIError as e:
                return {
                    "error": str(e),
                    "spot": spot_name,
                    "message": "Unable to fetch forecast data",
                }

        return {
            "error": "All forecast providers failed or are unavailable",
            "spot": spot_name,
            "message": "Unable to fetch forecast data",
        }

    async def fetch_contextual_info(self, spot_name: str) -> dict:
        """Aggregate contextual data from all providers.

        Args:
            spot_name: Name of the surf spot.

        Returns:
            Dict with parking, accessibility, reviews, and safety info.
        """
        parking = await self._parking.get_data(spot_name)
        access = await self._accessibility.get_data(spot_name)
        reviews = await self._reviews.get_data(spot_name)
        safety = await self._safety.get_data(spot_name)
        return {
            "spot_name": spot_name,
            "parking": parking.model_dump(),
            "accessibility": access.model_dump(),
            "reviews": reviews.model_dump(),
            "safety": safety.model_dump(),
        }

    async def get_spot_metadata(self, spot_name: str) -> dict:
        """Return metadata for a surf spot from researched data.

        Args:
            spot_name: Name of the surf spot.

        Returns:
            Dict with coordinates, timezone, break type, etc.
        """
        normalized = spot_name.lower().strip()

        # Check research data
        for key, data in self._research_data.items():
            if key == normalized or data.get("name", "").lower().strip() == normalized:
                return {
                    "name": data.get("name", spot_name),
                    "coordinates": {
                        "lat": data.get("latitude"),
                        "lon": data.get("longitude"),
                    },
                    "timezone": data.get("timezone", "UTC"),
                    "break_type": data.get("break_type", "unknown"),
                    "region": data.get("region"),
                    "country": data.get("country"),
                    "skill_levels": {
                        "minimum": data.get("skill_minimum", "intermediate"),
                        "recommended": data.get("skill_recommended", "intermediate"),
                    },
                    "hazards": data.get("hazards", []),
                    "description": data.get("description", ""),
                }

        # Fallback: check coordinates only
        coords = self._get_spot_coordinates(spot_name)
        if coords:
            return {
                "name": spot_name,
                "coordinates": {
                    "lat": coords.latitude,
                    "lon": coords.longitude,
                },
                "timezone": "UTC",
                "break_type": "unknown",
            }

        return {
            "error": (
                f"Spot '{spot_name}' not found. "
                "Call research_spot first to gather information about this location."
            )
        }

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return OpenAI function-calling schemas for all tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_forecast",
                    "description": (
                        "Fetch surf forecast data (waves, swell, wind, tide) "
                        "for a spot over N days. Returns hourly data normalised "
                        "to a unified schema."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_name": {
                                "type": "string",
                                "description": "Name of the surf spot",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days to forecast (1-7)",
                                "default": 3,
                            },
                        },
                        "required": ["spot_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_contextual_info",
                    "description": (
                        "Get parking, accessibility, reviews, and safety "
                        "information for a surf spot."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_name": {
                                "type": "string",
                                "description": "Name of the surf spot",
                            },
                        },
                        "required": ["spot_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_spot_metadata",
                    "description": (
                        "Get static metadata for a surf spot: coordinates, "
                        "timezone, break type."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_name": {
                                "type": "string",
                                "description": "Name of the surf spot",
                            },
                        },
                        "required": ["spot_name"],
                    },
                },
            },
        ]

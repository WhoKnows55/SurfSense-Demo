"""
Forecasting layer for SurfSense.

This module handles:
- External forecast API integration (Stormglass, Open-Meteo)
- Unified forecast data schema
- Forecast service orchestration
"""

from app.forecasting.models import (
    Coordinates,
    DataSource,
    ForecastPoint,
    ForecastResponse,
    SpotMetadata,
    SwellData,
    SwellDirection,
    TideData,
    TideState,
    WaveData,
    WeatherData,
    WindData,
    WindDirection,
)
from app.forecasting.stormglass_client import (
    StormglassClient,
    StormglassAPIError,
    fetch_stormglass_forecast,
)
from app.forecasting.openmeteo_client import (
    OpenMeteoClient,
    fetch_openmeteo_forecast,
)

__all__ = [
    # Core forecast models
    "ForecastPoint",
    "ForecastResponse",
    # Component models
    "WaveData",
    "SwellData",
    "WindData",
    "TideData",
    "WeatherData",
    # Metadata
    "SpotMetadata",
    "Coordinates",
    "DataSource",
    # Enums
    "WindDirection",
    "SwellDirection",
    "TideState",
    # API Clients
    "StormglassClient",
    "StormglassAPIError",
    "fetch_stormglass_forecast",
    "OpenMeteoClient",
    "fetch_openmeteo_forecast",
]

"""
Base interface for forecast API clients.

All forecast providers (Stormglass, NOAA, etc.) implement this interface
to ensure consistent behavior across the system.
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from .models import ForecastResponse, Coordinates


class ForecastAPIClient(ABC):
    """
    Abstract base class for forecast API clients.
    
    All forecast data providers must implement this interface to be
    compatible with the ForecastIntegrationAgent.
    """
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        pass
    
    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this client is properly configured (e.g., has API key)."""
        pass
    
    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Return True if this data source requires an API key."""
        pass
    
    @abstractmethod
    async def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 5
    ) -> Optional[ForecastResponse]:
        """
        Fetch forecast data for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location  
            days: Number of days to forecast (default: 5)
            
        Returns:
            ForecastResponse if successful, None if failed
        """
        pass
    
    @abstractmethod
    async def get_tide(
        self,
        lat: float,
        lon: float,
        date: Optional[datetime] = None
    ) -> Optional[dict]:
        """
        Fetch tide data for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            date: Date for tide data (default: today)
            
        Returns:
            Tide data dictionary if successful, None if failed
        """
        pass
    
    def supports_location(self, lat: float, lon: float) -> bool:
        """
        Check if this data source supports the given location.
        
        Some providers (like NOAA) only cover certain regions.
        Default implementation returns True (global coverage).
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if location is supported
        """
        return True
    
    async def health_check(self) -> bool:
        """
        Check if the API is accessible and working.
        
        Returns:
            True if API is healthy
        """
        return self.is_configured


class ForecastClientRegistry:
    """
    Registry for managing multiple forecast API clients.
    
    Provides source selection and fallback logic.
    """
    
    def __init__(self):
        self._clients: dict[str, ForecastAPIClient] = {}
        self._priority_order: list[str] = []
    
    def register(self, client: ForecastAPIClient, priority: int = 100) -> None:
        """
        Register a forecast client.
        
        Args:
            client: The forecast client to register
            priority: Lower number = higher priority (default: 100)
        """
        name = client.source_name
        self._clients[name] = client
        
        # Insert maintaining priority order
        inserted = False
        for i, existing_name in enumerate(self._priority_order):
            existing_client = self._clients[existing_name]
            # This is simplified - in practice you'd store priority separately
            if not inserted:
                self._priority_order.insert(i, name)
                inserted = True
                break
        
        if not inserted:
            self._priority_order.append(name)
    
    def get_client(self, name: str) -> Optional[ForecastAPIClient]:
        """Get a specific client by name."""
        return self._clients.get(name)
    
    def get_available_clients(self) -> list[ForecastAPIClient]:
        """Get all configured and available clients."""
        return [
            client for client in self._clients.values()
            if client.is_configured
        ]
    
    def get_best_client_for_location(
        self,
        lat: float,
        lon: float
    ) -> Optional[ForecastAPIClient]:
        """
        Get the best available client for a specific location.
        
        Considers:
        1. Whether client supports the location
        2. Whether client is configured
        3. Priority order
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Best available client, or None if no suitable client
        """
        for name in self._priority_order:
            client = self._clients[name]
            if client.is_configured and client.supports_location(lat, lon):
                return client
        return None
    
    @property
    def registered_sources(self) -> list[str]:
        """List all registered source names."""
        return list(self._clients.keys())


# Global registry instance
forecast_registry = ForecastClientRegistry()

"""
SurfSense Parking Data Provider

Provides parking information for surf spots.
"""

from typing import Optional

from app.contextual.base import (
    ContextualDataProvider,
    DataConfidence,
    ParkingInfo,
    ParkingType,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


class ParkingProvider(ContextualDataProvider):
    """
    Provider for surf spot parking information.
    
    Data sources (to be integrated):
    - Google Maps API
    - OpenStreetMap
    - User-contributed data
    - Local surf spot databases
    """
    
    def __init__(self):
        """Initialize the parking provider."""
        # In-memory cache/database for development
        self._parking_data: dict[str, ParkingInfo] = {}
        self._load_default_data()
    
    def _load_default_data(self) -> None:
        """Load default parking data for known spots."""
        # Sample data for development
        self._parking_data = {
            "Pipeline": ParkingInfo(
                parking_type=ParkingType.FREE_LOT,
                capacity=50,
                distance_to_beach_m=100,
                notes="Gets crowded early. Arrive before 6am for best spots.",
                confidence=DataConfidence.MEDIUM
            ),
            "Sunset Beach": ParkingInfo(
                parking_type=ParkingType.STREET,
                capacity=None,
                distance_to_beach_m=50,
                notes="Limited street parking along Kamehameha Highway.",
                confidence=DataConfidence.MEDIUM
            ),
            "Waikiki": ParkingInfo(
                parking_type=ParkingType.PAID_LOT,
                capacity=200,
                cost_per_hour=5.0,
                max_stay_hours=4,
                distance_to_beach_m=200,
                notes="Multiple paid lots available. Metered street parking also available.",
                confidence=DataConfidence.HIGH
            ),
        }
    
    async def get_data(self, spot_name: str) -> ParkingInfo:
        """
        Get parking information for a spot.
        
        Args:
            spot_name: Name of the surf spot.
            
        Returns:
            ParkingInfo for the spot.
        """
        self.log_debug(f"Fetching parking data for: {spot_name}")
        
        # Check local data first
        if spot_name in self._parking_data:
            return self._parking_data[spot_name]
        
        # TODO: Query external APIs (Google Maps, OSM, etc.)
        
        # Return unknown data if not found
        return ParkingInfo(
            parking_type=ParkingType.NONE,
            notes="Parking information not available for this spot.",
            confidence=DataConfidence.UNKNOWN
        )
    
    def get_data_type(self) -> str:
        """Return the data type identifier."""
        return "parking"
    
    def add_parking_data(
        self,
        spot_name: str,
        parking_info: ParkingInfo
    ) -> None:
        """
        Add or update parking data for a spot.
        
        Args:
            spot_name: Name of the surf spot.
            parking_info: Parking information to store.
        """
        self._parking_data[spot_name] = parking_info
        self.log_info(f"Updated parking data for: {spot_name}")

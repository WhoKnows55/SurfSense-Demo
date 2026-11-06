"""
SurfSense Accessibility Data Provider

Provides accessibility information for surf spots.
"""

from app.contextual.base import (
    AccessibilityInfo,
    AccessibilityLevel,
    ContextualDataProvider,
    DataConfidence,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


class AccessibilityProvider(ContextualDataProvider):
    """
    Provider for surf spot accessibility information.
    
    Data sources (to be integrated):
    - AccessSurf database
    - User-contributed data
    - Local disability services
    - Beach accessibility programs
    """
    
    def __init__(self):
        """Initialize the accessibility provider."""
        self._accessibility_data: dict[str, AccessibilityInfo] = {}
        self._load_default_data()
    
    def _load_default_data(self) -> None:
        """Load default accessibility data for known spots."""
        self._accessibility_data = {
            "Waikiki": AccessibilityInfo(
                level=AccessibilityLevel.FULLY_ACCESSIBLE,
                wheelchair_access=True,
                beach_wheelchair_available=True,
                paved_path=True,
                accessible_facilities=True,
                notes="Beach wheelchairs available at Waikiki Beach Center. "
                      "Accessible showers and restrooms nearby.",
                confidence=DataConfidence.HIGH
            ),
            "Pipeline": AccessibilityInfo(
                level=AccessibilityLevel.LIMITED,
                wheelchair_access=False,
                beach_wheelchair_available=False,
                paved_path=False,
                accessible_facilities=False,
                notes="Sand access only. Rocky entry in some areas.",
                confidence=DataConfidence.MEDIUM
            ),
            "San Onofre": AccessibilityInfo(
                level=AccessibilityLevel.PARTIALLY_ACCESSIBLE,
                wheelchair_access=True,
                beach_wheelchair_available=True,
                paved_path=True,
                accessible_facilities=True,
                notes="Beach wheelchairs available through state park. "
                      "Accessible trail to Old Man's area.",
                confidence=DataConfidence.HIGH
            ),
        }
    
    async def get_data(self, spot_name: str) -> AccessibilityInfo:
        """
        Get accessibility information for a spot.
        
        Args:
            spot_name: Name of the surf spot.
            
        Returns:
            AccessibilityInfo for the spot.
        """
        self.log_debug(f"Fetching accessibility data for: {spot_name}")
        
        if spot_name in self._accessibility_data:
            return self._accessibility_data[spot_name]
        
        # TODO: Query external APIs or databases
        
        return AccessibilityInfo(
            level=AccessibilityLevel.UNKNOWN,
            notes="Accessibility information not available for this spot.",
            confidence=DataConfidence.UNKNOWN
        )
    
    def get_data_type(self) -> str:
        """Return the data type identifier."""
        return "accessibility"
    
    def add_accessibility_data(
        self,
        spot_name: str,
        accessibility_info: AccessibilityInfo
    ) -> None:
        """Add or update accessibility data for a spot."""
        self._accessibility_data[spot_name] = accessibility_info
        self.log_info(f"Updated accessibility data for: {spot_name}")

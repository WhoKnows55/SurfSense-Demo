"""
SurfSense Safety Data Provider

Provides safety information for surf spots.
"""

from app.contextual.base import (
    ContextualDataProvider,
    DataConfidence,
    HazardType,
    SafetyInfo,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


class SafetyProvider(ContextualDataProvider):
    """
    Provider for surf spot safety information.
    
    Data sources (to be integrated):
    - Lifeguard reports
    - Beach safety databases
    - Shark tracking services
    - Water quality reports
    - User reports
    """
    
    def __init__(self):
        """Initialize the safety provider."""
        self._safety_data: dict[str, SafetyInfo] = {}
        self._load_default_data()
    
    def _load_default_data(self) -> None:
        """Load default safety data for known spots."""
        self._safety_data = {
            "Pipeline": SafetyInfo(
                hazards=[
                    HazardType.REEF,
                    HazardType.CURRENTS,
                    HazardType.CROWDS,
                    HazardType.LOCALISM
                ],
                lifeguard_on_duty=True,
                lifeguard_hours="7am - 7pm daily",
                emergency_access=True,
                recommended_skill_level="advanced",
                warnings=[
                    "Shallow reef - serious injury risk",
                    "Strong rip currents during big swells",
                    "Respect the locals and lineup"
                ],
                notes="One of the most dangerous waves in the world. "
                      "Only experienced surfers should attempt.",
                confidence=DataConfidence.HIGH
            ),
            "Waikiki": SafetyInfo(
                hazards=[
                    HazardType.CROWDS,
                    HazardType.JELLYFISH
                ],
                lifeguard_on_duty=True,
                lifeguard_hours="6am - 8pm daily",
                emergency_access=True,
                recommended_skill_level="beginner",
                warnings=[
                    "Watch for box jellyfish 8-10 days after full moon",
                    "Stay aware of other surfers and swimmers"
                ],
                notes="Generally safe beach with consistent lifeguard presence.",
                confidence=DataConfidence.HIGH
            ),
            "Mavericks": SafetyInfo(
                hazards=[
                    HazardType.ROCKS,
                    HazardType.CURRENTS,
                    HazardType.SHARKS
                ],
                lifeguard_on_duty=False,
                emergency_access=False,
                recommended_skill_level="expert",
                warnings=[
                    "No lifeguards - surf at your own risk",
                    "Great white shark territory",
                    "Cold water - wetsuit required",
                    "Difficult paddle out",
                    "Remote location - limited emergency access"
                ],
                notes="Big wave spot. Only for expert big wave surfers with proper safety equipment.",
                confidence=DataConfidence.HIGH
            ),
            "Huntington Beach": SafetyInfo(
                hazards=[
                    HazardType.CROWDS,
                    HazardType.CURRENTS
                ],
                lifeguard_on_duty=True,
                lifeguard_hours="6am - sunset daily (summer), reduced hours in winter",
                emergency_access=True,
                recommended_skill_level="beginner",
                warnings=[
                    "Can get very crowded, especially near pier",
                    "Rip currents possible during south swells"
                ],
                notes="Good beach for all levels with strong lifeguard presence.",
                confidence=DataConfidence.HIGH
            ),
        }
    
    async def get_data(self, spot_name: str) -> SafetyInfo:
        """
        Get safety information for a spot.
        
        Args:
            spot_name: Name of the surf spot.
            
        Returns:
            SafetyInfo for the spot.
        """
        self.log_debug(f"Fetching safety data for: {spot_name}")
        
        if spot_name in self._safety_data:
            return self._safety_data[spot_name]
        
        # TODO: Query external safety APIs
        
        return SafetyInfo(
            notes="Safety information not available for this spot. "
                  "Exercise caution and check local conditions.",
            confidence=DataConfidence.UNKNOWN
        )
    
    def get_data_type(self) -> str:
        """Return the data type identifier."""
        return "safety"
    
    def add_safety_data(
        self,
        spot_name: str,
        safety_info: SafetyInfo
    ) -> None:
        """Add or update safety data for a spot."""
        self._safety_data[spot_name] = safety_info
        self.log_info(f"Updated safety data for: {spot_name}")
    
    def add_warning(self, spot_name: str, warning: str) -> None:
        """Add a temporary warning to a spot."""
        if spot_name in self._safety_data:
            self._safety_data[spot_name].warnings.append(warning)
            self.log_info(f"Added warning for {spot_name}: {warning}")

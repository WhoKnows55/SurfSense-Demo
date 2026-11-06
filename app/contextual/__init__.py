"""
SurfSense Contextual Layer (Layer 2)

Gathers and integrates auxiliary data to enrich recommendations:
- Parking information
- Accessibility details
- User reviews and ratings
- Safety information

This data augments the forecast data to provide comprehensive
surf spot recommendations.
"""

from app.contextual.base import ContextualDataProvider, SpotContext
from app.contextual.parking import ParkingProvider
from app.contextual.accessibility import AccessibilityProvider
from app.contextual.reviews import ReviewsProvider
from app.contextual.safety import SafetyProvider

__all__ = [
    "ContextualDataProvider",
    "SpotContext",
    "ParkingProvider",
    "AccessibilityProvider",
    "ReviewsProvider",
    "SafetyProvider",
]

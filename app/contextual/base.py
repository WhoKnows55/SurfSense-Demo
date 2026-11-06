"""
SurfSense Contextual Layer Base

Base classes and models for contextual data providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.logger import LoggerMixin, get_logger

logger = get_logger(__name__)


class DataConfidence(str, Enum):
    """Confidence level in the data accuracy."""
    HIGH = "high"        # Recent, verified data
    MEDIUM = "medium"    # Somewhat recent or user-reported
    LOW = "low"          # Old data or unverified
    UNKNOWN = "unknown"  # No confidence information


class ParkingType(str, Enum):
    """Types of parking available."""
    FREE_LOT = "free_lot"
    PAID_LOT = "paid_lot"
    STREET = "street"
    METERED = "metered"
    PERMIT_REQUIRED = "permit"
    NONE = "none"


class AccessibilityLevel(str, Enum):
    """Accessibility rating for the spot."""
    FULLY_ACCESSIBLE = "fully_accessible"
    PARTIALLY_ACCESSIBLE = "partially_accessible"
    LIMITED = "limited"
    NOT_ACCESSIBLE = "not_accessible"
    UNKNOWN = "unknown"


class HazardType(str, Enum):
    """Types of hazards at a surf spot."""
    ROCKS = "rocks"
    REEF = "reef"
    CURRENTS = "currents"
    SHARKS = "sharks"
    JELLYFISH = "jellyfish"
    LOCALISM = "localism"
    CROWDS = "crowds"
    POLLUTION = "pollution"
    ACCESS = "access"


class ParkingInfo(BaseModel):
    """Parking information for a surf spot."""
    
    parking_type: ParkingType = Field(
        default=ParkingType.NONE,
        description="Type of parking available"
    )
    capacity: Optional[int] = Field(
        default=None,
        ge=0,
        description="Approximate number of spots"
    )
    cost_per_hour: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cost per hour in local currency"
    )
    max_stay_hours: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum parking duration"
    )
    distance_to_beach_m: Optional[int] = Field(
        default=None,
        ge=0,
        description="Walking distance to beach in meters"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional parking notes"
    )
    confidence: DataConfidence = Field(
        default=DataConfidence.UNKNOWN
    )


class AccessibilityInfo(BaseModel):
    """Accessibility information for a surf spot."""
    
    level: AccessibilityLevel = Field(
        default=AccessibilityLevel.UNKNOWN,
        description="Overall accessibility rating"
    )
    wheelchair_access: bool = Field(
        default=False,
        description="Wheelchair accessible to beach"
    )
    beach_wheelchair_available: bool = Field(
        default=False,
        description="Beach wheelchairs available for rent"
    )
    paved_path: bool = Field(
        default=False,
        description="Paved path to beach"
    )
    accessible_facilities: bool = Field(
        default=False,
        description="Accessible restrooms/showers"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional accessibility notes"
    )
    confidence: DataConfidence = Field(
        default=DataConfidence.UNKNOWN
    )


class ReviewSummary(BaseModel):
    """Summary of user reviews for a surf spot."""
    
    average_rating: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Average rating (1-5 stars)"
    )
    total_reviews: int = Field(
        default=0,
        ge=0,
        description="Total number of reviews"
    )
    recent_reviews: int = Field(
        default=0,
        ge=0,
        description="Reviews from last 6 months"
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Common positive themes"
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Common concerns mentioned"
    )
    best_for: list[str] = Field(
        default_factory=list,
        description="Skill levels this spot is best for"
    )
    confidence: DataConfidence = Field(
        default=DataConfidence.UNKNOWN
    )


class SafetyInfo(BaseModel):
    """Safety information for a surf spot."""
    
    hazards: list[HazardType] = Field(
        default_factory=list,
        description="Known hazards at this spot"
    )
    lifeguard_on_duty: Optional[bool] = Field(
        default=None,
        description="Whether lifeguards are typically present"
    )
    lifeguard_hours: Optional[str] = Field(
        default=None,
        description="Lifeguard duty hours"
    )
    emergency_access: bool = Field(
        default=True,
        description="Emergency vehicle access available"
    )
    recommended_skill_level: Optional[str] = Field(
        default=None,
        description="Minimum recommended skill level"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Current safety warnings"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional safety notes"
    )
    confidence: DataConfidence = Field(
        default=DataConfidence.UNKNOWN
    )


class SpotContext(BaseModel):
    """
    Complete contextual information for a surf spot.
    
    This combines all auxiliary data sources into a single model
    that can be used alongside forecast data.
    """
    
    spot_name: str = Field(
        ...,
        description="Name of the surf spot"
    )
    parking: ParkingInfo = Field(
        default_factory=ParkingInfo,
        description="Parking information"
    )
    accessibility: AccessibilityInfo = Field(
        default_factory=AccessibilityInfo,
        description="Accessibility information"
    )
    reviews: ReviewSummary = Field(
        default_factory=ReviewSummary,
        description="Review summary"
    )
    safety: SafetyInfo = Field(
        default_factory=SafetyInfo,
        description="Safety information"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this context was last updated"
    )
    
    def summary(self) -> str:
        """Generate a human-readable summary of the context."""
        parts = [f"Context for {self.spot_name}:"]
        
        # Parking
        parts.append(f"  Parking: {self.parking.parking_type.value}")
        if self.parking.cost_per_hour:
            parts.append(f"    Cost: ${self.parking.cost_per_hour}/hr")
        
        # Accessibility
        parts.append(f"  Accessibility: {self.accessibility.level.value}")
        
        # Reviews
        if self.reviews.average_rating:
            parts.append(
                f"  Rating: {self.reviews.average_rating:.1f}/5 "
                f"({self.reviews.total_reviews} reviews)"
            )
        
        # Safety
        if self.safety.hazards:
            hazard_str = ", ".join(h.value for h in self.safety.hazards)
            parts.append(f"  Hazards: {hazard_str}")
        
        return "\n".join(parts)


class ContextualDataProvider(ABC, LoggerMixin):
    """
    Abstract base class for contextual data providers.
    
    Each provider fetches a specific type of auxiliary data
    (parking, accessibility, reviews, safety).
    """
    
    @abstractmethod
    async def get_data(self, spot_name: str) -> BaseModel:
        """
        Fetch contextual data for a spot.
        
        Args:
            spot_name: Name of the surf spot.
            
        Returns:
            The relevant data model.
        """
        pass
    
    @abstractmethod
    def get_data_type(self) -> str:
        """Return the type of data this provider supplies."""
        pass

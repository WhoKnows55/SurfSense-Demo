"""
SurfSense Reviews Data Provider

Provides aggregated review data for surf spots.
"""

from app.contextual.base import (
    ContextualDataProvider,
    DataConfidence,
    ReviewSummary,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


class ReviewsProvider(ContextualDataProvider):
    """
    Provider for surf spot review summaries.
    
    Data sources (to be integrated):
    - Surfline reviews
    - Google Maps reviews
    - Yelp reviews
    - Dedicated surf review sites
    - User-contributed reviews
    """
    
    def __init__(self):
        """Initialize the reviews provider."""
        self._review_data: dict[str, ReviewSummary] = {}
        self._load_default_data()
    
    def _load_default_data(self) -> None:
        """Load default review data for known spots."""
        self._review_data = {
            "Pipeline": ReviewSummary(
                average_rating=4.8,
                total_reviews=2500,
                recent_reviews=150,
                highlights=[
                    "World-class waves",
                    "Perfect barrels",
                    "Iconic surf spot"
                ],
                concerns=[
                    "Crowded during competition season",
                    "Strong currents",
                    "Advanced skill required"
                ],
                best_for=["advanced", "expert"],
                confidence=DataConfidence.HIGH
            ),
            "Waikiki": ReviewSummary(
                average_rating=4.5,
                total_reviews=5000,
                recent_reviews=800,
                highlights=[
                    "Perfect for beginners",
                    "Consistent small waves",
                    "Many surf schools",
                    "Beautiful setting"
                ],
                concerns=[
                    "Very crowded",
                    "Tourist-heavy",
                    "Expensive board rentals"
                ],
                best_for=["beginner", "intermediate"],
                confidence=DataConfidence.HIGH
            ),
            "Mavericks": ReviewSummary(
                average_rating=4.9,
                total_reviews=500,
                recent_reviews=30,
                highlights=[
                    "Legendary big wave spot",
                    "Incredible power",
                    "Beautiful Northern California coast"
                ],
                concerns=[
                    "Extremely dangerous",
                    "Cold water",
                    "Difficult access",
                    "Sharks present"
                ],
                best_for=["expert"],
                confidence=DataConfidence.MEDIUM
            ),
        }
    
    async def get_data(self, spot_name: str) -> ReviewSummary:
        """
        Get review summary for a spot.
        
        Args:
            spot_name: Name of the surf spot.
            
        Returns:
            ReviewSummary for the spot.
        """
        self.log_debug(f"Fetching review data for: {spot_name}")
        
        if spot_name in self._review_data:
            return self._review_data[spot_name]
        
        # TODO: Query external review APIs
        
        return ReviewSummary(
            notes="Review data not available for this spot.",
            confidence=DataConfidence.UNKNOWN
        )
    
    def get_data_type(self) -> str:
        """Return the data type identifier."""
        return "reviews"
    
    def add_review_data(
        self,
        spot_name: str,
        review_summary: ReviewSummary
    ) -> None:
        """Add or update review data for a spot."""
        self._review_data[spot_name] = review_summary
        self.log_info(f"Updated review data for: {spot_name}")

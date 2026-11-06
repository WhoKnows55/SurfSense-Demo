"""
SurfSense Travel Utilities

Functions for calculating travel times and distances between locations.
"""

import math
from typing import Optional, Tuple

from app.core.logger import LoggerMixin, get_logger

logger = get_logger(__name__)


# Known location coordinates for common origins
KNOWN_LOCATIONS = {
    # California
    "los angeles": (34.0522, -118.2437),
    "la": (34.0522, -118.2437),
    "san francisco": (37.7749, -122.4194),
    "sf": (37.7749, -122.4194),
    "san diego": (32.7157, -117.1611),
    "san jose": (37.3382, -121.8863),
    "sacramento": (38.5816, -121.4944),
    "oakland": (37.8044, -122.2712),
    "santa cruz": (36.9741, -122.0308),
    "santa barbara": (34.4208, -119.6982),
    "ventura": (34.2805, -119.2945),
    "long beach": (33.7701, -118.1937),
    "orange county": (33.7175, -117.8311),
    "irvine": (33.6846, -117.8265),
    
    # Hawaii
    "honolulu": (21.3069, -157.8583),
    "waikiki": (21.2793, -157.8292),
    
    # Other US
    "phoenix": (33.4484, -112.0740),
    "las vegas": (36.1699, -115.1398),
    "seattle": (47.6062, -122.3321),
    "portland": (45.5152, -122.6784),
    
    # Australia
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "gold coast": (-28.0167, 153.4000),
    "brisbane": (-27.4698, 153.0251),
    
    # Europe
    "lisbon": (38.7223, -9.1393),
    "paris": (48.8566, 2.3522),
    "bordeaux": (44.8378, -0.5792),
    "biarritz": (43.4832, -1.5586),
    
    # Surf spots (also useful as destinations)
    "pipeline": (21.665, -158.0539),
    "north shore": (21.665, -158.0539),
    "sunset beach": (21.6789, -158.0417),
    "mavericks": (37.495, -122.4967),
    "huntington beach": (33.6553, -117.9992),
    "san onofre": (33.3753, -117.5689),
    "trestles": (33.3817, -117.5886),
    "rincon": (34.3742, -119.4761),
    "blacks beach": (32.8894, -117.2531),
    "la jolla": (32.8328, -117.2713),
    "ocean beach": (32.7479, -117.2505),
}


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Uses the Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).
        
    Returns:
        Distance in kilometers.
    """
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def get_location_coordinates(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for a known location.
    
    Args:
        location_name: Name of the location.
        
    Returns:
        Tuple of (latitude, longitude) or None if unknown.
    """
    normalized = location_name.lower().strip()
    return KNOWN_LOCATIONS.get(normalized)


def estimate_travel_time(
    origin: str,
    destination: str,
    is_driving: bool = True
) -> Optional[float]:
    """
    Estimate travel time between two locations.
    
    Args:
        origin: Origin location name or coordinates.
        destination: Destination location name or coordinates.
        is_driving: True for car, False for public transport.
        
    Returns:
        Estimated travel time in hours, or None if can't calculate.
    """
    # Get coordinates
    origin_coords = get_location_coordinates(origin)
    dest_coords = get_location_coordinates(destination)
    
    if origin_coords is None or dest_coords is None:
        return None
    
    # Calculate distance
    distance_km = haversine_distance(
        origin_coords[0], origin_coords[1],
        dest_coords[0], dest_coords[1]
    )
    
    # Estimate time based on transport mode
    if is_driving:
        # Average highway speed: 80 km/h (accounting for traffic)
        avg_speed = 80.0
        # Add 20% for actual road distance vs straight line
        road_factor = 1.2
    else:
        # Public transport is generally slower
        avg_speed = 40.0
        # More indirect routes
        road_factor = 1.5
    
    actual_distance = distance_km * road_factor
    travel_hours = actual_distance / avg_speed
    
    # Round to nearest 0.5 hour
    return round(travel_hours * 2) / 2


def format_travel_time(hours: float) -> str:
    """
    Format travel time as human-readable string.
    
    Args:
        hours: Travel time in hours.
        
    Returns:
        Formatted string like "2h 30min" or "45 min".
    """
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} min"
    
    h = int(hours)
    m = int((hours - h) * 60)
    
    if m == 0:
        return f"{h}h"
    return f"{h}h {m}min"


def get_departure_time(
    arrival_time: str,
    travel_hours: float,
    buffer_minutes: int = 15
) -> str:
    """
    Calculate departure time given desired arrival and travel duration.
    
    Args:
        arrival_time: Desired arrival time (HH:MM format).
        travel_hours: Travel time in hours.
        buffer_minutes: Extra buffer time to add.
        
    Returns:
        Departure time in HH:MM format.
    """
    from datetime import datetime, timedelta
    
    # Parse arrival time
    arrival = datetime.strptime(arrival_time, "%H:%M")
    
    # Calculate departure
    total_minutes = int(travel_hours * 60) + buffer_minutes
    departure = arrival - timedelta(minutes=total_minutes)
    
    return departure.strftime("%H:%M")


class TravelCalculator(LoggerMixin):
    """
    Calculates travel logistics for surf trips.
    """
    
    def __init__(self):
        """Initialize the travel calculator."""
        pass
    
    def calculate_trip_logistics(
        self,
        origin: str,
        destination: str,
        is_driving: bool,
        surf_window_start: str = "06:00"
    ) -> dict:
        """
        Calculate complete travel logistics.
        
        Args:
            origin: Starting location.
            destination: Surf spot or area.
            is_driving: True if driving, False for public transport.
            surf_window_start: When the surf session starts (HH:MM).
            
        Returns:
            Dictionary with travel details.
        """
        travel_hours = estimate_travel_time(origin, destination, is_driving)
        
        if travel_hours is None:
            return {
                "error": "Could not calculate travel time",
                "origin": origin,
                "destination": destination,
            }
        
        departure_time = get_departure_time(
            surf_window_start,
            travel_hours,
            buffer_minutes=15 if is_driving else 30
        )
        
        return {
            "origin": origin,
            "destination": destination,
            "transport_mode": "car" if is_driving else "public_transport",
            "distance_km": self._get_distance(origin, destination),
            "travel_time_hours": travel_hours,
            "travel_time_formatted": format_travel_time(travel_hours),
            "departure_time": departure_time,
            "arrival_time": surf_window_start,
        }
    
    def _get_distance(self, origin: str, destination: str) -> Optional[float]:
        """Get distance between two locations in km."""
        origin_coords = get_location_coordinates(origin)
        dest_coords = get_location_coordinates(destination)
        
        if origin_coords and dest_coords:
            return round(haversine_distance(
                origin_coords[0], origin_coords[1],
                dest_coords[0], dest_coords[1]
            ), 1)
        return None

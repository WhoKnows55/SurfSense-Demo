"""
Knowledge module for SurfSense.

This module provides access to:
- Surf spot database
- Spot information and recommendations
- Search and filtering capabilities
"""

from app.knowledge.spot_database import (
    SpotDatabase,
    SpotInfo,
    SpotLocation,
    SpotCharacteristics,
    SkillLevels,
    BestConditions,
    SkillLevel,
    BreakType,
    WaveDirection,
    CrowdLevel,
    get_spot_database,
)

__all__ = [
    "SpotDatabase",
    "SpotInfo",
    "SpotLocation",
    "SpotCharacteristics",
    "SkillLevels",
    "BestConditions",
    "SkillLevel",
    "BreakType",
    "WaveDirection",
    "CrowdLevel",
    "get_spot_database",
]

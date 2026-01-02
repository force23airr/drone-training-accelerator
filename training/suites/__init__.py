"""
Mission suites for drone training.
"""

from training.suites.mission_suites import (
    MissionSuite,
    MissionConfig,
    list_missions,
    get_missions_by_difficulty,
    register_mission,
)
from training.suites.military_mission_suites import (
    get_military_missions,
    get_missions_by_platform,
    get_curriculum_sequence,
)

__all__ = [
    # Base mission infrastructure
    "MissionSuite",
    "MissionConfig",
    "list_missions",
    "get_missions_by_difficulty",
    "register_mission",
    # Military missions
    "get_military_missions",
    "get_missions_by_platform",
    "get_curriculum_sequence",
]

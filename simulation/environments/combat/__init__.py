"""
Combat/Military Training Environments.

Provides Gymnasium-compatible environments for military UAV training:
- Base fixed-wing environment with jet aerodynamics
- Air-to-ground strike missions
- Loitering munition scenarios
- Carrier operations
"""

from .base_fixed_wing_env import (
    BaseFixedWingEnv,
    FixedWingObservation,
    WeaponState,
)
from .air_to_ground_env import (
    AirToGroundEnv,
    MissionPhase,
    ThreatType,
    Threat,
    GroundTarget,
    Weapon,
)
from .loitering_munition_env import (
    LoiteringMunitionEnv,
    LoiterPhase,
    TargetMotion,
    LoiterTarget,
    RadarEmitter,
)
from .carrier_ops_env import (
    CarrierOpsEnv,
    CarrierPhase,
    ApproachCase,
    CarrierState,
    CatapultState,
)

__all__ = [
    # Base environment
    "BaseFixedWingEnv",
    "FixedWingObservation",
    "WeaponState",
    # Air-to-ground
    "AirToGroundEnv",
    "MissionPhase",
    "ThreatType",
    "Threat",
    "GroundTarget",
    "Weapon",
    # Loitering munition
    "LoiteringMunitionEnv",
    "LoiterPhase",
    "TargetMotion",
    "LoiterTarget",
    "RadarEmitter",
    # Carrier operations
    "CarrierOpsEnv",
    "CarrierPhase",
    "ApproachCase",
    "CarrierState",
    "CatapultState",
]

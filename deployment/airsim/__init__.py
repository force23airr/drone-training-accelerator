"""
AirSim Deployment Module.

Provides environment configurations and deployment utilities for
visualizing trained policies in Microsoft AirSim.
"""

from .military_environments import (
    MilitaryEnvironmentType,
    MilitaryAirSimConfig,
    MILITARY_ENVIRONMENT_CONFIGS,
    get_military_environment,
    get_carrier_environment,
    get_airbase_environment,
    get_contested_airspace,
    get_urban_strike_zone,
)

__all__ = [
    "MilitaryEnvironmentType",
    "MilitaryAirSimConfig",
    "MILITARY_ENVIRONMENT_CONFIGS",
    "get_military_environment",
    "get_carrier_environment",
    "get_airbase_environment",
    "get_contested_airspace",
    "get_urban_strike_zone",
]

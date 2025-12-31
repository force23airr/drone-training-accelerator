"""
Simulation Module

Core simulation components for drone flight training.
Supports multiple physics backends with PyBullet as the primary engine.

Submodules:
- environments: Gymnasium-compatible drone environments
- platforms: Drone platform configurations
- physics: Physics engine backends
- control: PID controllers and motor mixing
- sensors: Sensor simulation (planned)
"""

# Core environments
from simulation.environments.base_drone_env import (
    BaseDroneEnv,
    EnvironmentalSimulator,  # Backwards compatibility
)

# Environmental conditions
from simulation.environments.environmental_conditions import (
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    TerrainType,
    create_clear_day,
    create_random_conditions,
)

# Mission-specific environments
from simulation.environments.urban_nav_env import UrbanNavigationEnv
from simulation.environments.maritime_patrol_env import MaritimePatrolEnv
from simulation.environments.swarm_coordination_env import SwarmCoordinationEnv

# Platform configurations
from simulation.platforms.platform_configs import (
    get_platform_config,
    list_platforms,
    register_platform,
    create_custom_platform,
    PlatformConfig,
)

# Physics backends
from simulation.physics import PyBulletBackend, SimulatorBackend

# Control systems
from simulation.control import (
    PIDGains,
    PIDController,
    CascadedDroneController,
    SimplePIDController,
)

# Integrated environment (requires gym-pybullet-drones)
from simulation.environments import INTEGRATED_ENV_AVAILABLE
if INTEGRATED_ENV_AVAILABLE:
    from simulation.environments.integrated_environment import IntegratedDroneEnvironment

__all__ = [
    # Environments
    "BaseDroneEnv",
    "EnvironmentalSimulator",
    "UrbanNavigationEnv",
    "MaritimePatrolEnv",
    "SwarmCoordinationEnv",
    # Environmental conditions
    "EnvironmentalConditions",
    "WeatherType",
    "TimeOfDay",
    "TerrainType",
    "create_clear_day",
    "create_random_conditions",
    # Platforms
    "get_platform_config",
    "list_platforms",
    "register_platform",
    "create_custom_platform",
    "PlatformConfig",
    # Physics
    "PyBulletBackend",
    "SimulatorBackend",
    # Control
    "PIDGains",
    "PIDController",
    "CascadedDroneController",
    "SimplePIDController",
    # Availability flags
    "INTEGRATED_ENV_AVAILABLE",
]

# Add integrated environment if available
if INTEGRATED_ENV_AVAILABLE:
    __all__.append("IntegratedDroneEnvironment")

"""
Environment definitions for drone simulation.

Available environments:
- BaseDroneEnv: Core environment with environmental conditions support
- EnvironmentalSimulator: Alias for BaseDroneEnv (backwards compatible)
- IntegratedDroneEnvironment: Full integration with validated dynamics + PID + RL
- UrbanNavigationEnv: Urban/indoor navigation with obstacles
- MaritimePatrolEnv: Maritime patrol and ship tracking
- SwarmCoordinationEnv: Multi-agent formation flying
"""

# Base environment
from simulation.environments.base_drone_env import (
    BaseDroneEnv,
    EnvironmentalSimulator,  # Backwards compatibility
)

# Environmental conditions system
from simulation.environments.environmental_conditions import (
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    TerrainType,
    WindModel,
    create_clear_day,
    create_windy_conditions,
    create_night_conditions,
    create_urban_conditions,
    create_indoor_conditions,
    create_adverse_conditions,
    create_random_conditions,
)

# Mission-specific environments
from simulation.environments.urban_nav_env import UrbanNavigationEnv
from simulation.environments.maritime_patrol_env import MaritimePatrolEnv
from simulation.environments.swarm_coordination_env import SwarmCoordinationEnv

# Integrated environment (requires gym-pybullet-drones)
try:
    from simulation.environments.integrated_environment import (
        IntegratedDroneEnvironment,
        train_hover_with_validated_dynamics,
        compare_rl_vs_pid,
    )
    INTEGRATED_ENV_AVAILABLE = True
except ImportError:
    INTEGRATED_ENV_AVAILABLE = False

__all__ = [
    # Base environment
    "BaseDroneEnv",
    "EnvironmentalSimulator",
    # Environmental conditions
    "EnvironmentalConditions",
    "WeatherType",
    "TimeOfDay",
    "TerrainType",
    "WindModel",
    "create_clear_day",
    "create_windy_conditions",
    "create_night_conditions",
    "create_urban_conditions",
    "create_indoor_conditions",
    "create_adverse_conditions",
    "create_random_conditions",
    # Mission environments
    "UrbanNavigationEnv",
    "MaritimePatrolEnv",
    "SwarmCoordinationEnv",
    # Availability flags
    "INTEGRATED_ENV_AVAILABLE",
]

# Add integrated environment exports if available
if INTEGRATED_ENV_AVAILABLE:
    __all__.extend([
        "IntegratedDroneEnvironment",
        "train_hover_with_validated_dynamics",
        "compare_rl_vs_pid",
    ])

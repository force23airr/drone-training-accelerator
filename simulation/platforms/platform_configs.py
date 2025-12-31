"""
Platform Configurations

Defines configuration profiles for various drone platforms.
Supports quadcopters, fixed-wing, VTOL, and custom platforms.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import copy


@dataclass
class PlatformConfig:
    """
    Configuration for a drone platform.

    Attributes:
        name: Human-readable platform name
        platform_type: Category (quadcopter, fixed_wing, vtol, custom)
        num_motors: Number of motors/actuators
        mass: Total mass in kg
        arm_length: Distance from center to motor (for multirotors)
        max_thrust_per_motor: Maximum thrust per motor in Newtons
        max_rpm: Maximum motor RPM
        observation_dim: Dimension of observation vector
        max_episode_steps: Maximum steps per episode
        physics_params: Additional physics parameters
    """
    name: str
    platform_type: str
    num_motors: int
    mass: float
    arm_length: float
    max_thrust_per_motor: float
    max_rpm: int = 10000
    observation_dim: int = 13
    max_episode_steps: int = 1000
    physics_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "platform_type": self.platform_type,
            "num_motors": self.num_motors,
            "mass": self.mass,
            "arm_length": self.arm_length,
            "max_thrust_per_motor": self.max_thrust_per_motor,
            "max_rpm": self.max_rpm,
            "observation_dim": self.observation_dim,
            "max_episode_steps": self.max_episode_steps,
            "physics_params": self.physics_params,
        }


# Registry of available platforms
_PLATFORM_REGISTRY: Dict[str, PlatformConfig] = {}


def register_platform(platform_id: str, config: PlatformConfig):
    """Register a new platform configuration."""
    _PLATFORM_REGISTRY[platform_id] = config


def get_platform_config(platform_id: str) -> Dict[str, Any]:
    """
    Get platform configuration by ID.

    Args:
        platform_id: Identifier for the platform

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If platform_id not found
    """
    if platform_id not in _PLATFORM_REGISTRY:
        available = ", ".join(_PLATFORM_REGISTRY.keys())
        raise ValueError(
            f"Unknown platform '{platform_id}'. Available: {available}"
        )
    return _PLATFORM_REGISTRY[platform_id].to_dict()


def list_platforms() -> List[str]:
    """List all available platform IDs."""
    return list(_PLATFORM_REGISTRY.keys())


def get_platforms_by_type(platform_type: str) -> List[str]:
    """Get all platforms of a specific type."""
    return [
        pid for pid, config in _PLATFORM_REGISTRY.items()
        if config.platform_type == platform_type
    ]


# =============================================================================
# QUADCOPTER PLATFORMS
# =============================================================================

register_platform(
    "quadcopter_basic",
    PlatformConfig(
        name="Basic Quadcopter",
        platform_type="quadcopter",
        num_motors=4,
        mass=1.0,
        arm_length=0.15,
        max_thrust_per_motor=5.0,
        max_rpm=10000,
        physics_params={
            "drag_coefficient": 0.1,
            "moment_of_inertia": [0.01, 0.01, 0.02],
        }
    )
)

register_platform(
    "quadcopter_racing",
    PlatformConfig(
        name="Racing Quadcopter",
        platform_type="quadcopter",
        num_motors=4,
        mass=0.5,
        arm_length=0.1,
        max_thrust_per_motor=8.0,
        max_rpm=25000,
        physics_params={
            "drag_coefficient": 0.05,
            "moment_of_inertia": [0.005, 0.005, 0.008],
        }
    )
)

register_platform(
    "quadcopter_heavy_lift",
    PlatformConfig(
        name="Heavy Lift Quadcopter",
        platform_type="quadcopter",
        num_motors=4,
        mass=5.0,
        arm_length=0.3,
        max_thrust_per_motor=25.0,
        max_rpm=8000,
        physics_params={
            "drag_coefficient": 0.15,
            "moment_of_inertia": [0.1, 0.1, 0.15],
            "payload_capacity": 3.0,
        }
    )
)

# =============================================================================
# HEXACOPTER PLATFORMS
# =============================================================================

register_platform(
    "hexacopter_standard",
    PlatformConfig(
        name="Standard Hexacopter",
        platform_type="hexacopter",
        num_motors=6,
        mass=3.0,
        arm_length=0.25,
        max_thrust_per_motor=10.0,
        max_rpm=10000,
        physics_params={
            "drag_coefficient": 0.12,
            "moment_of_inertia": [0.05, 0.05, 0.08],
            "redundancy": True,
        }
    )
)

# =============================================================================
# OCTOCOPTER PLATFORMS
# =============================================================================

register_platform(
    "octocopter_industrial",
    PlatformConfig(
        name="Industrial Octocopter",
        platform_type="octocopter",
        num_motors=8,
        mass=8.0,
        arm_length=0.4,
        max_thrust_per_motor=20.0,
        max_rpm=8000,
        physics_params={
            "drag_coefficient": 0.18,
            "moment_of_inertia": [0.2, 0.2, 0.3],
            "payload_capacity": 5.0,
            "redundancy": True,
        }
    )
)

# =============================================================================
# FIXED-WING PLATFORMS
# =============================================================================

register_platform(
    "fixed_wing_trainer",
    PlatformConfig(
        name="Fixed Wing Trainer",
        platform_type="fixed_wing",
        num_motors=1,
        mass=2.0,
        arm_length=0.0,  # N/A for fixed wing
        max_thrust_per_motor=15.0,
        observation_dim=16,  # Additional aerodynamic states
        physics_params={
            "wingspan": 1.5,
            "wing_area": 0.3,
            "lift_coefficient": 0.5,
            "drag_coefficient": 0.03,
            "stall_angle": 15.0,
        }
    )
)

register_platform(
    "fixed_wing_survey",
    PlatformConfig(
        name="Survey Fixed Wing",
        platform_type="fixed_wing",
        num_motors=1,
        mass=4.0,
        arm_length=0.0,
        max_thrust_per_motor=20.0,
        max_episode_steps=5000,  # Longer missions
        observation_dim=16,
        physics_params={
            "wingspan": 2.5,
            "wing_area": 0.6,
            "lift_coefficient": 0.6,
            "drag_coefficient": 0.025,
            "stall_angle": 12.0,
            "endurance_minutes": 90,
        }
    )
)

# =============================================================================
# VTOL PLATFORMS
# =============================================================================

register_platform(
    "vtol_tiltrotor",
    PlatformConfig(
        name="Tiltrotor VTOL",
        platform_type="vtol",
        num_motors=4,
        mass=3.5,
        arm_length=0.2,
        max_thrust_per_motor=12.0,
        observation_dim=18,  # Includes tilt angles
        physics_params={
            "wingspan": 1.2,
            "tilt_range": 90.0,
            "transition_speed": 15.0,
            "wing_area": 0.25,
        }
    )
)

register_platform(
    "vtol_tailsitter",
    PlatformConfig(
        name="Tailsitter VTOL",
        platform_type="vtol",
        num_motors=2,
        mass=2.0,
        arm_length=0.0,
        max_thrust_per_motor=15.0,
        observation_dim=16,
        physics_params={
            "wingspan": 1.0,
            "transition_pitch": 90.0,
            "wing_area": 0.2,
        }
    )
)

# =============================================================================
# SPECIALIZED PLATFORMS
# =============================================================================

register_platform(
    "fpv_micro",
    PlatformConfig(
        name="Micro FPV Drone",
        platform_type="quadcopter",
        num_motors=4,
        mass=0.25,
        arm_length=0.05,
        max_thrust_per_motor=2.0,
        max_rpm=30000,
        physics_params={
            "drag_coefficient": 0.03,
            "moment_of_inertia": [0.001, 0.001, 0.002],
            "acro_capable": True,
        }
    )
)

register_platform(
    "inspection_drone",
    PlatformConfig(
        name="Industrial Inspection Drone",
        platform_type="quadcopter",
        num_motors=4,
        mass=2.5,
        arm_length=0.2,
        max_thrust_per_motor=12.0,
        observation_dim=19,  # Includes camera gimbal state
        physics_params={
            "drag_coefficient": 0.12,
            "moment_of_inertia": [0.03, 0.03, 0.05],
            "gimbal_range": [-90, 30],
            "camera_resolution": [4096, 2160],
        }
    )
)


def create_custom_platform(
    name: str,
    base_platform: str,
    overrides: Dict[str, Any]
) -> str:
    """
    Create a custom platform based on an existing one.

    Args:
        name: Name for the new platform
        base_platform: Platform ID to use as base
        overrides: Parameters to override

    Returns:
        New platform ID
    """
    base_config = _PLATFORM_REGISTRY[base_platform]
    new_config_dict = base_config.to_dict()
    new_config_dict["name"] = name

    # Apply overrides
    for key, value in overrides.items():
        if key == "physics_params" and key in new_config_dict:
            new_config_dict[key].update(value)
        else:
            new_config_dict[key] = value

    new_config = PlatformConfig(**new_config_dict)
    platform_id = name.lower().replace(" ", "_")
    register_platform(platform_id, new_config)

    return platform_id

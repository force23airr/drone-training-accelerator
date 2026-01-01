"""
AirSim Environment Configurations

Defines realistic environments for drone testing and visualization.
These map to AirSim/Unreal Engine environments.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import gymnasium as gym
from gymnasium import spaces


class EnvironmentType(Enum):
    """Pre-built realistic environments available in AirSim."""

    # Urban Environments
    CITY_DOWNTOWN = "CityDowntown"       # Dense urban, skyscrapers
    CITY_RESIDENTIAL = "Neighborhood"     # Suburban streets
    CITY_INDUSTRIAL = "Warehouse"         # Industrial zone

    # Natural Environments
    FOREST = "Forest"                     # Dense tree coverage
    MOUNTAINS = "LandscapeMountains"      # Mountain terrain
    DESERT = "Africa"                     # Open desert/savanna
    COASTAL = "CoastalTerrain"            # Beach/ocean

    # Military/Special
    MILITARY_BASE = "MilitaryBase"        # Runways, hangars
    TRAINING_GROUND = "Blocks"            # Simple test environment

    # Indoor
    WAREHOUSE_INDOOR = "WarehouseIndoor"  # Indoor warehouse
    BUILDING_INTERIOR = "Building99"       # Office building

    # Custom
    FLORIDA_STREET = "FloridaStreet"      # Custom: Florida streets
    CUSTOM = "Custom"                      # User-defined


@dataclass
class RealisticEnvironmentConfig:
    """
    Configuration for a realistic simulation environment.

    Use this to define custom environments for testing trained policies.
    """

    # Environment identification
    name: str
    environment_type: EnvironmentType
    description: str = ""

    # Geographic settings (for custom environments)
    location: Optional[str] = None  # e.g., "Miami, FL" or GPS coordinates
    latitude: float = 25.7617       # Default: Miami
    longitude: float = -80.1918

    # Terrain settings
    terrain_size: Tuple[float, float] = (1000.0, 1000.0)  # meters
    terrain_type: str = "urban"  # urban, forest, desert, coastal, mountain
    elevation_range: Tuple[float, float] = (0.0, 50.0)  # min/max meters

    # Weather presets
    weather_preset: str = "clear"  # clear, cloudy, rain, storm, fog, snow
    wind_speed: float = 5.0        # m/s base wind speed
    wind_direction: float = 0.0    # degrees from north

    # Time settings
    time_of_day: str = "noon"      # dawn, morning, noon, afternoon, dusk, night
    date: str = "2024-06-21"       # Affects sun position

    # Obstacles/Objects
    building_density: str = "medium"  # none, low, medium, high
    tree_density: str = "low"
    vehicle_traffic: bool = False
    pedestrian_traffic: bool = False

    # Flight restrictions
    no_fly_zones: List[Dict[str, Any]] = field(default_factory=list)
    max_altitude: float = 120.0    # meters (FAA limit)
    min_altitude: float = 0.5

    # Spawn settings
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    spawn_orientation: float = 0.0  # yaw in degrees

    # Sensor configurations
    enable_cameras: bool = True
    enable_lidar: bool = False
    enable_gps_noise: bool = True
    gps_accuracy: float = 2.5      # meters

    @classmethod
    def florida_street(cls) -> "RealisticEnvironmentConfig":
        """Pre-configured Florida street environment."""
        return cls(
            name="Florida Street Test",
            environment_type=EnvironmentType.FLORIDA_STREET,
            description="Urban street in Miami, FL for realistic testing",
            location="Miami, FL",
            latitude=25.7617,
            longitude=-80.1918,
            terrain_type="urban",
            weather_preset="clear",
            time_of_day="noon",
            building_density="high",
            tree_density="medium",
            vehicle_traffic=True,
        )

    @classmethod
    def mountain_recon(cls) -> "RealisticEnvironmentConfig":
        """Mountain reconnaissance environment."""
        return cls(
            name="Mountain Reconnaissance",
            environment_type=EnvironmentType.MOUNTAINS,
            description="High-altitude mountain terrain for recon missions",
            terrain_type="mountain",
            elevation_range=(500.0, 3000.0),
            weather_preset="cloudy",
            wind_speed=15.0,
            building_density="none",
            tree_density="medium",
            max_altitude=500.0,
        )

    @classmethod
    def urban_search_rescue(cls) -> "RealisticEnvironmentConfig":
        """Urban search and rescue environment."""
        return cls(
            name="Urban Search & Rescue",
            environment_type=EnvironmentType.CITY_DOWNTOWN,
            description="Dense urban environment for SAR operations",
            terrain_type="urban",
            building_density="high",
            weather_preset="cloudy",
            enable_lidar=True,
            vehicle_traffic=True,
            pedestrian_traffic=True,
        )

    @classmethod
    def warehouse_inspection(cls) -> "RealisticEnvironmentConfig":
        """Indoor warehouse inspection environment."""
        return cls(
            name="Warehouse Inspection",
            environment_type=EnvironmentType.WAREHOUSE_INDOOR,
            description="Indoor warehouse for inspection drone testing",
            terrain_type="indoor",
            max_altitude=15.0,
            enable_lidar=True,
            enable_gps_noise=True,
            gps_accuracy=10.0,  # Poor GPS indoors
        )

    @classmethod
    def coastal_patrol(cls) -> "RealisticEnvironmentConfig":
        """Coastal/maritime patrol environment."""
        return cls(
            name="Coastal Patrol",
            environment_type=EnvironmentType.COASTAL,
            description="Coastal area for maritime patrol missions",
            terrain_type="coastal",
            weather_preset="clear",
            wind_speed=12.0,  # Higher coastal winds
            building_density="low",
            max_altitude=150.0,
        )


class AirSimDroneEnv(gym.Env):
    """
    Gymnasium environment using AirSim for photorealistic simulation.

    This environment is for TESTING trained policies, not training.
    Training should be done in PyBullet (faster), then policies
    are deployed here for realistic visualization and validation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        environment_config: Optional[RealisticEnvironmentConfig] = None,
        platform_config: Optional[Dict[str, Any]] = None,
        render_mode: str = "human",
    ):
        """
        Initialize AirSim environment.

        Args:
            environment_config: Realistic environment configuration
            platform_config: Drone platform configuration
            render_mode: Rendering mode
        """
        super().__init__()

        from simulation.physics.airsim_integration.airsim_backend import (
            AirSimBackend,
            AIRSIM_AVAILABLE,
        )

        if not AIRSIM_AVAILABLE:
            raise ImportError(
                "AirSim not available. Install with: pip install airsim\n"
                "And download AirSim: https://github.com/microsoft/AirSim/releases"
            )

        self.env_config = environment_config or RealisticEnvironmentConfig.florida_street()
        self.platform_config = platform_config or {}
        self.render_mode = render_mode

        # Initialize backend
        self.backend = AirSimBackend()
        self._initialized = False

        # Define spaces (matches PyBullet training environment)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        # Action space: motor commands or high-level position
        self.action_space = spaces.Box(
            low=np.array([-5, -5, 0, -np.pi]),
            high=np.array([5, 5, 3, np.pi]),
            dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.max_episode_steps = 1000
        self.target_position = np.array([0, 0, 2])

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if not self._initialized:
            self._initialize_airsim()

        self.backend.reset()
        self.current_step = 0

        # Set weather based on config
        self._apply_environment_settings()

        # Get initial observation
        obs = self._get_observation()
        info = {"step": 0, "environment": self.env_config.name}

        return obs, info

    def _initialize_airsim(self):
        """Initialize AirSim connection."""
        self.backend.initialize(
            render_mode=self.render_mode,
            environment=self.env_config.environment_type.value
        )
        self._initialized = True

    def _apply_environment_settings(self):
        """Apply environment configuration to AirSim."""
        # Weather
        weather_map = {
            "clear": (0, 0, 0, 0),
            "rain": (0.5, 0, 0, 0),
            "storm": (1.0, 0, 0, 0),
            "fog": (0, 0, 0.5, 0),
            "snow": (0, 0.5, 0, 0),
            "dust": (0, 0, 0, 0.5),
        }
        weather = weather_map.get(self.env_config.weather_preset, (0, 0, 0, 0))
        self.backend.set_weather(*weather)

        # Time of day
        time_map = {
            "dawn": "2024-06-21 06:00:00",
            "morning": "2024-06-21 09:00:00",
            "noon": "2024-06-21 12:00:00",
            "afternoon": "2024-06-21 15:00:00",
            "dusk": "2024-06-21 18:00:00",
            "night": "2024-06-21 22:00:00",
        }
        time_str = time_map.get(self.env_config.time_of_day, "2024-06-21 12:00:00")
        self.backend.set_time_of_day(True, time_str)

        # Wind
        wind_rad = np.radians(self.env_config.wind_direction)
        wind_vec = (
            self.env_config.wind_speed * np.cos(wind_rad),
            self.env_config.wind_speed * np.sin(wind_rad),
            0
        )
        self.backend.set_wind(wind_vec)

    def step(self, action):
        """Execute action and return observation."""
        self.current_step += 1

        # Interpret action as target position + yaw
        target_pos = action[:3]
        target_yaw = action[3]

        # Move drone toward target
        self.backend.move_to_position(tuple(target_pos), velocity=3.0)

        # Get observation
        obs = self._get_observation()

        # Compute reward (simplified for visualization)
        pos = self.backend.get_position()
        reward = -np.linalg.norm(pos - self.target_position)

        # Check termination
        terminated = False
        if pos[2] < self.env_config.min_altitude:
            terminated = True  # Crashed

        contacts = self.backend.get_contacts()
        if contacts:
            terminated = True  # Collision

        truncated = self.current_step >= self.max_episode_steps

        info = {
            "step": self.current_step,
            "position": pos.tolist(),
            "collision": len(contacts) > 0,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        state = self.backend.get_state()
        pos = state['position']
        vel = state['velocity']
        quat = state['orientation']
        ang_vel = state['angular_velocity']

        # Convert quaternion to euler
        rpy = self.backend.quaternion_to_euler(tuple(quat))

        obs = np.concatenate([
            pos,
            vel,
            np.array(rpy),
            ang_vel,
            self.target_position,
            [0.0]  # target yaw
        ])

        return obs.astype(np.float32)

    def render(self):
        """Render current frame."""
        if self.render_mode == "rgb_array":
            return self.backend.get_front_camera_image()
        return None

    def close(self):
        """Clean up."""
        if self._initialized:
            self.backend.shutdown()
            self._initialized = False

    def get_camera_view(self) -> np.ndarray:
        """Get first-person camera view."""
        return self.backend.get_front_camera_image()

    def get_depth_view(self) -> np.ndarray:
        """Get depth camera view."""
        return self.backend.get_depth_image()

    def get_lidar_points(self) -> np.ndarray:
        """Get LiDAR point cloud."""
        return self.backend.get_lidar_data()


# Pre-built environment configurations
PRESET_ENVIRONMENTS = {
    "florida_street": RealisticEnvironmentConfig.florida_street(),
    "mountain_recon": RealisticEnvironmentConfig.mountain_recon(),
    "urban_sar": RealisticEnvironmentConfig.urban_search_rescue(),
    "warehouse": RealisticEnvironmentConfig.warehouse_inspection(),
    "coastal_patrol": RealisticEnvironmentConfig.coastal_patrol(),
}


def get_environment_config(name: str) -> RealisticEnvironmentConfig:
    """Get a preset environment configuration by name."""
    if name in PRESET_ENVIRONMENTS:
        return PRESET_ENVIRONMENTS[name]
    raise ValueError(f"Unknown environment: {name}. Available: {list(PRESET_ENVIRONMENTS.keys())}")


def list_environments() -> List[str]:
    """List available environment presets."""
    return list(PRESET_ENVIRONMENTS.keys())

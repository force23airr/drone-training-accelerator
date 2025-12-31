"""
Urban Navigation Environment

Specialized environment for urban and indoor drone navigation training.
Features GPS-denied operation, obstacle avoidance, and waypoint navigation.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from gymnasium import spaces

from simulation.environments.base_drone_env import BaseDroneEnv
from simulation.environments.environmental_conditions import (
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    TerrainType,
    WindModel,
    create_urban_conditions,
    create_indoor_conditions,
)


class UrbanNavigationEnv(BaseDroneEnv):
    """
    Urban navigation mission environment.

    Mission objectives:
    - Navigate through urban environment with obstacles
    - GPS-denied/degraded indoor navigation
    - Reach target waypoints in sequence
    - Avoid collisions with structures

    Features:
    - Building and obstacle generation
    - Multiple waypoint targets
    - RF interference simulation
    - Limited visibility scenarios
    """

    def __init__(
        self,
        platform_config: Dict[str, Any],
        scenario: str = "outdoor_urban",
        num_waypoints: int = 5,
        obstacle_density: str = "medium",
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize urban navigation environment.

        Args:
            platform_config: Drone platform configuration
            scenario: 'outdoor_urban', 'indoor', 'parking_garage'
            num_waypoints: Number of waypoints to navigate
            obstacle_density: 'low', 'medium', 'high'
            render_mode: Rendering mode
        """
        # Set up environmental conditions based on scenario
        if scenario == "indoor":
            conditions = create_indoor_conditions()
        elif scenario == "parking_garage":
            conditions = EnvironmentalConditions(
                weather=WeatherType.CLEAR,
                time_of_day=TimeOfDay.DAY,
                terrain=TerrainType.INDOOR,
                visibility=30.0,
                rf_interference=0.6,
                gps_degradation=0.9,
            )
        else:  # outdoor_urban
            conditions = create_urban_conditions()

        self.scenario = scenario
        self.num_waypoints = num_waypoints
        self.obstacle_density = obstacle_density

        # Waypoint tracking
        self._waypoints: List[np.ndarray] = []
        self._current_waypoint_idx = 0
        self._waypoints_reached = 0

        # Extend observation space to include relative waypoint position
        original_obs_dim = platform_config.get("observation_dim", 13)
        platform_config["observation_dim"] = original_obs_dim + 3  # Add relative target

        super().__init__(
            platform_config=platform_config,
            environmental_conditions=conditions,
            render_mode=render_mode,
            **kwargs
        )

        # Override observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_obs_dim + 3,),
            dtype=np.float32
        )

    def _setup_environment(self):
        """Set up urban environment with buildings and obstacles."""
        self._generate_waypoints()
        self._generate_obstacles()
        self._current_waypoint_idx = 0
        self._waypoints_reached = 0
        self._update_target()

    def _generate_waypoints(self):
        """Generate waypoint sequence."""
        self._waypoints = []

        if self.scenario == "indoor":
            # Indoor waypoints in a smaller space
            for i in range(self.num_waypoints):
                wp = np.array([
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    np.random.uniform(0.5, 2.5)
                ])
                self._waypoints.append(wp)
        else:
            # Outdoor urban waypoints
            for i in range(self.num_waypoints):
                wp = np.array([
                    np.random.uniform(-15, 15),
                    np.random.uniform(-15, 15),
                    np.random.uniform(2, 10)
                ])
                self._waypoints.append(wp)

    def _generate_obstacles(self):
        """Generate urban obstacles (buildings, walls, etc.)."""
        density_map = {"low": 5, "medium": 10, "high": 20}
        num_obstacles = density_map.get(self.obstacle_density, 10)

        if self.scenario == "indoor":
            self._generate_indoor_obstacles(num_obstacles)
        elif self.scenario == "parking_garage":
            self._generate_parking_garage()
        else:
            self._generate_buildings(num_obstacles)

    def _generate_buildings(self, num_buildings: int):
        """Generate building-like obstacles for outdoor urban."""
        for _ in range(num_buildings):
            # Random building dimensions
            width = np.random.uniform(1, 4)
            depth = np.random.uniform(1, 4)
            height = np.random.uniform(5, 20)

            # Random position (avoid center spawn area)
            x = np.random.uniform(-20, 20)
            y = np.random.uniform(-20, 20)
            if abs(x) < 3 and abs(y) < 3:
                x = np.sign(x) * 5 if x != 0 else 5

            position = (x, y, height / 2)
            size = (width / 2, depth / 2, height / 2)

            # Vary building colors
            colors = [
                (0.6, 0.6, 0.6, 1.0),  # Gray
                (0.5, 0.4, 0.3, 1.0),  # Brown
                (0.7, 0.7, 0.8, 1.0),  # Light blue-gray
            ]
            color = colors[np.random.randint(len(colors))]

            self.add_obstacle("box", position, size, color)

    def _generate_indoor_obstacles(self, num_obstacles: int):
        """Generate indoor obstacles (furniture, walls, pillars)."""
        # Add walls
        wall_height = 3.0
        wall_thickness = 0.1
        room_size = 8.0

        # Partial walls with doorways
        self.add_obstacle(
            "box",
            (room_size / 2, 0, wall_height / 2),
            (wall_thickness, room_size / 3, wall_height / 2),
            (0.8, 0.8, 0.8, 1.0)
        )

        # Pillars
        for i in range(4):
            x = np.random.uniform(-room_size / 2, room_size / 2)
            y = np.random.uniform(-room_size / 2, room_size / 2)
            if abs(x) < 1 and abs(y) < 1:
                continue
            self.add_obstacle(
                "cylinder",
                (x, y, wall_height / 2),
                (0.2, wall_height),
                (0.5, 0.5, 0.5, 1.0)
            )

        # Furniture-like obstacles
        for _ in range(num_obstacles - 4):
            x = np.random.uniform(-room_size / 2, room_size / 2)
            y = np.random.uniform(-room_size / 2, room_size / 2)
            if abs(x) < 2 and abs(y) < 2:
                continue
            height = np.random.uniform(0.5, 1.5)
            size = np.random.uniform(0.3, 0.8)
            self.add_obstacle(
                "box",
                (x, y, height / 2),
                (size, size, height / 2),
                (0.4, 0.3, 0.2, 1.0)
            )

    def _generate_parking_garage(self):
        """Generate parking garage structure."""
        # Ceiling
        self.add_obstacle(
            "box",
            (0, 0, 3.0),
            (15, 15, 0.1),
            (0.5, 0.5, 0.5, 0.3)
        )

        # Pillars in grid
        for x in [-8, -4, 0, 4, 8]:
            for y in [-8, -4, 0, 4, 8]:
                if abs(x) < 2 and abs(y) < 2:
                    continue
                self.add_obstacle(
                    "cylinder",
                    (x, y, 1.5),
                    (0.3, 3.0),
                    (0.6, 0.6, 0.6, 1.0)
                )

        # Parked vehicles
        for _ in range(8):
            x = np.random.uniform(-12, 12)
            y = np.random.uniform(-12, 12)
            if abs(x) < 3 and abs(y) < 3:
                continue
            self.add_obstacle(
                "box",
                (x, y, 0.75),
                (2.0, 1.0, 0.75),
                (0.3, 0.3, 0.5, 1.0)
            )

    def _update_target(self):
        """Update target position to current waypoint."""
        if self._current_waypoint_idx < len(self._waypoints):
            self._target_position = self._waypoints[self._current_waypoint_idx]

    def _get_observation(self) -> np.ndarray:
        """Get observation including relative waypoint position."""
        # Get base observation
        base_obs = super()._get_observation()

        # Calculate relative position to current waypoint
        drone_pos = base_obs[:3]
        relative_target = self._target_position - drone_pos

        # Concatenate
        return np.concatenate([base_obs, relative_target]).astype(np.float32)

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute urban navigation reward."""
        pos = obs[:3]
        vel = obs[3:6]

        reward = 0.0

        # Distance to current waypoint
        target_dist = np.linalg.norm(pos - self._target_position)

        # Progress reward
        reward += 1.0 - 0.1 * target_dist

        # Waypoint reached bonus
        waypoint_threshold = 0.5 if self.scenario == "indoor" else 1.0
        if target_dist < waypoint_threshold:
            reward += 20.0
            self._waypoints_reached += 1
            self._current_waypoint_idx += 1
            if self._current_waypoint_idx < len(self._waypoints):
                self._update_target()

        # Velocity penalty (encourage smooth flight)
        reward -= 0.05 * np.linalg.norm(vel)

        # Action smoothness
        reward -= 0.01 * np.sum(action**2)

        # Collision penalty
        if self._check_collision():
            reward -= 50.0
            self._collision_count += 1

        # Height maintenance (indoor scenarios)
        if self.scenario in ["indoor", "parking_garage"]:
            if pos[2] < 0.3 or pos[2] > 2.5:
                reward -= 5.0

        return float(reward)

    def _check_termination(self, obs: np.ndarray) -> Tuple[bool, bool]:
        """Check termination conditions."""
        pos = obs[:3]
        terminated = False

        # Ground/ceiling collision
        if pos[2] < 0.05:
            terminated = True
        if self.scenario in ["indoor", "parking_garage"] and pos[2] > 2.8:
            terminated = True

        # Out of bounds
        bounds = 10 if self.scenario == "indoor" else 25
        if abs(pos[0]) > bounds or abs(pos[1]) > bounds:
            terminated = True

        # Obstacle collision
        if self._check_collision():
            terminated = True

        # Mission complete (all waypoints reached)
        if self._waypoints_reached >= self.num_waypoints:
            terminated = True  # Success!

        # Max steps
        max_steps = self.platform_config.get("max_episode_steps", 2000)
        truncated = self._step_count >= max_steps

        return terminated, truncated

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        info["num_waypoints"] = self.num_waypoints
        info["scenario"] = self.scenario

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step with waypoint tracking."""
        obs, reward, terminated, truncated, info = super().step(action)

        info["waypoints_reached"] = self._waypoints_reached
        info["current_waypoint"] = self._current_waypoint_idx
        info["mission_complete"] = self._waypoints_reached >= self.num_waypoints

        return obs, reward, terminated, truncated, info

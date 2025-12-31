"""
Maritime Patrol Environment

Specialized environment for maritime/coastal drone operations.
Features ship tracking, search patterns, and challenging wind conditions.
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
)


class MaritimePatrolEnv(BaseDroneEnv):
    """
    Maritime patrol mission environment.

    Mission objectives:
    - Search area patrol patterns
    - Moving target (ship) tracking
    - Return to base operations
    - Operations under maritime wind conditions

    Features:
    - Simulated ocean surface
    - Moving ship targets
    - Strong and variable winds
    - Search pattern generation
    - Endurance management
    """

    def __init__(
        self,
        platform_config: Dict[str, Any],
        mission_type: str = "patrol",
        search_area_size: float = 100.0,
        num_targets: int = 1,
        target_speed: float = 5.0,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize maritime patrol environment.

        Args:
            platform_config: Drone platform configuration
            mission_type: 'patrol', 'tracking', 'search_rescue'
            search_area_size: Size of patrol area in meters
            num_targets: Number of ships/targets to track
            target_speed: Target movement speed (m/s)
            render_mode: Rendering mode
        """
        # Maritime conditions: coastal terrain with higher winds
        conditions = EnvironmentalConditions(
            weather=WeatherType.CLEAR,
            time_of_day=TimeOfDay.DAY,
            terrain=TerrainType.COASTAL,
            humidity=75.0,
            wind=WindModel(
                base_speed=8.0,
                base_direction=np.random.uniform(0, 2 * np.pi),
                gust_intensity=5.0,
                gust_probability=0.15,
                turbulence_intensity=1.0,
            )
        )

        self.mission_type = mission_type
        self.search_area_size = search_area_size
        self.num_targets = num_targets
        self.target_speed = target_speed

        # Target tracking
        self._targets: List[Dict[str, Any]] = []
        self._detected_targets: List[int] = []
        self._tracking_target_idx: Optional[int] = None

        # Patrol pattern
        self._patrol_waypoints: List[np.ndarray] = []
        self._current_patrol_idx = 0

        # Base position
        self._base_position = np.array([0.0, 0.0, 0.0])

        # Fuel/endurance simulation
        self._fuel_remaining = 1.0  # 0-1 scale
        self._fuel_consumption_rate = 0.0001  # per step

        # Extend observation for maritime
        original_obs_dim = platform_config.get("observation_dim", 13)
        # Add: relative target pos (3), fuel (1), target detected (1)
        platform_config["observation_dim"] = original_obs_dim + 5

        super().__init__(
            platform_config=platform_config,
            environmental_conditions=conditions,
            render_mode=render_mode,
            **kwargs
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_obs_dim + 5,),
            dtype=np.float32
        )

    def _setup_environment(self):
        """Set up maritime environment."""
        # Create ocean surface (visual only, no collision)
        self._create_ocean_visual()

        # Generate targets (ships)
        self._generate_targets()

        # Generate patrol pattern
        self._generate_patrol_pattern()

        # Set initial mission target
        self._update_mission_target()

        # Reset fuel
        self._fuel_remaining = 1.0

    def _create_ocean_visual(self):
        """Create visual representation of ocean surface."""
        # Large blue plane for ocean
        # Note: We don't add this as a collision obstacle
        import pybullet as p

        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.search_area_size, self.search_area_size, 0.01],
            rgbaColor=[0.0, 0.2, 0.5, 0.8],
            physicsClientId=self.physics_client
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, -0.1],
            physicsClientId=self.physics_client
        )

    def _generate_targets(self):
        """Generate ship targets."""
        self._targets = []
        for i in range(self.num_targets):
            # Random position within search area
            pos = np.array([
                np.random.uniform(-self.search_area_size / 2, self.search_area_size / 2),
                np.random.uniform(-self.search_area_size / 2, self.search_area_size / 2),
                0.0  # On water surface
            ])

            # Random heading
            heading = np.random.uniform(0, 2 * np.pi)

            self._targets.append({
                "position": pos,
                "heading": heading,
                "speed": self.target_speed * np.random.uniform(0.5, 1.5),
                "detected": False,
                "id": i,
            })

            # Create visual for ship
            self._create_ship_visual(pos, i)

    def _create_ship_visual(self, position: np.ndarray, target_id: int):
        """Create visual representation of a ship."""
        import pybullet as p

        # Ship body
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[5.0, 1.5, 0.5],
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            physicsClientId=self.physics_client
        )
        ship_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=position.tolist(),
            physicsClientId=self.physics_client
        )
        self._targets[target_id]["visual_id"] = ship_id

    def _generate_patrol_pattern(self):
        """Generate search/patrol pattern waypoints."""
        self._patrol_waypoints = []

        if self.mission_type == "patrol":
            # Lawnmower pattern
            altitude = 20.0
            spacing = self.search_area_size / 5
            for i, x in enumerate(np.linspace(
                -self.search_area_size / 2,
                self.search_area_size / 2,
                5
            )):
                if i % 2 == 0:
                    y_range = np.linspace(
                        -self.search_area_size / 2,
                        self.search_area_size / 2,
                        5
                    )
                else:
                    y_range = np.linspace(
                        self.search_area_size / 2,
                        -self.search_area_size / 2,
                        5
                    )
                for y in y_range:
                    self._patrol_waypoints.append(np.array([x, y, altitude]))

        elif self.mission_type == "search_rescue":
            # Expanding square pattern
            altitude = 15.0
            center = np.array([0, 0])
            distance = 10.0
            for i in range(20):
                direction = i % 4
                if direction == 0:  # North
                    center[1] += distance
                elif direction == 1:  # East
                    center[0] += distance
                elif direction == 2:  # South
                    center[1] -= distance
                elif direction == 3:  # West
                    center[0] -= distance
                if i % 2 == 1:
                    distance += 10.0
                self._patrol_waypoints.append(
                    np.array([center[0], center[1], altitude])
                )

        else:  # tracking - waypoints set dynamically
            self._patrol_waypoints.append(np.array([0, 0, 15.0]))

    def _update_targets(self):
        """Update target positions (simulate ship movement)."""
        import pybullet as p

        for target in self._targets:
            # Move ship
            heading = target["heading"]
            speed = target["speed"]
            dt = 1.0 / self.control_hz

            target["position"][0] += speed * np.cos(heading) * dt
            target["position"][1] += speed * np.sin(heading) * dt

            # Boundary wrapping
            for i in range(2):
                if abs(target["position"][i]) > self.search_area_size / 2:
                    target["position"][i] = -np.sign(target["position"][i]) * self.search_area_size / 2

            # Random heading changes
            if np.random.random() < 0.01:
                target["heading"] += np.random.uniform(-0.3, 0.3)

            # Update visual
            if "visual_id" in target:
                orn = p.getQuaternionFromEuler([0, 0, heading])
                p.resetBasePositionAndOrientation(
                    target["visual_id"],
                    target["position"].tolist(),
                    orn,
                    physicsClientId=self.physics_client
                )

    def _update_mission_target(self):
        """Update mission target based on mission type."""
        if self.mission_type == "tracking" and self._tracking_target_idx is not None:
            # Track specific target
            target = self._targets[self._tracking_target_idx]
            # Offset above target
            self._target_position = target["position"] + np.array([0, 0, 10.0])
        elif self._patrol_waypoints:
            self._target_position = self._patrol_waypoints[self._current_patrol_idx]
        else:
            self._target_position = np.array([0, 0, 15.0])

    def _check_target_detection(self, drone_pos: np.ndarray):
        """Check if drone can detect targets."""
        detection_range = 30.0  # meters
        detection_angle = np.radians(60)  # downward cone

        for target in self._targets:
            if target["detected"]:
                continue

            # Distance check
            rel_pos = target["position"] - drone_pos
            distance = np.linalg.norm(rel_pos)

            if distance > detection_range:
                continue

            # Angle check (must be looking down at target)
            if drone_pos[2] > target["position"][2]:  # Above target
                angle = np.arctan2(np.sqrt(rel_pos[0]**2 + rel_pos[1]**2), -rel_pos[2])
                if angle < detection_angle:
                    target["detected"] = True
                    self._detected_targets.append(target["id"])

                    # Start tracking if in tracking mode
                    if self.mission_type == "tracking":
                        self._tracking_target_idx = target["id"]

    def _get_observation(self) -> np.ndarray:
        """Get maritime patrol observation."""
        base_obs = super()._get_observation()

        # Relative position to current target/waypoint
        drone_pos = base_obs[:3]
        relative_target = self._target_position - drone_pos

        # Target detected flag
        any_detected = 1.0 if len(self._detected_targets) > 0 else 0.0

        # Fuel remaining
        fuel = self._fuel_remaining

        return np.concatenate([
            base_obs,
            relative_target,
            [fuel],
            [any_detected]
        ]).astype(np.float32)

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute maritime patrol reward."""
        pos = obs[:3]
        vel = obs[3:6]

        reward = 0.0

        # Update target positions
        self._update_targets()

        # Check for new detections
        self._check_target_detection(pos)

        # Update mission target
        self._update_mission_target()

        # Distance to current objective
        target_dist = np.linalg.norm(pos - self._target_position)

        # Progress toward objective
        reward += 1.0 - 0.01 * target_dist

        # Waypoint reached
        waypoint_threshold = 5.0
        if target_dist < waypoint_threshold:
            reward += 10.0
            self._current_patrol_idx = (self._current_patrol_idx + 1) % len(self._patrol_waypoints)
            self._update_mission_target()

        # Target detection bonus
        if len(self._detected_targets) > 0:
            reward += 50.0 * len(self._detected_targets)
            self._detected_targets = []  # Reset for next detection

        # Tracking bonus (staying close to tracked target)
        if self._tracking_target_idx is not None:
            tracking_dist = np.linalg.norm(
                pos[:2] - self._targets[self._tracking_target_idx]["position"][:2]
            )
            if tracking_dist < 10.0:
                reward += 2.0

        # Altitude maintenance
        optimal_altitude = 15.0
        altitude_error = abs(pos[2] - optimal_altitude)
        reward -= 0.1 * altitude_error

        # Fuel penalty
        self._fuel_remaining -= self._fuel_consumption_rate
        if self._fuel_remaining < 0.2:
            reward -= 5.0  # Low fuel warning

        # Energy efficiency
        reward -= 0.01 * np.sum(action**2)

        # Water crash penalty
        if pos[2] < 1.0:
            reward -= 100.0

        return float(reward)

    def _check_termination(self, obs: np.ndarray) -> Tuple[bool, bool]:
        """Check termination conditions."""
        pos = obs[:3]
        terminated = False

        # Water crash
        if pos[2] < 0.5:
            terminated = True

        # Out of search area (too high or too far)
        if pos[2] > 100:
            terminated = True
        if abs(pos[0]) > self.search_area_size or abs(pos[1]) > self.search_area_size:
            terminated = True

        # Out of fuel
        if self._fuel_remaining <= 0:
            terminated = True

        # Mission complete conditions
        if self.mission_type == "patrol":
            # Completed patrol circuit
            if self._current_patrol_idx == 0 and self._step_count > 100:
                pass  # Could terminate on complete circuit
        elif self.mission_type == "tracking":
            # All targets detected and tracked
            if all(t["detected"] for t in self._targets):
                terminated = True

        # Max steps
        max_steps = self.platform_config.get("max_episode_steps", 5000)
        truncated = self._step_count >= max_steps

        return terminated, truncated

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step."""
        obs, reward, terminated, truncated, info = super().step(action)

        info["fuel_remaining"] = self._fuel_remaining
        info["targets_detected"] = sum(1 for t in self._targets if t["detected"])
        info["total_targets"] = len(self._targets)
        info["patrol_progress"] = self._current_patrol_idx / max(len(self._patrol_waypoints), 1)

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        # Reset tracking state
        self._detected_targets = []
        self._tracking_target_idx = None
        self._current_patrol_idx = 0

        obs, info = super().reset(seed=seed, options=options)

        info["mission_type"] = self.mission_type
        info["num_targets"] = self.num_targets
        info["search_area_size"] = self.search_area_size

        return obs, info

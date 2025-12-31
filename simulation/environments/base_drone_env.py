"""
Base Drone Environment

Core simulation environment for autonomous drone training.
Uses PyBullet physics with environmental conditions support.
Designed for extensibility via subclassing for mission-specific environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
import pybullet as p
import pybullet_data

from simulation.environments.environmental_conditions import (
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    TerrainType,
    create_clear_day,
)


class BaseDroneEnv(gym.Env):
    """
    Base environment for autonomous drone training.

    This environment wraps PyBullet physics simulation and provides
    a Gymnasium-compatible interface for RL training. It supports:
    - Environmental conditions (weather, wind, lighting)
    - Multiple drone platforms via configuration
    - Extensible reward functions
    - Collision detection

    Subclass this for mission-specific environments (urban nav, etc.)

    Attributes:
        platform_config: Configuration dict for the drone platform
        env_conditions: Environmental conditions affecting flight
        render_mode: 'human' for GUI, None for headless
        physics_client: PyBullet physics client ID
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        platform_config: Dict[str, Any],
        environmental_conditions: Optional[EnvironmentalConditions] = None,
        render_mode: Optional[str] = None,
        physics_hz: int = 240,
        control_hz: int = 48,
        domain_randomization: bool = False,
    ):
        """
        Initialize the base drone environment.

        Args:
            platform_config: Drone platform configuration
            environmental_conditions: Weather, wind, etc. (defaults to clear day)
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            physics_hz: Physics simulation frequency
            control_hz: Control loop frequency
            domain_randomization: Enable randomization for sim-to-real transfer
        """
        super().__init__()

        self.platform_config = platform_config
        self.env_conditions = environmental_conditions or create_clear_day()
        self.render_mode = render_mode
        self.physics_hz = physics_hz
        self.control_hz = control_hz
        self.physics_steps_per_control = physics_hz // control_hz
        self.domain_randomization = domain_randomization

        # Physics simulation
        self.physics_client: Optional[int] = None
        self.drone_id: Optional[int] = None
        self.ground_id: Optional[int] = None

        # Obstacle tracking
        self._obstacle_ids: List[int] = []

        # State tracking
        self._step_count = 0
        self._episode_count = 0
        self._collision_count = 0
        self._previous_action: Optional[np.ndarray] = None

        # Cumulative episode statistics
        self._episode_stats: Dict[str, float] = {}

        # Define action space (motor commands normalized to [-1, 1])
        num_motors = platform_config.get("num_motors", 4)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_motors,),
            dtype=np.float32
        )

        # Define observation space
        # [position(3), velocity(3), orientation(4), angular_velocity(3)]
        # Extended observations can include target info, sensor data, etc.
        obs_dim = platform_config.get("observation_dim", 13)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Target position (can be overridden by subclasses)
        self._target_position = np.array([0.0, 0.0, 1.0])

        # Initialize physics
        self._init_physics()

    def _init_physics(self):
        """Initialize PyBullet physics simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / self.physics_hz, physicsClientId=self.physics_client)

        # Physics solver parameters
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / self.physics_hz,
            numSolverIterations=50,
            numSubSteps=4,
            physicsClientId=self.physics_client
        )

        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

    def _load_drone(self):
        """Load drone model into simulation."""
        # Starting position with optional randomization
        start_pos = [0, 0, 1.0]
        if self.domain_randomization:
            start_pos[0] += np.random.uniform(-0.5, 0.5)
            start_pos[1] += np.random.uniform(-0.5, 0.5)
            start_pos[2] += np.random.uniform(-0.2, 0.2)

        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # Create drone body
        # TODO: Load actual URDF based on platform_config
        arm_length = self.platform_config.get("arm_length", 0.15)

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[arm_length, arm_length, 0.02],
            physicsClientId=self.physics_client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[arm_length, arm_length, 0.02],
            rgbaColor=[0.2, 0.2, 0.8, 1.0],
            physicsClientId=self.physics_client
        )

        mass = self.platform_config.get("mass", 1.0)
        if self.domain_randomization:
            mass *= np.random.uniform(0.9, 1.1)

        self.drone_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos,
            baseOrientation=start_orientation,
            physicsClientId=self.physics_client
        )

        # Set drone dynamics
        p.changeDynamics(
            self.drone_id, -1,
            linearDamping=0.1,
            angularDamping=0.1,
            physicsClientId=self.physics_client
        )

    def _setup_environment(self):
        """
        Set up mission-specific environment elements.
        Override in subclasses to add obstacles, targets, etc.
        """
        pass

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        if self.drone_id is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        pos, orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self.physics_client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.drone_id, physicsClientId=self.physics_client
        )

        # Apply sensor noise based on environmental conditions
        pos = np.array(pos)
        vel = np.array(vel)
        orn = np.array(orn)
        ang_vel = np.array(ang_vel)

        # Add GPS noise
        gps_noise = self.env_conditions.get_sensor_noise_scale("gps")
        pos += np.random.normal(0, 0.01 * gps_noise, 3)

        # Add IMU noise
        imu_noise = self.env_conditions.get_sensor_noise_scale("imu")
        vel += np.random.normal(0, 0.01 * imu_noise, 3)
        ang_vel += np.random.normal(0, 0.01 * imu_noise, 3)

        obs = np.concatenate([pos, vel, orn, ang_vel]).astype(np.float32)

        return obs

    def _apply_action(self, action: np.ndarray):
        """Apply motor commands with environmental effects."""
        if self.drone_id is None:
            return

        # Get motor efficiency modifier from conditions
        efficiency = self.env_conditions.get_motor_efficiency_modifier()

        # Convert normalized actions to thrust forces
        max_thrust = self.platform_config.get("max_thrust_per_motor", 5.0)
        max_thrust *= efficiency

        # Map [-1,1] to [0, max_thrust]
        thrusts = (action + 1) / 2 * max_thrust
        total_thrust = np.sum(thrusts)

        # Apply thrust force in body frame
        p.applyExternalForce(
            self.drone_id,
            -1,  # Base link
            forceObj=[0, 0, total_thrust],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.physics_client
        )

        # Calculate differential thrust for torques (simplified quad model)
        if len(action) >= 4:
            arm_length = self.platform_config.get("arm_length", 0.15)

            # Roll torque (motors 0,3 vs 1,2)
            roll_torque = arm_length * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3])) * 0.5

            # Pitch torque (motors 0,1 vs 2,3)
            pitch_torque = arm_length * ((thrusts[2] + thrusts[3]) - (thrusts[0] + thrusts[1])) * 0.5

            # Yaw torque (CW vs CCW motors)
            yaw_coeff = 0.01
            yaw_torque = yaw_coeff * ((thrusts[0] + thrusts[2]) - (thrusts[1] + thrusts[3]))

            p.applyExternalTorque(
                self.drone_id,
                -1,
                torqueObj=[roll_torque, pitch_torque, yaw_torque],
                flags=p.LINK_FRAME,
                physicsClientId=self.physics_client
            )

        # Apply wind disturbance from environmental conditions
        wind_force = self.env_conditions.get_wind_vector(dt=1.0 / self.physics_hz)

        # Scale wind force by drone area and drag coefficient
        drag_mod = self.env_conditions.get_drag_coefficient_modifier()
        drag_coeff = self.platform_config.get("physics_params", {}).get("drag_coefficient", 0.1)
        effective_drag = drag_coeff * drag_mod

        wind_effect = wind_force * effective_drag * self.platform_config.get("mass", 1.0)

        p.applyExternalForce(
            self.drone_id,
            -1,
            forceObj=wind_effect.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Compute reward for current state-action pair.
        Override in subclasses for mission-specific rewards.

        Default: hover stability reward at target position.
        """
        pos = obs[:3]
        vel = obs[3:6]

        # Distance to target
        target_dist = np.linalg.norm(pos - self._target_position)

        # Compute reward components
        reward = 0.0

        # Alive bonus
        reward += 1.0

        # Position error penalty
        reward -= 0.5 * target_dist

        # Velocity penalty (encourage hovering)
        velocity_penalty = np.linalg.norm(vel)
        reward -= 0.2 * velocity_penalty

        # Action smoothness
        reward -= 0.01 * np.sum(action**2)

        # Bonus for being close to target
        if target_dist < 0.1:
            reward += 2.0
        elif target_dist < 0.3:
            reward += 1.0

        # Collision penalty
        if self._check_collision():
            reward -= 10.0
            self._collision_count += 1

        return float(reward)

    def _check_collision(self) -> bool:
        """Check if drone has collided with any obstacle."""
        if self.drone_id is None:
            return False

        # Check collision with ground
        contacts = p.getContactPoints(
            bodyA=self.drone_id,
            bodyB=self.ground_id,
            physicsClientId=self.physics_client
        )
        if len(contacts) > 0:
            return True

        # Check collision with obstacles
        for obs_id in self._obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self.drone_id,
                bodyB=obs_id,
                physicsClientId=self.physics_client
            )
            if len(contacts) > 0:
                return True

        return False

    def _check_termination(self, obs: np.ndarray) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        pos = obs[:3]

        # Terminated: crashed or flew out of bounds
        terminated = False

        # Ground collision (too low)
        if pos[2] < 0.05:
            terminated = True

        # Out of bounds
        bounds = self.platform_config.get("bounds", {"xy": 10, "z_max": 20})
        if isinstance(bounds, dict):
            xy_bound = bounds.get("xy", 10)
            z_max = bounds.get("z_max", 20)
        else:
            xy_bound = 10
            z_max = 20

        if abs(pos[0]) > xy_bound or abs(pos[1]) > xy_bound or pos[2] > z_max:
            terminated = True

        # Obstacle collision
        if self._check_collision():
            terminated = True

        # Truncated: max steps reached
        max_steps = self.platform_config.get("max_episode_steps", 1000)
        truncated = self._step_count >= max_steps

        return terminated, truncated

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_count += 1
        self._collision_count = 0
        self._previous_action = None
        self._episode_stats = {}

        # Reset environmental conditions
        self.env_conditions.reset()

        # Optionally randomize conditions
        if options and options.get("randomize_conditions", False):
            from simulation.environments.environmental_conditions import create_random_conditions
            difficulty = options.get("condition_difficulty", "medium")
            self.env_conditions = create_random_conditions(difficulty, seed)

        # Reset physics simulation
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / self.physics_hz, physicsClientId=self.physics_client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # Clear obstacles
        self._obstacle_ids = []

        # Load drone
        self._load_drone()

        # Set up mission-specific environment
        self._setup_environment()

        obs = self._get_observation()
        info = {
            "episode": self._episode_count,
            "conditions": self.env_conditions.to_dict(),
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        action = np.clip(action, -1.0, 1.0)

        # Apply action and step physics
        for _ in range(self.physics_steps_per_control):
            self._apply_action(action)
            p.stepSimulation(physicsClientId=self.physics_client)

        self._step_count += 1
        self._previous_action = action.copy()

        # Get new state
        obs = self._get_observation()
        reward = self._compute_reward(obs, action)
        terminated, truncated = self._check_termination(obs)

        # Update episode stats
        self._update_episode_stats(obs, action, reward)

        info = {
            "step": self._step_count,
            "position": obs[:3].tolist(),
            "velocity": obs[3:6].tolist(),
            "collisions": self._collision_count,
            "target_distance": float(np.linalg.norm(obs[:3] - self._target_position)),
        }

        # Add termination info
        if terminated or truncated:
            info["episode_stats"] = self._episode_stats

        return obs, reward, terminated, truncated, info

    def _update_episode_stats(self, obs: np.ndarray, action: np.ndarray, reward: float):
        """Update cumulative episode statistics."""
        pos = obs[:3]
        target_dist = np.linalg.norm(pos - self._target_position)

        # Update running averages
        if "avg_position_error" not in self._episode_stats:
            self._episode_stats["avg_position_error"] = target_dist
            self._episode_stats["total_reward"] = reward
            self._episode_stats["min_target_distance"] = target_dist
        else:
            n = self._step_count
            self._episode_stats["avg_position_error"] = (
                (self._episode_stats["avg_position_error"] * (n - 1) + target_dist) / n
            )
            self._episode_stats["total_reward"] += reward
            self._episode_stats["min_target_distance"] = min(
                self._episode_stats["min_target_distance"], target_dist
            )

        # Track hover time (within 0.5m of target)
        if target_dist < 0.5:
            dt = 1.0 / self.control_hz
            self._episode_stats["hover_time"] = self._episode_stats.get("hover_time", 0) + dt

        self._episode_stats["survival_time"] = self._step_count / self.control_hz
        self._episode_stats["total_collisions"] = self._collision_count

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 1],
                distance=3,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100,
                physicsClientId=self.physics_client
            )
            _, _, rgba, _, _ = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.physics_client
            )
            return np.array(rgba)[:, :, :3]
        return None

    def close(self):
        """Clean up environment resources."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_target_position(self, position: np.ndarray):
        """Set target position for the drone."""
        self._target_position = np.array(position)

    def set_conditions(self, conditions: EnvironmentalConditions):
        """Update environmental conditions."""
        self.env_conditions = conditions

    def add_obstacle(
        self,
        shape: str,
        position: Tuple[float, float, float],
        size: Tuple[float, ...],
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    ) -> int:
        """
        Add an obstacle to the environment.

        Args:
            shape: 'box', 'sphere', or 'cylinder'
            position: (x, y, z) position
            size: Shape-specific size (half_extents for box, radius for sphere, etc.)
            color: RGBA color

        Returns:
            Obstacle ID
        """
        if shape == "box":
            collision_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=size, physicsClientId=self.physics_client
            )
            visual_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=size, rgbaColor=color,
                physicsClientId=self.physics_client
            )
        elif shape == "sphere":
            collision_id = p.createCollisionShape(
                p.GEOM_SPHERE, radius=size[0], physicsClientId=self.physics_client
            )
            visual_id = p.createVisualShape(
                p.GEOM_SPHERE, radius=size[0], rgbaColor=color,
                physicsClientId=self.physics_client
            )
        elif shape == "cylinder":
            collision_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size[0], height=size[1],
                physicsClientId=self.physics_client
            )
            visual_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=color,
                physicsClientId=self.physics_client
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")

        obstacle_id = p.createMultiBody(
            baseMass=0,  # Static obstacle
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            physicsClientId=self.physics_client
        )

        self._obstacle_ids.append(obstacle_id)
        return obstacle_id

    def get_telemetry(self) -> Dict[str, Any]:
        """Get detailed telemetry data."""
        obs = self._get_observation()
        return {
            "position": obs[:3].tolist(),
            "velocity": obs[3:6].tolist(),
            "orientation_quat": obs[6:10].tolist(),
            "angular_velocity": obs[10:13].tolist(),
            "target_position": self._target_position.tolist(),
            "target_distance": float(np.linalg.norm(obs[:3] - self._target_position)),
            "step": self._step_count,
            "episode": self._episode_count,
            "collisions": self._collision_count,
            "conditions": self.env_conditions.to_dict(),
        }


# Backwards compatibility alias
EnvironmentalSimulator = BaseDroneEnv

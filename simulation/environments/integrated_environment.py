"""
Complete Integration Example

Shows how to integrate gym-pybullet-drones validated dynamics
with your existing platform's environmental conditions and training infrastructure.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

# Your existing imports
try:
    from simulation.environments.environmental_conditions import (
        EnvironmentalConditions,
        WeatherType,
        TimeOfDay,
        create_urban_conditions
    )
    HAVE_ENV_CONDITIONS = True
except ImportError:
    HAVE_ENV_CONDITIONS = False
    print("Warning: Environmental conditions not available")

# New validated dynamics (corrected import path)
from simulation.physics.validated_dynamics import (
    ValidatedDroneEnvironment,
    DroneType,
    GYM_PYBULLET_DRONES_AVAILABLE
)

# New controllers
from simulation.control.pid_controller import CascadedDroneController


class IntegratedDroneEnvironment(gym.Env):
    """
    Complete integrated environment combining:
    1. Validated dynamics (gym-pybullet-drones)
    2. Environmental conditions (your platform)
    3. PID controllers (industry standard)
    4. RL training interface (Gymnasium)

    This is the recommended environment for training.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        drone_type: DroneType = DroneType.CRAZYFLIE_2X,
        task: str = "hover",  # "hover", "waypoint", "trajectory"
        environmental_conditions: Optional[EnvironmentalConditions] = None,
        use_pid_baseline: bool = False,  # Use PID as baseline or train RL
        physics_hz: int = 240,
        control_hz: int = 48,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize integrated environment.

        Args:
            drone_type: Type of drone to simulate
            task: Task type ("hover", "waypoint", "trajectory")
            environmental_conditions: Weather/wind conditions
            use_pid_baseline: If True, use PID controller (for comparison)
            physics_hz: Physics simulation frequency
            control_hz: Control loop frequency
            render_mode: Render mode for visualization
        """
        super().__init__()

        if not GYM_PYBULLET_DRONES_AVAILABLE:
            raise ImportError("gym-pybullet-drones required")

        self.drone_type = drone_type
        self.task = task
        self.use_pid_baseline = use_pid_baseline
        self.render_mode = render_mode

        # Environmental conditions
        if environmental_conditions is None and HAVE_ENV_CONDITIONS:
            self.env_conditions = create_urban_conditions()
        else:
            self.env_conditions = environmental_conditions

        # Create validated dynamics environment
        self.dynamics_env = ValidatedDroneEnvironment(
            drone_type=drone_type,
            environmental_conditions=self.env_conditions,
            physics_hz=physics_hz,
            control_hz=control_hz,
            gui=(render_mode == "human"),
        )

        # PID controller (for baseline comparison)
        if self.use_pid_baseline:
            params = self.dynamics_env.URDF_PARAMS
            self.pid_controller = CascadedDroneController(
                mass=params['m'],
                arm_length=params['l'],
                ixx=params['ixx'],
                iyy=params['iyy'],
                izz=params['izz'],
                kf=params['kf'],
                km=params['km'],
            )

        # Define action space
        if self.use_pid_baseline:
            # High-level actions: target position + yaw
            self.action_space = spaces.Box(
                low=np.array([-5, -5, 0, -np.pi]),
                high=np.array([5, 5, 3, np.pi]),
                dtype=np.float32
            )
        else:
            # Low-level actions: motor RPMs
            self.action_space = spaces.Box(
                low=np.array([0, 0, 0, 0]),
                high=np.array([21702, 21702, 21702, 21702]),  # Max Crazyflie RPM
                dtype=np.float32
            )

        # Define observation space
        # [pos(3), vel(3), rpy(3), ang_vel(3), target_pos(3), target_yaw(1)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

        # Task parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        self.target_pos = np.array([0, 0, 1])  # Default hover target
        self.target_yaw = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ):
        """Reset environment."""
        super().reset(seed=seed)

        self.current_step = 0

        # Reset dynamics
        obs_dict = self.dynamics_env.reset()

        # Set task-specific target
        if self.task == "hover":
            self.target_pos = np.array([0, 0, 1])
            self.target_yaw = 0.0
        elif self.task == "waypoint":
            # Random waypoint
            self.target_pos = np.array([
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(0.5, 2)
            ])
            self.target_yaw = np.random.uniform(-np.pi, np.pi)

        # Reset PID if using
        if self.use_pid_baseline:
            self.pid_controller.reset()

        # Construct observation
        obs = self._get_observation()
        info = {"step": self.current_step}

        return obs, info

    def step(self, action):
        """Step environment."""
        self.current_step += 1

        # Convert action
        if self.use_pid_baseline:
            # Action is target position + yaw
            target_pos = action[:3]
            target_yaw = action[3]

            # Get current state
            state = self._get_full_state()

            # Compute motor commands via PID
            motor_rpm = self.pid_controller.compute_control(
                state,
                target_pos,
                target_yaw,
                dt=1/48
            )
        else:
            # Action is direct motor RPM
            motor_rpm = action

        # Apply action to dynamics (convert RPM to normalized [0,1])
        max_rpm = 21702
        normalized_rpm = np.clip(motor_rpm / max_rpm, 0, 1)

        # Step dynamics
        obs_dict, reward, terminated, truncated, info = self.dynamics_env.step(
            np.array([normalized_rpm])  # Shape: (1, 4)
        )

        # Apply environmental disturbances
        if self.env_conditions is not None:
            wind = self.env_conditions.get_wind_vector()
            # Wind is applied inside dynamics_env already

        # Compute custom reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_termination()
        truncated = (self.current_step >= self.max_episode_steps)

        # Construct observation
        obs = self._get_observation()

        info.update({
            "step": self.current_step,
            "success": self._check_success(),
        })

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get observation vector."""
        # Get state from dynamics
        state = self._get_full_state()

        pos = state[0:3]
        vel = state[3:6]
        rpy = state[6:9]
        ang_vel = state[9:12]

        # Construct observation
        obs = np.concatenate([
            pos,
            vel,
            rpy,
            ang_vel,
            self.target_pos,
            [self.target_yaw]
        ])

        return obs.astype(np.float32)

    def _get_full_state(self) -> np.ndarray:
        """Get full 12D state from dynamics environment."""
        # Extract from gym-pybullet-drones
        pos = self.dynamics_env._getDroneStateVector(0)[0:3]
        vel = self.dynamics_env._getDroneStateVector(0)[10:13]
        quat = self.dynamics_env._getDroneStateVector(0)[3:7]
        ang_vel = self.dynamics_env._getDroneStateVector(0)[13:16]

        # Convert quaternion to roll-pitch-yaw
        rpy = self._quat_to_rpy(quat)

        state = np.concatenate([pos, vel, rpy, ang_vel])
        return state

    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to roll-pitch-yaw."""
        import pybullet as p
        rpy = p.getEulerFromQuaternion(quat)
        return np.array(rpy)

    def _compute_reward(self) -> float:
        """Compute reward based on task."""
        state = self._get_full_state()
        pos = state[0:3]
        vel = state[3:6]
        rpy = state[6:9]

        if self.task == "hover":
            # Hover reward: minimize position error and velocity
            pos_error = np.linalg.norm(pos - self.target_pos)
            vel_magnitude = np.linalg.norm(vel)

            reward = (
                1.0  # Alive bonus
                - 1.0 * pos_error  # Position error
                - 0.5 * vel_magnitude  # Velocity penalty
                - 0.1 * (abs(rpy[0]) + abs(rpy[1]))  # Tilt penalty
            )

        elif self.task == "waypoint":
            # Waypoint reward: progress toward target
            distance = np.linalg.norm(pos - self.target_pos)

            reward = -0.5 * distance

            # Success bonus
            if distance < 0.2:  # Within 20cm
                reward += 100.0

        else:
            reward = 0.0

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if episode should terminate (crash)."""
        state = self._get_full_state()
        pos = state[0:3]
        rpy = state[6:9]

        # Crashed (too low or flipped)
        if pos[2] < 0.05:
            return True

        # Flipped over
        if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
            return True

        # Flew too far
        if abs(pos[0]) > 5 or abs(pos[1]) > 5 or pos[2] > 5:
            return True

        return False

    def _check_success(self) -> bool:
        """Check if task was successful."""
        state = self._get_full_state()
        pos = state[0:3]

        distance = np.linalg.norm(pos - self.target_pos)
        return distance < 0.2  # Within 20cm

    def render(self):
        """Render environment."""
        if self.render_mode == "human":
            # GUI is already shown by dynamics_env
            pass
        return None

    def close(self):
        """Clean up."""
        self.dynamics_env.close()


# ============================================================================
# Example Usage
# ============================================================================

def train_hover_with_validated_dynamics():
    """Example: Train hover task with validated dynamics."""
    print("Training Hover Task with Validated Dynamics")
    print("=" * 60)

    # Create environment with environmental conditions
    if HAVE_ENV_CONDITIONS:
        env_conditions = create_urban_conditions()
    else:
        env_conditions = None

    env = IntegratedDroneEnvironment(
        drone_type=DroneType.CRAZYFLIE_2X,
        task="hover",
        environmental_conditions=env_conditions,
        use_pid_baseline=False,  # Train RL policy
        render_mode=None,  # No GUI for training
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Train with Stable-Baselines3
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )

    print("\nStarting training...")
    model.learn(total_timesteps=100_000)

    # Save model
    model.save("hover_policy_validated")
    print("Model saved!")

    # Evaluate
    print("\nEvaluating...")
    obs, info = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Success: {info.get('success', False)}")
            break

    env.close()


def compare_rl_vs_pid():
    """Compare RL policy vs PID controller."""
    print("Comparing RL vs PID Controller")
    print("=" * 60)

    # Test PID controller
    env_pid = IntegratedDroneEnvironment(
        drone_type=DroneType.CRAZYFLIE_2X,
        task="hover",
        use_pid_baseline=True,  # Use PID
        render_mode="human",
    )

    print("Testing PID controller...")
    obs, info = env_pid.reset()

    for i in range(500):
        # PID action: target position + yaw
        action = np.array([0, 0, 1, 0])  # Hover at (0, 0, 1)
        obs, reward, term, trunc, info = env_pid.step(action)

        if i % 100 == 0:
            pos = obs[0:3]
            print(f"Step {i}: Position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    env_pid.close()

    # TODO: Compare with trained RL policy


if __name__ == "__main__":
    # Check dependencies
    if not GYM_PYBULLET_DRONES_AVAILABLE:
        print("Error: gym-pybullet-drones not installed")
        print("Install with: pip install gym-pybullet-drones")
        exit(1)

    print("Integrated Drone Environment - Ready!")
    print("\nAvailable functions:")
    print("  - train_hover_with_validated_dynamics()")
    print("  - compare_rl_vs_pid()")

    # Uncomment to run:
    # train_hover_with_validated_dynamics()
    # compare_rl_vs_pid()

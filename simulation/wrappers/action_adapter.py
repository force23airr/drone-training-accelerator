"""
Action Adapter Wrapper

Maps high-level policy actions to low-level motor commands using
a cascaded PID controller. This enables separation between
high-level policies and low-level control.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.control import CascadedDroneController, ActionMode


@dataclass
class ActionAdapterConfig:
    """Configuration for action-space adaptation."""
    action_mode: ActionMode = ActionMode.MOTOR_THRUSTS

    # Limits for normalized action mapping
    max_tilt_deg: float = 35.0
    max_rate_deg_s: float = 180.0
    max_yaw_rate_deg_s: float = 180.0
    max_horizontal_speed_m_s: float = 8.0
    max_vertical_speed_m_s: float = 3.0

    # Throttle mapping (normalized action -> throttle fraction)
    min_throttle: float = 0.1
    max_throttle: float = 1.0

    # Velocity action format
    include_yaw_in_velocity: bool = False

    # Controller timing
    control_hz: Optional[float] = None

    # Logging
    log_actions: bool = False


class ActionAdapterWrapper(gym.Wrapper):
    """
    Wraps an environment to expose a higher-level action space and
    converts actions to low-level motor commands.

    Supported action modes:
    - motor_thrusts: raw motor commands (passthrough)
    - attitude_rates: [roll_rate, pitch_rate, yaw_rate, thrust]
    - attitude: [roll, pitch, yaw, thrust]
    - velocity: [vx, vy, vz] (optionally yaw)
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[ActionAdapterConfig] = None,
        state_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(env)
        self.config = config or ActionAdapterConfig()
        self.action_mode = self.config.action_mode
        self._state_extractor = state_extractor or self._default_state_extractor
        self._last_obs: Optional[np.ndarray] = None

        self._controller: Optional[CascadedDroneController] = None
        self._num_motors = None
        self._max_rpm = None
        self._max_total_thrust = None
        self._mass = None

        if self.action_mode == ActionMode.MOTOR_THRUSTS:
            self.action_space = env.action_space
        else:
            self._init_controller()
            self.action_space = self._build_action_space()

        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._controller is not None:
            self._controller.reset()
        self._last_obs = obs
        return obs, info

    def step(self, action):
        if self.action_mode == ActionMode.MOTOR_THRUSTS:
            motor_action = action
        else:
            if self._last_obs is None:
                raise RuntimeError("ActionAdapterWrapper received action before reset().")
            state = self._state_extractor(self._last_obs)
            motor_action = self._convert_action_to_motor(action, state)

        obs, reward, terminated, truncated, info = self.env.step(motor_action)
        self._last_obs = obs

        if self.config.log_actions:
            info["action_adapter"] = {
                "action_mode": self.action_mode.value,
                "raw_action": np.array(action, dtype=np.float32).tolist(),
                "motor_action": np.array(motor_action, dtype=np.float32).tolist(),
            }

        return obs, reward, terminated, truncated, info

    def _build_action_space(self) -> spaces.Box:
        if self.action_mode in (ActionMode.ATTITUDE, ActionMode.ATTITUDE_RATES):
            return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        if self.action_mode == ActionMode.VELOCITY:
            dim = 4 if self.config.include_yaw_in_velocity else 3
            return spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
        return self.env.action_space

    def _init_controller(self):
        if not hasattr(self.env, "platform_config"):
            raise ValueError("ActionAdapterWrapper requires env.platform_config for control mapping.")

        platform = self.env.platform_config
        self._num_motors = platform.get("num_motors", 4)
        if self._num_motors != 4:
            raise ValueError(
                "ActionAdapterWrapper only supports 4-motor multirotors for high-level actions."
            )

        physics_params = platform.get("physics_params", {})
        self._mass = platform.get("mass", 1.0)
        arm_length = platform.get("arm_length", 0.1)

        self._controller = CascadedDroneController(
            mass=self._mass,
            arm_length=arm_length,
            ixx=physics_params.get("ixx", 1.4e-5),
            iyy=physics_params.get("iyy", 1.4e-5),
            izz=physics_params.get("izz", 2.17e-5),
            kf=physics_params.get("kf", 3.16e-10),
            km=physics_params.get("km", 7.94e-12),
        )

        self._max_rpm = platform.get("max_rpm", 10000)
        max_thrust_per_motor = platform.get("max_thrust_per_motor", None)
        if max_thrust_per_motor is not None:
            self._max_total_thrust = max_thrust_per_motor * self._num_motors
        else:
            # Fallback: assume 2g max thrust if not specified
            self._max_total_thrust = self._mass * 9.81 * 2.0

    def _convert_action_to_motor(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        action = np.array(action, dtype=np.float32).flatten()
        cfg = self.config

        dt = 1.0 / (cfg.control_hz or getattr(self.env, "control_hz", 48.0))

        if self.action_mode == ActionMode.ATTITUDE:
            roll = action[0] * np.deg2rad(cfg.max_tilt_deg)
            pitch = action[1] * np.deg2rad(cfg.max_tilt_deg)
            yaw = action[2] * np.pi
            thrust = self._throttle_to_thrust(self._scale_throttle(action[3]))

            motor_rpm = self._controller.compute_attitude_control(
                state=state,
                target_thrust=thrust,
                target_rpy=np.array([roll, pitch, yaw]),
                dt=dt,
            )
        elif self.action_mode == ActionMode.ATTITUDE_RATES:
            roll_rate = action[0] * np.deg2rad(cfg.max_rate_deg_s)
            pitch_rate = action[1] * np.deg2rad(cfg.max_rate_deg_s)
            yaw_rate = action[2] * np.deg2rad(cfg.max_yaw_rate_deg_s)
            thrust = self._throttle_to_thrust(self._scale_throttle(action[3]))

            motor_rpm = self._controller.compute_rate_control(
                state=state,
                target_rates=np.array([roll_rate, pitch_rate, yaw_rate]),
                target_thrust=thrust,
                dt=dt,
            )
        elif self.action_mode == ActionMode.VELOCITY:
            vx = action[0] * cfg.max_horizontal_speed_m_s
            vy = action[1] * cfg.max_horizontal_speed_m_s
            vz = action[2] * cfg.max_vertical_speed_m_s

            if cfg.include_yaw_in_velocity and action.shape[0] >= 4:
                target_yaw = action[3] * np.pi
            else:
                target_yaw = state[8]

            motor_rpm = self._controller.compute_velocity_control(
                state=state,
                target_vel=np.array([vx, vy, vz]),
                target_yaw=target_yaw,
                dt=dt,
            )
        else:
            motor_rpm = action

        return self._rpm_to_normalized(motor_rpm)

    def _scale_throttle(self, normalized: float) -> float:
        cfg = self.config
        return (normalized + 1.0) * 0.5 * (cfg.max_throttle - cfg.min_throttle) + cfg.min_throttle

    def _throttle_to_thrust(self, throttle: float) -> float:
        throttle = float(np.clip(throttle, 0.0, 1.0))
        return throttle * self._max_total_thrust

    def _rpm_to_normalized(self, motor_rpm: np.ndarray) -> np.ndarray:
        motor_rpm = np.array(motor_rpm, dtype=np.float32)
        max_rpm = float(self._max_rpm or 1.0)
        rpm_ratio = np.clip(motor_rpm / max_rpm, 0.0, 1.0)
        # Thrust proportional to rpm^2; map to normalized [-1, 1]
        thrust_ratio = rpm_ratio ** 2
        return np.clip(thrust_ratio * 2.0 - 1.0, -1.0, 1.0)

    def _default_state_extractor(self, obs: np.ndarray) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32).flatten()

        if obs.shape[0] < 9:
            raise ValueError("Observation too small for state extraction.")

        position = obs[0:3]
        velocity = obs[3:6]

        # Orientation: quaternion (len>=10) or Euler (len>=9)
        if obs.shape[0] >= 10:
            orientation = obs[6:10]
            if orientation.shape[0] == 4 and self._looks_like_quaternion(orientation):
                roll, pitch, yaw = self._quat_to_rpy(orientation)
            else:
                roll, pitch, yaw = orientation[0:3]
        else:
            roll, pitch, yaw = obs[6:9]

        if obs.shape[0] >= 13:
            ang_vel = obs[10:13]
        else:
            ang_vel = np.zeros(3, dtype=np.float32)

        return np.concatenate([position, velocity, [roll, pitch, yaw], ang_vel])

    def _quat_to_rpy(self, q: np.ndarray) -> Tuple[float, float, float]:
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        return roll, pitch, yaw

    def _looks_like_quaternion(self, q: np.ndarray) -> bool:
        norm = np.linalg.norm(q)
        return 0.5 <= norm <= 1.5


def make_action_adapted_env(
    env: gym.Env,
    config: Optional[ActionAdapterConfig] = None,
    state_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> ActionAdapterWrapper:
    """Convenience function to wrap environment with action adapter."""
    return ActionAdapterWrapper(env, config=config, state_extractor=state_extractor)

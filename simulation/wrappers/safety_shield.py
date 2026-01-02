"""
Safety Shield Wrapper

Last line of defense for deployable policies. Enforces hard safety constraints
that cannot be violated, regardless of what the policy outputs.

Implements:
- Tilt limits (max roll/pitch angle)
- Descent rate limits
- Minimum altitude floor
- Geofence (position bounds)
- Optional guardian controller takeover on risk triggers
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Callable
from enum import Enum

from simulation.control import ActionMode


class ShieldAction(Enum):
    """Action taken by the safety shield."""
    NONE = "none"              # No intervention
    CLAMP = "clamp"            # Action clamped to limits
    OVERRIDE = "override"      # Action replaced by guardian
    EMERGENCY = "emergency"    # Emergency stop triggered


@dataclass
class SafetyLimits:
    """Hard safety limits that cannot be violated."""
    # Attitude limits
    max_tilt_deg: float = 45.0              # Max roll/pitch angle
    max_tilt_rate_deg_s: float = 180.0      # Max tilt rate

    # Altitude limits
    min_altitude_m: float = 0.3             # Minimum altitude above ground
    max_altitude_m: float = 120.0           # Maximum altitude (legal limit in many countries)
    max_descent_rate_m_s: float = 3.0       # Max descent rate

    # Velocity limits
    max_horizontal_speed_m_s: float = 15.0  # Max horizontal speed
    max_vertical_speed_m_s: float = 5.0     # Max vertical speed

    # Geofence (rectangular bounds)
    geofence_x_min: float = -100.0
    geofence_x_max: float = 100.0
    geofence_y_min: float = -100.0
    geofence_y_max: float = 100.0
    geofence_margin_m: float = 5.0          # Soft boundary before hard limit

    # Motor limits
    min_throttle: float = 0.1               # Minimum throttle (prevent free-fall)
    max_throttle: float = 1.0               # Maximum throttle


@dataclass
class ShieldConfig:
    """
    Configuration for safety shield behavior.

    IMPORTANT: Reward penalties are OFF by default.
    The shield enforces safety and logs interventions, but the trainer
    decides how to penalize interventions (so you can tune per algorithm/task).

    If you want reward shaping, set apply_penalties=True.
    """
    limits: SafetyLimits = field(default_factory=SafetyLimits)

    # Intervention settings
    enable_clamp: bool = True               # Enable action clamping
    enable_guardian: bool = True            # Enable guardian controller takeover
    enable_emergency_stop: bool = True      # Enable emergency stop

    # Guardian takeover thresholds
    guardian_tilt_threshold_deg: float = 40.0    # Takeover if tilt exceeds this
    guardian_altitude_margin_m: float = 1.0      # Takeover this close to min altitude
    guardian_geofence_margin_m: float = 10.0     # Takeover this close to geofence

    # Penalties (for RL training) - OFF by default
    # The shield logs interventions; the trainer decides on penalties
    apply_penalties: bool = False           # Set True to enable reward penalties
    clamp_penalty: float = -0.1             # Penalty per clamped action
    guardian_penalty: float = -1.0          # Penalty for guardian takeover
    emergency_penalty: float = -10.0        # Penalty for emergency stop

    # Logging
    log_interventions: bool = True

    # Action interface
    action_mode: Optional[ActionMode] = None  # Auto-detect if None


@dataclass
class ShieldState:
    """Current state of the safety shield."""
    action_taken: ShieldAction = ShieldAction.NONE
    original_action: Optional[np.ndarray] = None
    modified_action: Optional[np.ndarray] = None
    intervention_reason: str = ""
    intervention_count: int = 0
    guardian_active: bool = False


class SafetyShieldWrapper(gym.Wrapper):
    """
    Safety shield wrapper that enforces hard safety constraints.

    This is the last line of defense - it ensures that certain safety
    invariants can never be violated regardless of what the policy outputs.

    Features:
    - Clamps actions to safe limits
    - Activates guardian controller when approaching danger zones
    - Triggers emergency stop in critical situations
    - Applies penalties for interventions (for RL training)
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[ShieldConfig] = None,
        guardian_controller: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            env: Base environment
            config: Shield configuration
            guardian_controller: Optional guardian controller function
                                 Takes observation, returns safe action
        """
        super().__init__(env)
        self.config = config or ShieldConfig()
        self.action_mode = self._resolve_action_mode(self.config.action_mode)
        self.guardian_controller = guardian_controller or self._default_guardian

        self._state = ShieldState()
        self._step_count = 0
        self._total_interventions = 0
        self._intervention_history: List[Dict[str, Any]] = []

        self._num_motors = None
        if hasattr(env, "platform_config"):
            self._num_motors = env.platform_config.get("num_motors")

    @property
    def shield_state(self) -> ShieldState:
        """Get current shield state."""
        return self._state

    @property
    def intervention_count(self) -> int:
        """Total number of interventions."""
        return self._total_interventions

    def _resolve_action_mode(self, mode: Optional[ActionMode]) -> ActionMode:
        """Resolve action mode from config or environment."""
        if mode is not None:
            if isinstance(mode, ActionMode):
                return mode
            try:
                return ActionMode(mode)
            except ValueError:
                pass

        env_mode = getattr(self.env, "action_mode", None)
        if env_mode is not None:
            if isinstance(env_mode, ActionMode):
                return env_mode
            try:
                return ActionMode(env_mode)
            except ValueError:
                pass

        # Infer from action space and platform config when possible
        action_shape = getattr(getattr(self.env, "action_space", None), "shape", None)
        num_motors = None
        if hasattr(self.env, "platform_config"):
            num_motors = self.env.platform_config.get("num_motors")

        if action_shape and num_motors and action_shape == (num_motors,):
            return ActionMode.MOTOR_THRUSTS
        if action_shape == (3,):
            return ActionMode.VELOCITY

        return ActionMode.ATTITUDE

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and shield state."""
        self._state = ShieldState()
        self._step_count = 0
        self._intervention_history = []

        obs, info = self.env.reset(seed=seed, options=options)
        info['shield'] = {'active': True, 'interventions': 0}

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step with safety shield enforcement.

        Args:
            action: Policy action

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        action = np.array(action, dtype=np.float32)

        # Reset shield state for this step
        self._state = ShieldState(original_action=action.copy())

        # Get current state estimate from observation
        obs_before = self._get_current_obs()

        # Apply safety checks in order of severity
        safe_action, penalty = self._apply_safety_checks(action, obs_before)

        # Step environment with safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # Apply intervention penalty ONLY if enabled
        # By default, shield logs but doesn't modify reward
        if self.config.apply_penalties:
            reward += penalty

        # Check for post-step violations (environment might have dynamics we can't predict)
        if self._detect_post_step_violation(obs):
            # Could trigger emergency stop here if needed
            pass

        # Add shield info
        info['shield'] = {
            'action_taken': self._state.action_taken.value,
            'intervention_reason': self._state.intervention_reason,
            'guardian_active': self._state.guardian_active,
            'intervention_count': self._total_interventions,
            'penalty': penalty,
        }

        return obs, reward, terminated, truncated, info

    def _apply_safety_checks(
        self,
        action: np.ndarray,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Apply safety checks and return safe action.

        Returns:
            (safe_action, penalty)
        """
        penalty = 0.0
        safe_action = action.copy()
        cfg = self.config
        limits = cfg.limits

        # Extract state from observation (assumes standard format)
        position, velocity, orientation, angular_vel = self._parse_observation(obs)

        # 1. Check for emergency conditions
        if self._is_emergency(position, velocity, orientation):
            if cfg.enable_emergency_stop:
                safe_action = self._emergency_stop_action()
                self._state.action_taken = ShieldAction.EMERGENCY
                self._state.intervention_reason = "Emergency stop triggered"
                self._log_intervention("emergency")
                return safe_action, cfg.emergency_penalty

        # 2. Check if guardian should take over
        if cfg.enable_guardian and self._should_guardian_takeover(position, velocity, orientation):
            safe_action = self.guardian_controller(obs)
            self._state.action_taken = ShieldAction.OVERRIDE
            self._state.guardian_active = True
            self._state.intervention_reason = "Guardian takeover"
            self._log_intervention("guardian")
            penalty = cfg.guardian_penalty

        # 3. Clamp action to safe limits
        elif cfg.enable_clamp:
            clamped_action, was_clamped, clamp_reason = self._clamp_action(
                safe_action, position, velocity, orientation
            )
            if was_clamped:
                safe_action = clamped_action
                self._state.action_taken = ShieldAction.CLAMP
                self._state.intervention_reason = clamp_reason
                self._log_intervention("clamp")
                penalty = cfg.clamp_penalty

        self._state.modified_action = safe_action.copy()
        return safe_action, penalty

    def _is_emergency(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
    ) -> bool:
        """Check for emergency conditions requiring immediate stop."""
        limits = self.config.limits

        # Extreme tilt (about to flip)
        tilt = self._compute_tilt(orientation)
        if tilt > 80:  # Very close to flipping
            return True

        # Below minimum altitude with high descent rate
        if position[2] < limits.min_altitude_m * 0.5:
            if velocity[2] < -limits.max_descent_rate_m_s * 1.5:
                return True

        # Way outside geofence
        margin = limits.geofence_margin_m * 3
        if (position[0] < limits.geofence_x_min - margin or
            position[0] > limits.geofence_x_max + margin or
            position[1] < limits.geofence_y_min - margin or
            position[1] > limits.geofence_y_max + margin):
            return True

        return False

    def _should_guardian_takeover(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
    ) -> bool:
        """Check if guardian controller should take over."""
        cfg = self.config
        limits = cfg.limits

        # High tilt angle
        tilt = self._compute_tilt(orientation)
        if tilt > cfg.guardian_tilt_threshold_deg:
            return True

        # Close to minimum altitude
        if position[2] < limits.min_altitude_m + cfg.guardian_altitude_margin_m:
            return True

        # Close to geofence
        margin = cfg.guardian_geofence_margin_m
        if (position[0] < limits.geofence_x_min + margin or
            position[0] > limits.geofence_x_max - margin or
            position[1] < limits.geofence_y_min + margin or
            position[1] > limits.geofence_y_max - margin):
            return True

        return False

    def _clamp_action(
        self,
        action: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
    ) -> Tuple[np.ndarray, bool, str]:
        """
        Clamp action to safe limits.

        Returns:
            (clamped_action, was_clamped, reason)
        """
        limits = self.config.limits
        clamped = action.copy()
        was_clamped = False
        reasons = []
        mode = self.action_mode

        if mode == ActionMode.MOTOR_THRUSTS:
            clamped = np.clip(clamped, -1.0, 1.0)

            # Near minimum altitude: enforce minimum thrust
            if position[2] < limits.min_altitude_m + 0.5:
                min_action = self._action_from_throttle(limits.min_throttle + 0.2)
                if np.any(clamped < min_action):
                    clamped = np.maximum(clamped, min_action)
                    was_clamped = True
                    reasons.append("min_altitude_thrust")

            # Near maximum altitude: limit climb
            if position[2] > limits.max_altitude_m - 1.0:
                max_action = self._action_from_throttle(min(0.5, limits.max_throttle))
                if np.any(clamped > max_action):
                    clamped = np.minimum(clamped, max_action)
                    was_clamped = True
                    reasons.append("max_altitude_thrust")

        elif mode == ActionMode.ATTITUDE and len(action) >= 4:
            max_tilt_rad = np.deg2rad(limits.max_tilt_deg)

            roll = np.clip(action[0] * max_tilt_rad, -max_tilt_rad, max_tilt_rad)
            pitch = np.clip(action[1] * max_tilt_rad, -max_tilt_rad, max_tilt_rad)
            yaw = np.clip(action[2] * np.pi, -np.pi, np.pi)
            throttle = self._throttle_from_action(action[3])

            throttle, throttle_reasons = self._clamp_throttle_for_altitude(throttle, position)
            reasons.extend(throttle_reasons)
            if throttle_reasons:
                was_clamped = True

            clamped[0] = roll / max_tilt_rad
            clamped[1] = pitch / max_tilt_rad
            clamped[2] = yaw / np.pi
            clamped[3] = self._action_from_throttle(throttle)

            if np.any(clamped != action):
                was_clamped = True
                if abs(action[0]) > 1.0:
                    reasons.append("roll")
                if abs(action[1]) > 1.0:
                    reasons.append("pitch")

        elif mode == ActionMode.ATTITUDE_RATES and len(action) >= 4:
            max_rate = np.deg2rad(limits.max_tilt_rate_deg_s)
            max_yaw_rate = np.deg2rad(limits.max_tilt_rate_deg_s)

            roll_rate = np.clip(action[0] * max_rate, -max_rate, max_rate)
            pitch_rate = np.clip(action[1] * max_rate, -max_rate, max_rate)
            yaw_rate = np.clip(action[2] * max_yaw_rate, -max_yaw_rate, max_yaw_rate)
            throttle = self._throttle_from_action(action[3])

            throttle, throttle_reasons = self._clamp_throttle_for_altitude(throttle, position)
            reasons.extend(throttle_reasons)
            if throttle_reasons:
                was_clamped = True

            clamped[0] = roll_rate / max_rate
            clamped[1] = pitch_rate / max_rate
            clamped[2] = yaw_rate / max_yaw_rate
            clamped[3] = self._action_from_throttle(throttle)

            if np.any(clamped != action):
                was_clamped = True

        elif mode == ActionMode.VELOCITY and len(action) >= 3:
            max_h = limits.max_horizontal_speed_m_s
            max_v = limits.max_vertical_speed_m_s

            vx = np.clip(action[0] * max_h, -max_h, max_h)
            vy = np.clip(action[1] * max_h, -max_h, max_h)
            vz = np.clip(action[2] * max_v, -max_v, max_v)

            # Geofence-aware velocity clamping
            margin = limits.geofence_margin_m
            if position[0] < limits.geofence_x_min + margin and vx < 0:
                vx = 0.0
                was_clamped = True
                reasons.append("geofence_x_min")
            if position[0] > limits.geofence_x_max - margin and vx > 0:
                vx = 0.0
                was_clamped = True
                reasons.append("geofence_x_max")
            if position[1] < limits.geofence_y_min + margin and vy < 0:
                vy = 0.0
                was_clamped = True
                reasons.append("geofence_y_min")
            if position[1] > limits.geofence_y_max - margin and vy > 0:
                vy = 0.0
                was_clamped = True
                reasons.append("geofence_y_max")

            clamped[0] = vx / max_h if max_h > 0 else 0.0
            clamped[1] = vy / max_h if max_h > 0 else 0.0
            clamped[2] = vz / max_v if max_v > 0 else 0.0
            if len(action) >= 4:
                clamped[3] = np.clip(action[3], -1.0, 1.0)

        reason = f"Clamped: {', '.join(reasons)}" if reasons else ""
        return clamped, was_clamped, reason

    def _emergency_stop_action(self) -> np.ndarray:
        """Generate emergency stop action (level and maintain altitude)."""
        mode = self.action_mode
        if mode == ActionMode.MOTOR_THRUSTS:
            return self._hover_motor_action()
        if mode == ActionMode.VELOCITY:
            shape = getattr(getattr(self.env, "action_space", None), "shape", (3,))
            if shape and len(shape) == 1 and shape[0] == 4:
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Default: attitude or attitude-rate
        return np.array([0.0, 0.0, 0.0, self._action_from_throttle(self._hover_throttle())], dtype=np.float32)

    def _default_guardian(self, obs: np.ndarray) -> np.ndarray:
        """Default guardian controller - stabilize and hover."""
        position, velocity, orientation, angular_vel = self._parse_observation(obs)
        roll, pitch, yaw = self._orientation_to_rpy(orientation)
        limits = self.config.limits

        # Conservative PD gains for recovery
        roll_correction = -roll * 2.0 - angular_vel[0] * 0.5
        pitch_correction = -pitch * 2.0 - angular_vel[1] * 0.5

        # Altitude hold
        altitude_error = 2.0 - position[2]  # Target 2m
        thrust = self._hover_throttle() + altitude_error * 0.3 - velocity[2] * 0.2
        thrust = float(np.clip(thrust, limits.min_throttle, limits.max_throttle))

        if self.action_mode == ActionMode.MOTOR_THRUSTS:
            return self._hover_motor_action()

        if self.action_mode == ActionMode.VELOCITY:
            vz = 0.0
            if position[2] < limits.min_altitude_m + 1.0:
                vz = limits.max_vertical_speed_m_s * 0.5
            action = [0.0, 0.0, vz / limits.max_vertical_speed_m_s]
            shape = getattr(getattr(self.env, "action_space", None), "shape", (3,))
            if shape and len(shape) == 1 and shape[0] == 4:
                action.append(0.0)
            return np.array(action, dtype=np.float32)

        if self.action_mode == ActionMode.ATTITUDE:
            max_tilt = np.deg2rad(limits.max_tilt_deg)
            roll_cmd = np.clip(roll_correction, -max_tilt, max_tilt)
            pitch_cmd = np.clip(pitch_correction, -max_tilt, max_tilt)
            return np.array([
                roll_cmd / max_tilt,
                pitch_cmd / max_tilt,
                yaw / np.pi,
                self._action_from_throttle(thrust),
            ], dtype=np.float32)

        # ATTITUDE_RATES fallback
        max_rate = np.deg2rad(limits.max_tilt_rate_deg_s)
        roll_rate = np.clip(roll_correction, -max_rate, max_rate)
        pitch_rate = np.clip(pitch_correction, -max_rate, max_rate)
        return np.array([
            roll_rate / max_rate,
            pitch_rate / max_rate,
            0.0,
            self._action_from_throttle(thrust),
        ], dtype=np.float32)

    def _throttle_from_action(self, action_val: float) -> float:
        limits = self.config.limits
        return (action_val + 1.0) * 0.5 * (limits.max_throttle - limits.min_throttle) + limits.min_throttle

    def _action_from_throttle(self, throttle: float) -> float:
        limits = self.config.limits
        if limits.max_throttle <= limits.min_throttle:
            return 0.0
        return ((throttle - limits.min_throttle) / (limits.max_throttle - limits.min_throttle)) * 2.0 - 1.0

    def _clamp_throttle_for_altitude(
        self,
        throttle: float,
        position: np.ndarray,
    ) -> Tuple[float, List[str]]:
        limits = self.config.limits
        reasons: List[str] = []

        if position[2] < limits.min_altitude_m + 0.5:
            min_throttle = limits.min_throttle + 0.2
            if throttle < min_throttle:
                throttle = min_throttle
                reasons.append("min_altitude_thrust")

        if position[2] > limits.max_altitude_m - 1.0:
            max_throttle = min(0.5, limits.max_throttle)
            if throttle > max_throttle:
                throttle = max_throttle
                reasons.append("max_altitude_thrust")

        throttle = float(np.clip(throttle, limits.min_throttle, limits.max_throttle))
        return throttle, reasons

    def _hover_throttle(self) -> float:
        limits = self.config.limits
        if hasattr(self.env, "platform_config"):
            cfg = self.env.platform_config
            mass = cfg.get("mass", 1.0)
            num_motors = cfg.get("num_motors", 4)
            max_thrust = cfg.get("max_thrust_per_motor", None)
            if max_thrust:
                hover = (mass * 9.81) / (max_thrust * max(1, num_motors))
                return float(np.clip(hover, limits.min_throttle, limits.max_throttle))
        return float(np.clip(0.5, limits.min_throttle, limits.max_throttle))

    def _hover_motor_action(self) -> np.ndarray:
        num_motors = self._num_motors or 4
        hover_action = self._action_from_throttle(self._hover_throttle())
        return np.full((num_motors,), hover_action, dtype=np.float32)

    def _orientation_to_rpy(self, orientation: np.ndarray) -> Tuple[float, float, float]:
        if len(orientation) == 4:
            return self._quat_to_rpy(orientation)
        if len(orientation) >= 3:
            return float(orientation[0]), float(orientation[1]), float(orientation[2])
        return 0.0, 0.0, 0.0

    def _compute_tilt(self, orientation: np.ndarray) -> float:
        """Compute total tilt angle in degrees."""
        if len(orientation) == 4:
            # Quaternion - convert to Euler
            roll, pitch = self._quat_to_rp(orientation)
        else:
            roll, pitch = orientation[0], orientation[1]

        tilt = np.sqrt(roll**2 + pitch**2)
        return np.rad2deg(tilt)

    def _quat_to_rp(self, q: np.ndarray) -> Tuple[float, float]:
        """Convert quaternion to roll, pitch."""
        roll, pitch, _ = self._quat_to_rpy(q)
        return roll, pitch

    def _quat_to_rpy(self, q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to roll, pitch, yaw."""
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        return roll, pitch, yaw

    def _parse_observation(
        self,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse observation into components.

        Assumes format: [position(3), velocity(3), orientation(4), angular_velocity(3), ...]
        Override this for different observation formats.
        """
        if len(obs) >= 13:
            position = obs[0:3]
            velocity = obs[3:6]
            orientation = obs[6:10]  # Quaternion
            angular_vel = obs[10:13]
        else:
            # Fallback for shorter observations
            position = np.zeros(3)
            velocity = np.zeros(3)
            orientation = np.array([1, 0, 0, 0])  # Identity quaternion
            angular_vel = np.zeros(3)

            if len(obs) >= 3:
                position = obs[0:3]
            if len(obs) >= 6:
                velocity = obs[3:6]

        return position, velocity, orientation, angular_vel

    def _get_current_obs(self) -> np.ndarray:
        """Get current observation from environment."""
        # Try to get observation from wrapped env
        if hasattr(self.env, '_last_obs'):
            return self.env._last_obs
        if hasattr(self.env, 'state'):
            return self.env.state
        # Fallback - return zeros
        return np.zeros(self.observation_space.shape)

    def _detect_post_step_violation(self, obs: np.ndarray) -> bool:
        """Detect if post-step state violates safety constraints."""
        position, velocity, orientation, _ = self._parse_observation(obs)
        limits = self.config.limits

        # Check altitude
        if position[2] < limits.min_altitude_m * 0.5:
            return True

        # Check geofence
        if (position[0] < limits.geofence_x_min or
            position[0] > limits.geofence_x_max or
            position[1] < limits.geofence_y_min or
            position[1] > limits.geofence_y_max):
            return True

        return False

    def _log_intervention(self, intervention_type: str):
        """Log an intervention."""
        self._total_interventions += 1
        self._state.intervention_count = self._total_interventions

        if self.config.log_interventions:
            self._intervention_history.append({
                'step': self._step_count,
                'type': intervention_type,
                'reason': self._state.intervention_reason,
            })

    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get intervention statistics."""
        if not self._intervention_history:
            return {'total': 0}

        type_counts = {}
        for entry in self._intervention_history:
            t = entry['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            'total': self._total_interventions,
            'by_type': type_counts,
            'intervention_rate': self._total_interventions / max(1, self._step_count),
        }


def make_shielded_env(
    env: gym.Env,
    config: Optional[ShieldConfig] = None,
    guardian: Optional[Callable] = None,
) -> SafetyShieldWrapper:
    """
    Convenience function to wrap environment with safety shield.

    Args:
        env: Base environment
        config: Shield configuration
        guardian: Optional guardian controller

    Returns:
        Shielded environment
    """
    return SafetyShieldWrapper(env, config, guardian)

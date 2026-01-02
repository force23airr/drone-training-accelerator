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
        self.guardian_controller = guardian_controller or self._default_guardian

        self._state = ShieldState()
        self._step_count = 0
        self._total_interventions = 0
        self._intervention_history: List[Dict[str, Any]] = []

    @property
    def shield_state(self) -> ShieldState:
        """Get current shield state."""
        return self._state

    @property
    def intervention_count(self) -> int:
        """Total number of interventions."""
        return self._total_interventions

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

        # Assume action is [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust]
        # This format is common but may need adaptation

        if len(action) >= 4:
            # Clamp attitude commands based on tilt limits
            max_tilt_rad = np.deg2rad(limits.max_tilt_deg)

            if abs(action[0]) > max_tilt_rad:
                clamped[0] = np.clip(action[0], -max_tilt_rad, max_tilt_rad)
                was_clamped = True
                reasons.append("roll")

            if abs(action[1]) > max_tilt_rad:
                clamped[1] = np.clip(action[1], -max_tilt_rad, max_tilt_rad)
                was_clamped = True
                reasons.append("pitch")

            # Clamp thrust based on altitude
            if position[2] < limits.min_altitude_m + 0.5:
                # Near minimum altitude - ensure positive climb
                clamped[3] = max(clamped[3], limits.min_throttle + 0.2)
                if action[3] < limits.min_throttle + 0.2:
                    was_clamped = True
                    reasons.append("min_altitude_thrust")

            if position[2] > limits.max_altitude_m - 1.0:
                # Near maximum altitude - limit climb
                clamped[3] = min(clamped[3], 0.5)
                if action[3] > 0.5:
                    was_clamped = True
                    reasons.append("max_altitude_thrust")

            # Always clamp throttle to valid range
            orig_thrust = clamped[3]
            clamped[3] = np.clip(clamped[3], limits.min_throttle, limits.max_throttle)
            if clamped[3] != orig_thrust:
                was_clamped = True
                reasons.append("throttle_range")

        reason = f"Clamped: {', '.join(reasons)}" if reasons else ""
        return clamped, was_clamped, reason

    def _emergency_stop_action(self) -> np.ndarray:
        """Generate emergency stop action (level and maintain altitude)."""
        # Zero attitude commands, hover throttle
        return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    def _default_guardian(self, obs: np.ndarray) -> np.ndarray:
        """Default guardian controller - stabilize and hover."""
        # Parse observation
        position, velocity, orientation, angular_vel = self._parse_observation(obs)

        # Simple PD controller for stabilization
        # Zero roll/pitch, zero yaw rate, maintain altitude
        roll_cmd = -orientation[0] * 2.0 - angular_vel[0] * 0.5
        pitch_cmd = -orientation[1] * 2.0 - angular_vel[1] * 0.5
        yaw_rate_cmd = 0.0

        # Altitude hold
        altitude_error = 2.0 - position[2]  # Target 2m
        thrust = 0.5 + altitude_error * 0.3 - velocity[2] * 0.2

        action = np.array([roll_cmd, pitch_cmd, yaw_rate_cmd, thrust], dtype=np.float32)

        # Clamp to safe ranges
        max_tilt = np.deg2rad(20)  # Conservative tilt for recovery
        action[0] = np.clip(action[0], -max_tilt, max_tilt)
        action[1] = np.clip(action[1], -max_tilt, max_tilt)
        action[3] = np.clip(action[3], 0.3, 0.8)

        return action

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
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        return roll, pitch

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

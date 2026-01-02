"""
Smoothness Metrics Collector

Measures control smoothness and trajectory quality:
- Jerk (rate of change of acceleration)
- Motor command variance
- Action rate of change
- Oscillation detection
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

from evaluation.metrics.base_metrics import MetricCollector, MetricResult


class SmoothnessMetrics(MetricCollector):
    """
    Measures control smoothness and stability.

    Tracks:
    - Jerk (3rd derivative of position) - lower is smoother
    - Motor command variance - lower is more stable
    - Action rate of change - lower is smoother control
    - Angular velocity variance - lower is more stable
    """

    def __init__(
        self,
        control_hz: float = 48.0,
        jerk_window: int = 10,
        oscillation_threshold_hz: float = 5.0,
    ):
        """
        Args:
            control_hz: Control loop frequency
            jerk_window: Window size for jerk computation
            oscillation_threshold_hz: Frequency threshold for oscillation detection
        """
        super().__init__(name="SmoothnessMetrics")

        self.dt = 1.0 / control_hz
        self.control_hz = control_hz
        self.jerk_window = jerk_window
        self.oscillation_threshold_hz = oscillation_threshold_hz

        # State buffers
        self._positions: deque = deque(maxlen=jerk_window)
        self._velocities: deque = deque(maxlen=jerk_window)
        self._actions: List[np.ndarray] = []
        self._angular_velocities: List[np.ndarray] = []

        # Computed metrics
        self._jerks: List[float] = []
        self._action_changes: List[float] = []

    def reset(self) -> None:
        super().reset()
        self._positions.clear()
        self._velocities.clear()
        self._actions = []
        self._angular_velocities = []
        self._jerks = []
        self._action_changes = []

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        super().step(obs, action, reward, next_obs, terminated, truncated, info)

        # Extract state
        pos = next_obs[:3]
        vel = next_obs[3:6]
        angular_vel = next_obs[10:13] if len(next_obs) >= 13 else np.zeros(3)

        # Store for analysis
        self._positions.append(pos.copy())
        self._velocities.append(vel.copy())
        self._actions.append(action.copy())
        self._angular_velocities.append(angular_vel.copy())

        # Compute jerk if enough samples
        if len(self._velocities) >= 3:
            jerk = self._compute_jerk()
            self._jerks.append(jerk)

        # Compute action change rate
        if len(self._actions) >= 2:
            action_change = np.linalg.norm(self._actions[-1] - self._actions[-2])
            self._action_changes.append(action_change)

    def _compute_jerk(self) -> float:
        """Compute jerk magnitude from recent velocities."""
        if len(self._velocities) < 3:
            return 0.0

        # Get last 3 velocities
        v0 = self._velocities[-3]
        v1 = self._velocities[-2]
        v2 = self._velocities[-1]

        # Compute accelerations
        a0 = (v1 - v0) / self.dt
        a1 = (v2 - v1) / self.dt

        # Compute jerk
        jerk = (a1 - a0) / self.dt

        return float(np.linalg.norm(jerk))

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        results = []

        # Jerk statistics
        if self._jerks:
            jerks = np.array(self._jerks)
            results.extend([
                MetricResult('mean_jerk', float(np.mean(jerks)), 'm/s^3', higher_is_better=False),
                MetricResult('max_jerk', float(np.max(jerks)), 'm/s^3', higher_is_better=False),
                MetricResult('jerk_std', float(np.std(jerks)), 'm/s^3', higher_is_better=False),
                MetricResult('jerk_percentile_95', float(np.percentile(jerks, 95)), 'm/s^3', higher_is_better=False),
            ])
        else:
            results.extend([
                MetricResult('mean_jerk', 0.0, 'm/s^3', higher_is_better=False),
                MetricResult('max_jerk', 0.0, 'm/s^3', higher_is_better=False),
                MetricResult('jerk_std', 0.0, 'm/s^3', higher_is_better=False),
                MetricResult('jerk_percentile_95', 0.0, 'm/s^3', higher_is_better=False),
            ])

        # Action smoothness
        if self._actions:
            actions = np.array(self._actions)

            # Motor variance (per motor, then average)
            motor_variance = float(np.var(actions, axis=0).mean())

            # Action rate of change
            if self._action_changes:
                action_rate = float(np.mean(self._action_changes))
                max_action_change = float(np.max(self._action_changes))
            else:
                action_rate = 0.0
                max_action_change = 0.0

            # Action saturation (time spent at limits)
            saturation_ratio = float(np.mean(np.abs(actions) > 0.95))

            results.extend([
                MetricResult('motor_variance', motor_variance, 'normalized', higher_is_better=False),
                MetricResult('action_rate', action_rate, 'change/step', higher_is_better=False),
                MetricResult('max_action_change', max_action_change, 'change', higher_is_better=False),
                MetricResult('saturation_ratio', saturation_ratio, 'ratio', higher_is_better=False),
            ])
        else:
            results.extend([
                MetricResult('motor_variance', 0.0, 'normalized', higher_is_better=False),
                MetricResult('action_rate', 0.0, 'change/step', higher_is_better=False),
                MetricResult('max_action_change', 0.0, 'change', higher_is_better=False),
                MetricResult('saturation_ratio', 0.0, 'ratio', higher_is_better=False),
            ])

        # Angular velocity smoothness
        if self._angular_velocities:
            angular_vels = np.array(self._angular_velocities)

            # Angular velocity variance (stability indicator)
            angular_variance = float(np.var(angular_vels, axis=0).mean())

            # Angular velocity magnitude (overall rotation activity)
            angular_magnitude = float(np.linalg.norm(angular_vels, axis=1).mean())

            results.extend([
                MetricResult('angular_velocity_variance', angular_variance, 'rad^2/s^2', higher_is_better=False),
                MetricResult('mean_angular_velocity', angular_magnitude, 'rad/s', higher_is_better=False),
            ])
        else:
            results.extend([
                MetricResult('angular_velocity_variance', 0.0, 'rad^2/s^2', higher_is_better=False),
                MetricResult('mean_angular_velocity', 0.0, 'rad/s', higher_is_better=False),
            ])

        # Oscillation detection
        oscillation_score = self._detect_oscillations()
        results.append(
            MetricResult('oscillation_score', oscillation_score, 'normalized', higher_is_better=False)
        )

        # Overall smoothness score
        smoothness_score = self._compute_smoothness_score()
        results.append(
            MetricResult('smoothness_score', smoothness_score, 'normalized', higher_is_better=True)
        )

        return results

    def _detect_oscillations(self) -> float:
        """
        Detect oscillations in control/trajectory.

        Returns score 0-1 where 0 is no oscillation, 1 is severe oscillation.
        """
        if len(self._actions) < 20:
            return 0.0

        actions = np.array(self._actions)

        # Look for sign changes in action derivatives
        action_diff = np.diff(actions, axis=0)
        sign_changes = np.sum(np.diff(np.sign(action_diff), axis=0) != 0, axis=0)

        # Normalize by number of steps
        oscillation_rate = sign_changes / len(actions)

        # High oscillation rate indicates instability
        return float(np.mean(np.clip(oscillation_rate / 0.5, 0, 1)))

    def _compute_smoothness_score(self) -> float:
        """
        Compute overall smoothness score (0-1, higher is smoother).
        """
        scores = []

        # Jerk score (inverse of normalized jerk)
        if self._jerks:
            mean_jerk = np.mean(self._jerks)
            jerk_score = 1.0 / (1.0 + mean_jerk / 100.0)  # Normalize by 100 m/s^3
            scores.append(jerk_score)

        # Action smoothness score
        if self._action_changes:
            mean_change = np.mean(self._action_changes)
            action_score = 1.0 / (1.0 + mean_change / 0.5)  # Normalize by 0.5
            scores.append(action_score)

        # Motor variance score
        if self._actions:
            motor_var = np.var(self._actions, axis=0).mean()
            motor_score = 1.0 / (1.0 + motor_var / 0.1)  # Normalize by 0.1
            scores.append(motor_score)

        if scores:
            return float(np.mean(scores))
        return 1.0

    def get_running_metrics(self) -> Dict[str, float]:
        metrics = {}

        if self._jerks:
            metrics['current_jerk'] = self._jerks[-1] if self._jerks else 0.0
            metrics['mean_jerk'] = float(np.mean(self._jerks[-100:])) if self._jerks else 0.0

        if self._action_changes:
            metrics['action_rate'] = float(np.mean(self._action_changes[-100:]))

        return metrics

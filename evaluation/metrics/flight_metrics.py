"""
Flight Performance Metrics Collector

Measures core flight performance:
- Success rate (task completion)
- Episode length and time
- Position tracking accuracy
- Goal achievement
"""

import numpy as np
from typing import Dict, Any, List, Optional

from evaluation.metrics.base_metrics import MetricCollector, MetricResult


class FlightMetrics(MetricCollector):
    """
    Measures core flight performance metrics.

    Tracks:
    - Episode success/failure
    - Episode duration (steps and time)
    - Position error (from target)
    - Velocity tracking
    - Goal achievement metrics
    """

    def __init__(
        self,
        target_position: Optional[np.ndarray] = None,
        success_radius: float = 0.5,
        control_hz: float = 48.0,
    ):
        """
        Args:
            target_position: Target position for hover/waypoint tasks
            success_radius: Radius within which position is considered successful
            control_hz: Control loop frequency for time calculations
        """
        super().__init__(name="FlightMetrics")

        self.target_position = target_position if target_position is not None else np.array([0, 0, 1])
        self.success_radius = success_radius
        self.dt = 1.0 / control_hz

        # Episode tracking
        self._total_reward: float = 0.0
        self._position_errors: List[float] = []
        self._velocity_magnitudes: List[float] = []
        self._time_at_target: float = 0.0
        self._time_to_target: Optional[float] = None
        self._reached_target: bool = False
        self._waypoints_reached: int = 0
        self._total_waypoints: int = 0

    def reset(self) -> None:
        super().reset()
        self._total_reward = 0.0
        self._position_errors = []
        self._velocity_magnitudes = []
        self._time_at_target = 0.0
        self._time_to_target = None
        self._reached_target = False
        self._waypoints_reached = 0
        self._total_waypoints = 0

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

        pos = next_obs[:3]
        vel = next_obs[3:6]

        # Accumulate reward
        self._total_reward += reward

        # Update target from info if available
        if 'target_position' in info:
            self.target_position = np.array(info['target_position'])

        # Position error
        pos_error = np.linalg.norm(pos - self.target_position)
        self._position_errors.append(pos_error)

        # Velocity magnitude
        self._velocity_magnitudes.append(np.linalg.norm(vel))

        # Check if at target
        at_target = pos_error < self.success_radius
        if at_target:
            self._time_at_target += self.dt
            if not self._reached_target:
                self._reached_target = True
                self._time_to_target = self._step_count * self.dt

        # Waypoint tracking
        if 'waypoints_reached' in info:
            self._waypoints_reached = info['waypoints_reached']
        if 'total_waypoints' in info:
            self._total_waypoints = info['total_waypoints']

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        results = []

        # Episode duration
        episode_steps = self._step_count
        episode_time = episode_steps * self.dt

        results.append(
            MetricResult('episode_steps', float(episode_steps), 'steps', higher_is_better=True)
        )
        results.append(
            MetricResult('episode_time', episode_time, 'seconds', higher_is_better=True)
        )

        # Total reward
        results.append(
            MetricResult('total_reward', self._total_reward, 'reward', higher_is_better=True)
        )
        results.append(
            MetricResult('mean_reward_per_step', self._total_reward / max(episode_steps, 1), 'reward/step', higher_is_better=True)
        )

        # Success determination
        success = self._determine_success(info)
        results.append(
            MetricResult('success', float(success), 'binary', higher_is_better=True)
        )

        # Position error statistics
        if self._position_errors:
            errors = np.array(self._position_errors)
            results.extend([
                MetricResult('mean_position_error', float(np.mean(errors)), 'meters', higher_is_better=False),
                MetricResult('max_position_error', float(np.max(errors)), 'meters', higher_is_better=False),
                MetricResult('min_position_error', float(np.min(errors)), 'meters', higher_is_better=False),
                MetricResult('final_position_error', float(errors[-1]), 'meters', higher_is_better=False),
                MetricResult('position_error_std', float(np.std(errors)), 'meters', higher_is_better=False),
            ])
        else:
            results.extend([
                MetricResult('mean_position_error', 0.0, 'meters', higher_is_better=False),
                MetricResult('max_position_error', 0.0, 'meters', higher_is_better=False),
                MetricResult('min_position_error', 0.0, 'meters', higher_is_better=False),
                MetricResult('final_position_error', 0.0, 'meters', higher_is_better=False),
                MetricResult('position_error_std', 0.0, 'meters', higher_is_better=False),
            ])

        # Velocity statistics
        if self._velocity_magnitudes:
            velocities = np.array(self._velocity_magnitudes)
            results.extend([
                MetricResult('mean_velocity', float(np.mean(velocities)), 'm/s', higher_is_better=False),
                MetricResult('max_velocity', float(np.max(velocities)), 'm/s', higher_is_better=False),
            ])
        else:
            results.extend([
                MetricResult('mean_velocity', 0.0, 'm/s', higher_is_better=False),
                MetricResult('max_velocity', 0.0, 'm/s', higher_is_better=False),
            ])

        # Target achievement
        results.append(
            MetricResult('reached_target', float(self._reached_target), 'binary', higher_is_better=True)
        )
        results.append(
            MetricResult('time_at_target', self._time_at_target, 'seconds', higher_is_better=True)
        )

        if self._time_to_target is not None:
            results.append(
                MetricResult('time_to_target', self._time_to_target, 'seconds', higher_is_better=False)
            )
        else:
            results.append(
                MetricResult('time_to_target', episode_time, 'seconds', higher_is_better=False)
            )

        # Target time ratio (fraction of episode at target)
        target_ratio = self._time_at_target / max(episode_time, 0.01)
        results.append(
            MetricResult('target_time_ratio', target_ratio, 'ratio', higher_is_better=True)
        )

        # Waypoint completion
        if self._total_waypoints > 0:
            waypoint_completion = self._waypoints_reached / self._total_waypoints
        else:
            waypoint_completion = 1.0 if self._reached_target else 0.0

        results.append(
            MetricResult('waypoints_reached', float(self._waypoints_reached), 'count', higher_is_better=True)
        )
        results.append(
            MetricResult('waypoint_completion', waypoint_completion, 'ratio', higher_is_better=True)
        )

        # Termination info
        terminated_flag = info.get('terminated', False)
        truncated_flag = info.get('truncated', False)
        termination_reason = info.get('termination_reason', 'unknown')

        results.append(
            MetricResult('terminated', float(terminated_flag), 'binary', higher_is_better=False)
        )
        results.append(
            MetricResult('truncated', float(truncated_flag), 'binary', higher_is_better=True)
        )

        return results

    def _determine_success(self, info: Dict[str, Any]) -> bool:
        """
        Determine if episode was successful.

        Success criteria (in order of priority):
        1. Explicit success flag in info
        2. Reached target and stayed there
        3. Completed waypoints
        4. High target time ratio
        """
        # Check explicit success flag
        if 'success' in info:
            return bool(info['success'])

        # Check episode stats
        if 'episode_stats' in info:
            stats = info['episode_stats']
            if 'success' in stats:
                return bool(stats['success'])

        # Heuristic: successful if reached target and spent significant time there
        if self._reached_target and self._time_at_target > 1.0:
            return True

        # Heuristic: successful if completed waypoints
        if self._total_waypoints > 0 and self._waypoints_reached >= self._total_waypoints:
            return True

        # Heuristic: successful if low final position error
        if self._position_errors and self._position_errors[-1] < self.success_radius:
            return True

        return False

    def get_running_metrics(self) -> Dict[str, float]:
        metrics = {
            'total_reward': self._total_reward,
            'time_at_target': self._time_at_target,
        }

        if self._position_errors:
            metrics['current_error'] = self._position_errors[-1]
            metrics['mean_error'] = float(np.mean(self._position_errors[-100:]))

        return metrics

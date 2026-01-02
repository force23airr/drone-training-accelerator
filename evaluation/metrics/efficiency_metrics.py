"""
Efficiency Metrics Collector

Measures energy efficiency and resource utilization:
- Thrust integral (energy proxy)
- Distance traveled per unit energy
- Hover efficiency
- Path efficiency
"""

import numpy as np
from typing import Dict, Any, List, Optional

from evaluation.metrics.base_metrics import MetricCollector, MetricResult


class EfficiencyMetrics(MetricCollector):
    """
    Measures energy efficiency.

    Tracks:
    - Thrust integral (sum of motor commands over time)
    - Distance traveled
    - Distance per unit energy (efficiency ratio)
    - Hover efficiency (deviation from optimal hover thrust)
    - Path efficiency (direct distance vs actual distance)
    """

    def __init__(
        self,
        hover_thrust: float = 0.5,
        mass_kg: float = 1.0,
        gravity: float = 9.81,
        motor_power_factor: float = 1.0,
    ):
        """
        Args:
            hover_thrust: Normalized thrust required for hover (0-1)
            mass_kg: Drone mass for power calculations
            gravity: Gravitational acceleration
            motor_power_factor: Scaling factor for power estimation
        """
        super().__init__(name="EfficiencyMetrics")

        self.hover_thrust = hover_thrust
        self.mass_kg = mass_kg
        self.gravity = gravity
        self.motor_power_factor = motor_power_factor

        self._thrust_integral: float = 0.0
        self._power_integral: float = 0.0
        self._total_distance: float = 0.0
        self._direct_distance: float = 0.0
        self._hover_deviation_integral: float = 0.0
        self._prev_pos: Optional[np.ndarray] = None
        self._start_pos: Optional[np.ndarray] = None
        self._end_pos: Optional[np.ndarray] = None
        self._num_hover_steps: int = 0
        self._actions: List[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self._thrust_integral = 0.0
        self._power_integral = 0.0
        self._total_distance = 0.0
        self._direct_distance = 0.0
        self._hover_deviation_integral = 0.0
        self._prev_pos = None
        self._start_pos = None
        self._end_pos = None
        self._num_hover_steps = 0
        self._actions = []

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

        # Track position
        if self._start_pos is None:
            self._start_pos = pos.copy()
        self._end_pos = pos.copy()

        # Thrust integral (sum of absolute motor commands)
        thrust = np.sum(np.abs(action))
        self._thrust_integral += thrust

        # Power estimation: P = k * thrust^1.5 (empirical relationship)
        power = self.motor_power_factor * np.sum(np.power(np.abs(action), 1.5))
        self._power_integral += power

        # Distance traveled
        if self._prev_pos is not None:
            dist = np.linalg.norm(pos - self._prev_pos)
            self._total_distance += dist
        self._prev_pos = pos.copy()

        # Hover efficiency (deviation from ideal hover thrust)
        mean_thrust = np.mean(action)
        hover_deviation = abs(mean_thrust - self.hover_thrust)
        self._hover_deviation_integral += hover_deviation

        # Track if hovering (low velocity)
        speed = np.linalg.norm(vel)
        if speed < 0.5:  # Considered hovering if speed < 0.5 m/s
            self._num_hover_steps += 1

        self._actions.append(action.copy())

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        results = []

        # Basic efficiency metrics
        results.append(
            MetricResult('thrust_integral', self._thrust_integral, 'thrust*steps', higher_is_better=False)
        )
        results.append(
            MetricResult('power_integral', self._power_integral, 'power*steps', higher_is_better=False)
        )
        results.append(
            MetricResult('total_distance', self._total_distance, 'meters', higher_is_better=True)
        )

        # Distance per energy (efficiency ratio)
        if self._thrust_integral > 0:
            efficiency = self._total_distance / self._thrust_integral
        else:
            efficiency = 0.0
        results.append(
            MetricResult('distance_per_thrust', efficiency, 'm/thrust', higher_is_better=True)
        )

        # Power efficiency
        if self._power_integral > 0:
            power_efficiency = self._total_distance / self._power_integral
        else:
            power_efficiency = 0.0
        results.append(
            MetricResult('distance_per_power', power_efficiency, 'm/power', higher_is_better=True)
        )

        # Direct distance (start to end)
        if self._start_pos is not None and self._end_pos is not None:
            self._direct_distance = np.linalg.norm(self._end_pos - self._start_pos)
        results.append(
            MetricResult('direct_distance', self._direct_distance, 'meters', higher_is_better=True)
        )

        # Path efficiency (direct/total, 1.0 = perfect straight line)
        if self._total_distance > 0:
            path_efficiency = self._direct_distance / self._total_distance
        else:
            path_efficiency = 1.0
        results.append(
            MetricResult('path_efficiency', path_efficiency, 'ratio', higher_is_better=True)
        )

        # Hover efficiency
        if self._step_count > 0:
            mean_hover_deviation = self._hover_deviation_integral / self._step_count
            hover_efficiency = 1.0 / (1.0 + mean_hover_deviation)
        else:
            hover_efficiency = 1.0
        results.append(
            MetricResult('hover_efficiency', hover_efficiency, 'normalized', higher_is_better=True)
        )

        # Hover time ratio
        if self._step_count > 0:
            hover_ratio = self._num_hover_steps / self._step_count
        else:
            hover_ratio = 0.0
        results.append(
            MetricResult('hover_time_ratio', hover_ratio, 'ratio', higher_is_better=True)
        )

        # Action efficiency (how much of action range is used effectively)
        if self._actions:
            actions = np.array(self._actions)
            action_utilization = np.mean(np.abs(actions))  # Average absolute action
            action_variance = np.var(actions)

            results.append(
                MetricResult('action_utilization', float(action_utilization), 'normalized', higher_is_better=False)
            )
            results.append(
                MetricResult('action_variance', float(action_variance), 'normalized', higher_is_better=False)
            )

        # Estimated flight time (if battery capacity known)
        # Assuming linear relationship: time_remaining = capacity / mean_power
        if self._power_integral > 0 and self._step_count > 0:
            mean_power = self._power_integral / self._step_count
            # Normalize to represent relative efficiency
            power_normalized = mean_power / max(self.hover_thrust, 0.1)
        else:
            power_normalized = 1.0

        results.append(
            MetricResult('normalized_power_consumption', float(power_normalized), 'ratio', higher_is_better=False)
        )

        # Overall efficiency score
        efficiency_score = self._compute_efficiency_score()
        results.append(
            MetricResult('efficiency_score', efficiency_score, 'normalized', higher_is_better=True)
        )

        return results

    def _compute_efficiency_score(self) -> float:
        """
        Compute overall efficiency score (0-1).

        Combines:
        - Path efficiency (40%)
        - Thrust efficiency (30%)
        - Hover efficiency (30%)
        """
        scores = []

        # Path efficiency
        if self._total_distance > 0:
            path_eff = min(self._direct_distance / self._total_distance, 1.0)
        else:
            path_eff = 1.0
        scores.append(('path', path_eff, 0.4))

        # Thrust efficiency (inverse of normalized thrust)
        if self._step_count > 0:
            mean_thrust = self._thrust_integral / self._step_count
            # Normalize: ideal is around hover_thrust * num_motors
            ideal_thrust = self.hover_thrust * 4  # Assuming 4 motors
            thrust_eff = 1.0 / (1.0 + abs(mean_thrust - ideal_thrust) / ideal_thrust)
        else:
            thrust_eff = 1.0
        scores.append(('thrust', thrust_eff, 0.3))

        # Hover efficiency
        if self._step_count > 0:
            mean_dev = self._hover_deviation_integral / self._step_count
            hover_eff = 1.0 / (1.0 + mean_dev * 5)
        else:
            hover_eff = 1.0
        scores.append(('hover', hover_eff, 0.3))

        # Weighted average
        total = sum(score * weight for _, score, weight in scores)
        return float(total)

    def get_running_metrics(self) -> Dict[str, float]:
        return {
            'thrust_integral': self._thrust_integral,
            'distance': self._total_distance,
            'efficiency': self._total_distance / max(self._thrust_integral, 1e-6),
        }

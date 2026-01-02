"""
Base Metric Collector Interface

Abstract base class for all metric collectors in the evaluation harness.
Each collector is responsible for tracking specific aspects of drone performance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class MetricResult:
    """
    Result from a metric computation.

    Attributes:
        name: Metric identifier (e.g., 'mean_jerk', 'crash_count')
        value: Numeric value of the metric
        unit: Unit of measurement (e.g., 'm/s^3', 'count', 'normalized')
        higher_is_better: Whether higher values indicate better performance
        metadata: Additional context (thresholds violated, timestamps, etc.)
    """
    name: str
    value: float
    unit: str
    higher_is_better: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f} {self.unit}"


@dataclass
class ConstraintViolation:
    """
    Record of a single constraint violation.

    Used by SafetyMetrics to track when limits are exceeded.
    """
    timestamp: float
    step: int
    constraint_type: str  # 'tilt_limit', 'min_altitude', 'max_speed', 'no_fly_zone'
    value: float          # Actual value that violated constraint
    limit: float          # The limit that was exceeded
    position: Optional[np.ndarray] = None
    severity: str = "warning"  # 'warning', 'critical'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'step': self.step,
            'constraint_type': self.constraint_type,
            'value': self.value,
            'limit': self.limit,
            'position': self.position.tolist() if self.position is not None else None,
            'severity': self.severity,
        }


class MetricCollector(ABC):
    """
    Abstract base class for metric collectors.

    Lifecycle:
    1. reset() - Called at start of each episode
    2. step() - Called after each environment step
    3. episode_end() - Called at episode termination, returns metrics

    Implementations should be stateless between episodes (reset clears state).
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._step_count = 0

    @abstractmethod
    def reset(self) -> None:
        """
        Reset collector state for a new episode.

        Must clear all accumulated data from previous episode.
        """
        self._step_count = 0

    @abstractmethod
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
        """
        Process one environment step.

        Args:
            obs: Observation before action
            action: Action taken
            reward: Reward received
            next_obs: Observation after action
            terminated: Whether episode terminated (e.g., crash)
            truncated: Whether episode was truncated (e.g., time limit)
            info: Additional info from environment
        """
        self._step_count += 1

    @abstractmethod
    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        """
        Compute final metrics at episode end.

        Args:
            info: Final info dict from environment

        Returns:
            List of MetricResult objects
        """
        pass

    def get_running_metrics(self) -> Dict[str, float]:
        """
        Get current running metrics (for logging during episode).

        Override to provide real-time metric updates.

        Returns:
            Dict of metric_name -> current_value
        """
        return {}


class CompositeMetricCollector(MetricCollector):
    """
    Combines multiple metric collectors into one.

    Useful for creating a single collector that tracks all metrics.
    """

    def __init__(self, collectors: List[MetricCollector]):
        super().__init__(name="CompositeCollector")
        self.collectors = collectors

    def reset(self) -> None:
        super().reset()
        for collector in self.collectors:
            collector.reset()

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
        for collector in self.collectors:
            collector.step(obs, action, reward, next_obs, terminated, truncated, info)

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        results = []
        for collector in self.collectors:
            results.extend(collector.episode_end(info))
        return results

    def get_running_metrics(self) -> Dict[str, float]:
        metrics = {}
        for collector in self.collectors:
            metrics.update(collector.get_running_metrics())
        return metrics


def aggregate_metrics(
    episode_metrics: List[List[MetricResult]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple episodes.

    Args:
        episode_metrics: List of metric results per episode

    Returns:
        Dict of metric_name -> {mean, std, min, max, count}
    """
    # Collect all values per metric name
    metric_values: Dict[str, List[float]] = {}

    for episode in episode_metrics:
        for result in episode:
            if result.name not in metric_values:
                metric_values[result.name] = []
            metric_values[result.name].append(result.value)

    # Compute statistics
    aggregated = {}
    for name, values in metric_values.items():
        arr = np.array(values)
        aggregated[name] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': len(values),
        }

    return aggregated

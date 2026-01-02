"""
Evaluation Metrics Module

Comprehensive metric collectors for UAV policy evaluation.

Metrics are organized by category:
- Flight: Success rate, episode time, position tracking
- Safety: Crashes, near-misses, constraint violations
- Smoothness: Jerk, motor variance, oscillations
- Efficiency: Energy consumption, path efficiency
- Behavioral: Expert similarity, distribution matching
"""

from evaluation.metrics.base_metrics import (
    MetricResult,
    MetricCollector,
    CompositeMetricCollector,
    ConstraintViolation,
    aggregate_metrics,
)

from evaluation.metrics.flight_metrics import (
    FlightMetrics,
)

from evaluation.metrics.safety_metrics import (
    SafetyMetrics,
    NoFlyZone,
)

from evaluation.metrics.smoothness_metrics import (
    SmoothnessMetrics,
)

from evaluation.metrics.efficiency_metrics import (
    EfficiencyMetrics,
)

from evaluation.metrics.behavioral_metrics import (
    BehavioralMetrics,
    DistributionMetrics,
)

__all__ = [
    # Base
    "MetricResult",
    "MetricCollector",
    "CompositeMetricCollector",
    "ConstraintViolation",
    "aggregate_metrics",
    # Flight
    "FlightMetrics",
    # Safety
    "SafetyMetrics",
    "NoFlyZone",
    # Smoothness
    "SmoothnessMetrics",
    # Efficiency
    "EfficiencyMetrics",
    # Behavioral
    "BehavioralMetrics",
    "DistributionMetrics",
]


def create_default_collectors(
    target_position=None,
    expert_dataset=None,
) -> CompositeMetricCollector:
    """
    Create a composite collector with all default metrics.

    Args:
        target_position: Target position for flight metrics
        expert_dataset: Expert demonstrations for behavioral metrics

    Returns:
        CompositeMetricCollector with all standard metrics
    """
    collectors = [
        FlightMetrics(target_position=target_position),
        SafetyMetrics(),
        SmoothnessMetrics(),
        EfficiencyMetrics(),
    ]

    if expert_dataset is not None:
        collectors.append(BehavioralMetrics.from_dataset(expert_dataset))

    return CompositeMetricCollector(collectors)

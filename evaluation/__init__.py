"""
Evaluation Module

Comprehensive evaluation infrastructure for drone policy assessment:
- Metric collectors for flight, safety, smoothness, efficiency, behavioral metrics
- Promotion gates for production readiness checks
- Evaluation harness for standardized testing
- Report generation in multiple formats
"""

from evaluation.metrics import (
    MetricResult,
    MetricCollector,
    CompositeMetricCollector,
    ConstraintViolation,
    FlightMetrics,
    SafetyMetrics,
    SmoothnessMetrics,
    EfficiencyMetrics,
    BehavioralMetrics,
    NoFlyZone,
    aggregate_metrics,
    create_default_collectors,
)

from evaluation.gates import (
    GateCheckResult,
    PromotionGate,
    PerformanceGate,
    SafetyGate,
    RegressionGate,
    SuccessRateGate,
    CompositeGate,
    create_default_gates,
)

from evaluation.harness import (
    EpisodeResult,
    EvaluationResult,
    EvaluationHarness,
    evaluate_policy,
)

from evaluation.reports import (
    ReportGenerator,
    generate_markdown_report,
    generate_json_report,
)

__all__ = [
    # Metrics
    "MetricResult",
    "MetricCollector",
    "CompositeMetricCollector",
    "ConstraintViolation",
    "FlightMetrics",
    "SafetyMetrics",
    "SmoothnessMetrics",
    "EfficiencyMetrics",
    "BehavioralMetrics",
    "NoFlyZone",
    "aggregate_metrics",
    "create_default_collectors",
    # Gates
    "GateCheckResult",
    "PromotionGate",
    "PerformanceGate",
    "SafetyGate",
    "RegressionGate",
    "SuccessRateGate",
    "CompositeGate",
    "create_default_gates",
    # Harness
    "EpisodeResult",
    "EvaluationResult",
    "EvaluationHarness",
    "evaluate_policy",
    # Reports
    "ReportGenerator",
    "generate_markdown_report",
    "generate_json_report",
]

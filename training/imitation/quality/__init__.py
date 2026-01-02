"""
Quality Module for Demonstration Data

Auto-scoring and filtering of demonstrations.
"""

from training.imitation.quality.auto_scorer import (
    QualityScoreBreakdown,
    ScorerConfig,
    DemonstrationQualityScorer,
    score_demonstrations,
    filter_by_quality,
)

from training.imitation.quality.quality_filters import (
    FilterResult,
    FilterConfig,
    QualityFilter,
    create_strict_filter,
    create_lenient_filter,
)

__all__ = [
    # Auto-scorer
    "QualityScoreBreakdown",
    "ScorerConfig",
    "DemonstrationQualityScorer",
    "score_demonstrations",
    "filter_by_quality",
    # Filters
    "FilterResult",
    "FilterConfig",
    "QualityFilter",
    "create_strict_filter",
    "create_lenient_filter",
]

"""
Training Pipelines Module

Unified training pipelines for end-to-end workflows.

The main entry point is `train_from_demonstrations()` which runs the complete
pipeline from raw demonstrations to a promoted model artifact.

Example:
    >>> from training.pipelines import train_from_demonstrations, PipelineConfig
    >>>
    >>> config = PipelineConfig(
    ...     task_type="hover",
    ...     bc_epochs=100,
    ...     eval_episodes=100,
    ... )
    >>>
    >>> artifact = train_from_demonstrations(
    ...     demo_path="data/expert_flights/",
    ...     output_dir="artifacts/model_v1/",
    ...     env=env,
    ...     config=config,
    ... )
    >>>
    >>> if artifact.promoted:
    ...     print(f"Model ready for deployment: {artifact.model_path}")
"""

from training.pipelines.training_pipeline import (
    # Main entry point
    train_from_demonstrations,
    quick_train,
    load_artifact,
    # Deployment helpers
    promote_latest,
    find_artifacts,
    # Configuration
    PipelineConfig,
    TrainingArtifact,
)

__all__ = [
    # Main entry point
    "train_from_demonstrations",
    "quick_train",
    "load_artifact",
    # Deployment helpers
    "promote_latest",
    "find_artifacts",
    # Configuration
    "PipelineConfig",
    "TrainingArtifact",
]

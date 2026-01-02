"""
Training Module

RL algorithms, mission suites, parallel training, and unified pipelines.

Main entry point for end-to-end training:
    >>> from training import train_from_demonstrations
    >>> artifact = train_from_demonstrations(
    ...     demo_path="demos/",
    ...     output_dir="artifacts/",
    ...     env=env,
    ... )
"""

from training.suites.mission_suites import MissionSuite
from training.parallel.parallel_trainer import ParallelTrainer

# Unified training pipeline
from training.pipelines import (
    train_from_demonstrations,
    quick_train,
    load_artifact,
    promote_latest,
    find_artifacts,
    PipelineConfig,
    TrainingArtifact,
)

# Combat training (dogfight self-play)
from training.combat import (
    ContinuousDogfightTrainer,
    OpponentPool,
    train_dogfight,
    train_1v1,
    train_swarm,
    CombatAnalyzer,
    analyze_training_run,
)

__all__ = [
    # Mission suites
    "MissionSuite",
    # Parallel training
    "ParallelTrainer",
    # Unified pipeline (main entry point)
    "train_from_demonstrations",
    "quick_train",
    "load_artifact",
    # Deployment helpers
    "promote_latest",
    "find_artifacts",
    # Configuration
    "PipelineConfig",
    "TrainingArtifact",
    # Combat training
    "ContinuousDogfightTrainer",
    "OpponentPool",
    "train_dogfight",
    "train_1v1",
    "train_swarm",
    "CombatAnalyzer",
    "analyze_training_run",
]

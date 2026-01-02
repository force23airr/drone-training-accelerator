"""
Combat Training Module

Self-play training systems for dogfight and air combat scenarios.
Designed for continuous training over days/weeks to discover optimal strategies.

Key Components:
- SelfPlayTrainer: Continuous self-play with opponent pool
- CombatAnalyzer: Strategy extraction and analytics
- OpponentPool: Elo-based matchmaking system
"""

from .selfplay_trainer import (
    ContinuousDogfightTrainer,
    OpponentPool,
    OpponentRecord,
    MatchResult,
    TrainingStats,
    SelfPlayCallback,
    train_dogfight,
    train_1v1,
    train_swarm,
)

from .combat_analytics import (
    CombatAnalyzer,
    CombatAnalytics,
    EngagementType,
    ManeuverType,
    EngagementRecord,
    ManeuverRecord,
    StrategyProfile,
    analyze_training_run,
)

from .drone_memory import (
    DroneMemoryDB,
    DroneStats,
    MistakeRecord,
    AchievementRecord,
    ManeuverStats,
    OpponentModel,
    MistakeType,
    AchievementType,
)

__all__ = [
    # Self-play training
    "ContinuousDogfightTrainer",
    "OpponentPool",
    "OpponentRecord",
    "MatchResult",
    "TrainingStats",
    "SelfPlayCallback",
    "train_dogfight",
    "train_1v1",
    "train_swarm",
    # Combat analytics
    "CombatAnalyzer",
    "CombatAnalytics",
    "EngagementType",
    "ManeuverType",
    "EngagementRecord",
    "ManeuverRecord",
    "StrategyProfile",
    "analyze_training_run",
    # Drone memory
    "DroneMemoryDB",
    "DroneStats",
    "MistakeRecord",
    "AchievementRecord",
    "ManeuverStats",
    "OpponentModel",
    "MistakeType",
    "AchievementType",
]

"""
Imitation Learning Module

Learn from human demonstrations to accelerate drone AI training.

The key insight: Instead of learning from scratch with RL (slow),
start by imitating expert pilots (fast), then improve with RL (superhuman).

Workflow:
1. Collect demonstrations (videos, telemetry, control inputs)
2. Train initial policy via Behavioral Cloning
3. Fine-tune with RL to exceed human performance
4. Optionally use GAIL to learn implicit reward functions

Supported Methods:
- Behavioral Cloning (BC): Supervised learning from demonstrations
- DAgger: Interactive imitation with expert correction
- GAIL: Generative Adversarial Imitation Learning
- AIRL: Adversarial Inverse RL (learns transferable rewards)
- BC + RL: Pre-train with BC, fine-tune with PPO/SAC
"""

from training.imitation.demonstration import (
    Demonstration,
    DemonstrationDataset,
    DemonstrationRecorder,
    load_demonstrations,
    save_demonstrations,
)

from training.imitation.behavioral_cloning import (
    BehavioralCloning,
    train_bc,
    pretrain_from_demos,
)

from training.imitation.gail import (
    GAIL,
    Discriminator,
    train_gail,
)

from training.imitation.video_extractor import (
    VideoTrajectoryExtractor,
    extract_from_video,
    extract_from_flight_log,
)

from training.imitation.hybrid_trainer import (
    HybridILRLTrainer,
    train_from_demonstrations,
)

__all__ = [
    # Demonstrations
    "Demonstration",
    "DemonstrationDataset",
    "DemonstrationRecorder",
    "load_demonstrations",
    "save_demonstrations",
    # Behavioral Cloning
    "BehavioralCloning",
    "train_bc",
    "pretrain_from_demos",
    # GAIL
    "GAIL",
    "Discriminator",
    "train_gail",
    # Video extraction
    "VideoTrajectoryExtractor",
    "extract_from_video",
    "extract_from_flight_log",
    # Hybrid training
    "HybridILRLTrainer",
    "train_from_demonstrations",
]

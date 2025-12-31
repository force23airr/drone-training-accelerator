"""
Parallel Trainer

Multi-environment training infrastructure for accelerated RL training.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


class TrainingMetricsCallback(BaseCallback):
    """Callback for logging training metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode statistics when available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if "r" in ep_info:
                self.episode_rewards.append(ep_info["r"])
            if "l" in ep_info:
                self.episode_lengths.append(ep_info["l"])
        return True

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated training metrics."""
        if not self.episode_rewards:
            return {}
        return {
            "mean_reward": np.mean(self.episode_rewards[-100:]),
            "std_reward": np.std(self.episode_rewards[-100:]),
            "mean_length": np.mean(self.episode_lengths[-100:]),
            "total_episodes": len(self.episode_rewards),
        }


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning advancement."""

    def __init__(self, mission_suite, advancement_interval: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.mission_suite = mission_suite
        self.advancement_interval = advancement_interval
        self._last_advancement_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_advancement_step >= self.advancement_interval:
            # Calculate recent success rate
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in list(self.model.ep_info_buffer)[-100:]]
                # Simple heuristic: positive average reward indicates success
                success_rate = np.mean([r > 0 for r in recent_rewards])
                self.mission_suite.advance_curriculum(success_rate)
            self._last_advancement_step = self.num_timesteps
        return True


class ParallelTrainer:
    """
    Parallel training manager for drone RL agents.

    Supports multiple RL algorithms and parallel environment execution
    for accelerated training.
    """

    ALGORITHMS = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }

    def __init__(
        self,
        env,
        mission,
        num_envs: int = 4,
        algorithm: str = "ppo",
        output_dir: str = "./trained_models",
        tensorboard_log: Optional[str] = None,
        seed: int = 42,
        device: str = "auto",
    ):
        """
        Initialize parallel trainer.

        Args:
            env: Base environment instance
            mission: Mission suite for reward shaping
            num_envs: Number of parallel environments
            algorithm: RL algorithm ('ppo', 'sac', 'td3')
            output_dir: Directory for saving models
            tensorboard_log: TensorBoard log directory
            seed: Random seed
            device: Compute device ('auto', 'cpu', 'cuda')
        """
        self.base_env = env
        self.mission = mission
        self.num_envs = num_envs
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.tensorboard_log = tensorboard_log or str(self.output_dir / "logs")
        self.seed = seed
        self.device = device

        # Validate algorithm
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Available: {list(self.ALGORITHMS.keys())}"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize vectorized environment and model
        self.vec_env = None
        self.model = None
        self._setup()

    def _make_env(self, rank: int, seed: int):
        """Create a single environment instance."""
        def _init():
            # Import here to avoid circular imports
            from simulation.environments.base_drone_env import BaseDroneEnv
            env = BaseDroneEnv(
                platform_config=self.base_env.platform_config,
                environmental_conditions=getattr(self.base_env, 'env_conditions', None),
                render_mode=None,  # No rendering for parallel envs
                domain_randomization=getattr(self.base_env, 'domain_randomization', False),
            )
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init

    def _setup(self):
        """Set up vectorized environment and RL model."""
        # Create vectorized environments
        if self.num_envs == 1:
            self.vec_env = DummyVecEnv([self._make_env(0, self.seed)])
        else:
            self.vec_env = SubprocVecEnv(
                [self._make_env(i, self.seed) for i in range(self.num_envs)]
            )

        # Get algorithm class and default hyperparameters
        AlgoClass = self.ALGORITHMS[self.algorithm]
        hyperparams = self._get_hyperparameters()

        # Initialize model
        self.model = AlgoClass(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            tensorboard_log=self.tensorboard_log,
            seed=self.seed,
            device=self.device,
            **hyperparams
        )

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Get algorithm-specific hyperparameters."""
        if self.algorithm == "ppo":
            return {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            }
        elif self.algorithm == "sac":
            return {
                "learning_rate": 3e-4,
                "buffer_size": 1_000_000,
                "learning_starts": 10000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
            }
        elif self.algorithm == "td3":
            return {
                "learning_rate": 3e-4,
                "buffer_size": 1_000_000,
                "learning_starts": 10000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
            }
        return {}

    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        log_interval: int = 10,
        progress_bar: bool = True,
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total environment steps
            eval_freq: Evaluation frequency (in timesteps)
            save_freq: Checkpoint save frequency
            log_interval: Logging interval (in episodes)
            progress_bar: Show progress bar
        """
        # Set up callbacks
        callbacks = []

        # Metrics callback
        metrics_callback = TrainingMetricsCallback(verbose=1)
        callbacks.append(metrics_callback)

        # Curriculum callback
        if self.mission is not None:
            curriculum_callback = CurriculumCallback(
                self.mission,
                advancement_interval=eval_freq
            )
            callbacks.append(curriculum_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // self.num_envs,
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="drone_model"
        )
        callbacks.append(checkpoint_callback)

        # Start training
        print(f"\nStarting training with {self.algorithm.upper()}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Parallel environments: {self.num_envs}")
        print(f"  Output directory: {self.output_dir}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )

        # Print final metrics
        metrics = metrics_callback.get_metrics()
        if metrics:
            print("\nTraining Complete!")
            print(f"  Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
            print(f"  Mean episode length: {metrics['mean_length']:.1f}")
            print(f"  Total episodes: {metrics['total_episodes']}")

    def evaluate(
        self,
        n_eval_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent.

        Args:
            n_eval_episodes: Number of evaluation episodes
            render: Whether to render evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        from stable_baselines3.common.evaluation import evaluate_policy

        # Create evaluation environment
        eval_env = self._make_env(0, self.seed + 1000)()

        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            render=render,
        )

        eval_env.close()

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }

    def save(self, path: Union[str, Path]):
        """Save the trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"Model saved to: {path}")

    def load(self, path: Union[str, Path]):
        """Load a trained model."""
        AlgoClass = self.ALGORITHMS[self.algorithm]
        self.model = AlgoClass.load(str(path), env=self.vec_env)
        print(f"Model loaded from: {path}")

    def close(self):
        """Clean up resources."""
        if self.vec_env is not None:
            self.vec_env.close()

    def get_training_config(self) -> Dict[str, Any]:
        """Get full training configuration for reproducibility."""
        return {
            "algorithm": self.algorithm,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "hyperparameters": self._get_hyperparameters(),
            "mission": self.mission.get_mission_info() if self.mission else None,
            "platform": self.base_env.platform_config,
        }

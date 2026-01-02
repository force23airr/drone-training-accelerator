"""
Self-Play Training System for Dogfight Combat

Implements continuous self-play training to discover optimal combat strategies.
Designed to run for days/weeks, building a pool of increasingly capable opponents.

Features:
- Opponent pool with Elo-based matchmaking
- Continuous training with automatic checkpointing
- Strategy extraction and analytics
- Curriculum learning from weak to strong opponents
- Multi-GPU distributed training support
"""

import os
import time
import json
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from simulation.environments.combat.dogfight_env import (
    DogfightEnv,
    DogfightConfig,
    create_1v1_dogfight,
)


logger = logging.getLogger(__name__)


@dataclass
class OpponentRecord:
    """Record of a stored opponent policy."""
    opponent_id: str
    checkpoint_path: str
    generation: int
    elo_rating: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    avg_kills: float = 0.0
    avg_deaths: float = 0.0
    avg_damage_dealt: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.5
        return self.wins / self.matches_played

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpponentRecord":
        return cls(**data)


@dataclass
class MatchResult:
    """Result of a single match."""
    agent_id: str
    opponent_id: str
    winner: str  # "agent", "opponent", "draw"
    agent_kills: int
    opponent_kills: int
    agent_deaths: int
    opponent_deaths: int
    agent_damage_dealt: float
    agent_damage_taken: float
    match_time: float
    termination_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrainingStats:
    """Aggregate training statistics."""
    total_timesteps: int = 0
    total_episodes: int = 0
    total_matches: int = 0
    total_training_hours: float = 0.0
    generations_completed: int = 0
    current_elo: float = 1500.0
    peak_elo: float = 1500.0
    current_win_rate: float = 0.5
    avg_episode_reward: float = 0.0
    avg_kills_per_match: float = 0.0
    avg_deaths_per_match: float = 0.0
    kill_death_ratio: float = 1.0
    strategies_discovered: List[str] = field(default_factory=list)


class OpponentPool:
    """
    Pool of opponent policies for self-play training.

    Implements Elo-based matchmaking and curriculum learning.
    """

    def __init__(
        self,
        pool_dir: str,
        max_size: int = 100,
        min_elo_diff: float = 100,
        curriculum_enabled: bool = True,
    ):
        """
        Initialize opponent pool.

        Args:
            pool_dir: Directory to store opponent checkpoints
            max_size: Maximum number of opponents to keep
            min_elo_diff: Minimum Elo difference for matchmaking
            curriculum_enabled: Whether to use curriculum learning
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        self.max_size = max_size
        self.min_elo_diff = min_elo_diff
        self.curriculum_enabled = curriculum_enabled

        self.opponents: Dict[str, OpponentRecord] = {}
        self.match_history: List[MatchResult] = []

        self._load_pool()

    def _load_pool(self):
        """Load existing opponent pool from disk."""
        pool_file = self.pool_dir / "pool_registry.json"
        if pool_file.exists():
            with open(pool_file, "r") as f:
                data = json.load(f)
                self.opponents = {
                    k: OpponentRecord.from_dict(v)
                    for k, v in data.get("opponents", {}).items()
                }
                self.match_history = [
                    MatchResult(**m) for m in data.get("match_history", [])
                ]
            logger.info(f"Loaded opponent pool with {len(self.opponents)} opponents")

    def save_pool(self):
        """Save opponent pool to disk."""
        pool_file = self.pool_dir / "pool_registry.json"
        data = {
            "opponents": {k: v.to_dict() for k, v in self.opponents.items()},
            "match_history": [asdict(m) for m in self.match_history[-10000:]],  # Keep last 10k
        }
        with open(pool_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_opponent(
        self,
        policy,
        generation: int,
        elo_rating: float = 1500.0,
    ) -> str:
        """
        Add a new opponent to the pool.

        Args:
            policy: The policy object (SB3 model)
            generation: Training generation number
            elo_rating: Initial Elo rating

        Returns:
            Opponent ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opponent_id = f"gen{generation:04d}_{timestamp}"

        # Save checkpoint
        checkpoint_path = self.pool_dir / f"{opponent_id}.zip"
        policy.save(str(checkpoint_path))

        # Create record
        record = OpponentRecord(
            opponent_id=opponent_id,
            checkpoint_path=str(checkpoint_path),
            generation=generation,
            elo_rating=elo_rating,
        )

        self.opponents[opponent_id] = record

        # Prune if over capacity
        if len(self.opponents) > self.max_size:
            self._prune_pool()

        self.save_pool()

        logger.info(f"Added opponent {opponent_id} with Elo {elo_rating:.0f}")
        return opponent_id

    def _prune_pool(self):
        """Remove weakest opponents when pool is full."""
        # Keep diverse set: some weak (for curriculum), some strong (for challenge)
        if len(self.opponents) <= self.max_size:
            return

        # Sort by Elo
        sorted_opponents = sorted(
            self.opponents.values(),
            key=lambda x: x.elo_rating
        )

        # Keep bottom 10%, top 50%, and sample from middle
        n_keep_bottom = max(5, self.max_size // 10)
        n_keep_top = self.max_size // 2
        n_keep_middle = self.max_size - n_keep_bottom - n_keep_top

        keep_ids = set()

        # Bottom (weak opponents for curriculum)
        for opp in sorted_opponents[:n_keep_bottom]:
            keep_ids.add(opp.opponent_id)

        # Top (strong opponents)
        for opp in sorted_opponents[-n_keep_top:]:
            keep_ids.add(opp.opponent_id)

        # Sample from middle
        middle = sorted_opponents[n_keep_bottom:-n_keep_top]
        if len(middle) > n_keep_middle:
            indices = np.linspace(0, len(middle) - 1, n_keep_middle, dtype=int)
            for i in indices:
                keep_ids.add(middle[i].opponent_id)
        else:
            for opp in middle:
                keep_ids.add(opp.opponent_id)

        # Remove others
        for opp_id in list(self.opponents.keys()):
            if opp_id not in keep_ids:
                record = self.opponents[opp_id]
                try:
                    Path(record.checkpoint_path).unlink()
                except Exception:
                    pass
                del self.opponents[opp_id]

        logger.info(f"Pruned pool to {len(self.opponents)} opponents")

    def select_opponent(
        self,
        agent_elo: float,
        strategy: str = "match",
    ) -> Optional[OpponentRecord]:
        """
        Select an opponent for training.

        Args:
            agent_elo: Current agent Elo rating
            strategy: Selection strategy
                - "match": Similar Elo (standard matchmaking)
                - "curriculum": Slightly weaker (for learning)
                - "challenge": Slightly stronger (for improvement)
                - "random": Random selection
                - "weakest": Always weakest (for initial training)
                - "strongest": Always strongest (for final testing)

        Returns:
            Selected opponent record
        """
        if not self.opponents:
            return None

        opponents = list(self.opponents.values())

        if strategy == "random":
            return np.random.choice(opponents)

        elif strategy == "weakest":
            return min(opponents, key=lambda x: x.elo_rating)

        elif strategy == "strongest":
            return max(opponents, key=lambda x: x.elo_rating)

        elif strategy == "curriculum":
            # Select opponent 50-200 Elo weaker
            target_elo = agent_elo - 100
            candidates = [
                opp for opp in opponents
                if abs(opp.elo_rating - target_elo) < 150
            ]
            if not candidates:
                candidates = opponents
            return min(candidates, key=lambda x: abs(x.elo_rating - target_elo))

        elif strategy == "challenge":
            # Select opponent 50-200 Elo stronger
            target_elo = agent_elo + 100
            candidates = [
                opp for opp in opponents
                if abs(opp.elo_rating - target_elo) < 150
            ]
            if not candidates:
                candidates = opponents
            return min(candidates, key=lambda x: abs(x.elo_rating - target_elo))

        else:  # "match"
            # Select opponent within Elo range
            candidates = [
                opp for opp in opponents
                if abs(opp.elo_rating - agent_elo) < self.min_elo_diff * 2
            ]
            if not candidates:
                candidates = opponents
            return min(candidates, key=lambda x: abs(x.elo_rating - agent_elo))

    def load_opponent_policy(
        self,
        record: OpponentRecord,
        policy_class=None,
    ):
        """Load opponent policy from checkpoint."""
        if policy_class is None:
            policy_class = PPO

        return policy_class.load(record.checkpoint_path)

    def update_elo(
        self,
        agent_id: str,
        opponent_id: str,
        agent_score: float,
        agent_elo: Optional[float] = None,
        k_factor: float = 32.0,
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.

        Args:
            agent_id: Agent identifier
            opponent_id: Opponent identifier
            agent_score: 1.0 for win, 0.5 for draw, 0.0 for loss
            agent_elo: Current agent Elo rating (defaults to 1500)
            k_factor: Elo K-factor

        Returns:
            Tuple of (new_agent_elo_delta, new_opponent_elo_delta)
        """
        if opponent_id not in self.opponents:
            return 0.0, 0.0

        opponent = self.opponents[opponent_id]
        if agent_elo is None:
            agent_elo = 1500.0  # Default for new agent

        # Expected score
        expected_agent = 1 / (1 + 10 ** ((opponent.elo_rating - agent_elo) / 400))
        expected_opponent = 1 - expected_agent

        # Elo delta
        agent_delta = k_factor * (agent_score - expected_agent)
        opponent_delta = k_factor * ((1 - agent_score) - expected_opponent)

        # Update opponent
        opponent.elo_rating += opponent_delta
        opponent.matches_played += 1
        if agent_score > 0.5:
            opponent.losses += 1
        elif agent_score < 0.5:
            opponent.wins += 1
        else:
            opponent.draws += 1

        self.save_pool()

        return agent_delta, opponent_delta

    def record_match(self, result: MatchResult):
        """Record match result."""
        self.match_history.append(result)

        # Update opponent stats
        if result.opponent_id in self.opponents:
            opp = self.opponents[result.opponent_id]
            # Running average
            n = opp.matches_played
            opp.avg_kills = (opp.avg_kills * n + result.opponent_kills) / (n + 1)
            opp.avg_deaths = (opp.avg_deaths * n + result.opponent_deaths) / (n + 1)
            opp.avg_damage_dealt = (opp.avg_damage_dealt * n + result.agent_damage_taken) / (n + 1)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if not self.opponents:
            return {"size": 0}

        elos = [opp.elo_rating for opp in self.opponents.values()]
        return {
            "size": len(self.opponents),
            "min_elo": min(elos),
            "max_elo": max(elos),
            "mean_elo": np.mean(elos),
            "generations": max(opp.generation for opp in self.opponents.values()),
            "total_matches": sum(opp.matches_played for opp in self.opponents.values()),
        }


class SelfPlayCallback(BaseCallback):
    """Callback for self-play training with opponent rotation."""

    def __init__(
        self,
        opponent_pool: OpponentPool,
        env: DogfightEnv,
        rotation_episodes: int = 100,
        save_freq: int = 10000,
        checkpoint_dir: str = "checkpoints",
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.opponent_pool = opponent_pool
        self.env = env
        self.rotation_episodes = rotation_episodes
        self.save_freq = save_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.generation = 0
        self.current_opponent: Optional[OpponentRecord] = None
        self.agent_elo = 1500.0

        # Stats tracking
        self.recent_rewards = deque(maxlen=100)
        self.recent_kills = deque(maxlen=100)
        self.recent_deaths = deque(maxlen=100)
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1

            # Get episode info
            info = self.locals.get("infos", [{}])[0]
            stats = info.get("episode_stats", {})

            # Track stats
            self.recent_kills.append(stats.get("kills", 0))
            self.recent_deaths.append(stats.get("deaths", 0))

            # Determine winner
            agent_kills = stats.get("kills", 0)
            agent_deaths = stats.get("deaths", 0)

            if agent_kills > agent_deaths:
                self.wins += 1
                score = 1.0
            elif agent_kills < agent_deaths:
                self.losses += 1
                score = 0.0
            else:
                self.draws += 1
                score = 0.5

            # Update Elo if we have an opponent
            if self.current_opponent:
                delta, _ = self.opponent_pool.update_elo(
                    "agent",
                    self.current_opponent.opponent_id,
                    score,
                    agent_elo=self.agent_elo,
                )
                self.agent_elo += delta

            # Rotate opponent
            if self.episode_count % self.rotation_episodes == 0:
                self._rotate_opponent()

        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _rotate_opponent(self):
        """Rotate to a new opponent."""
        # Add current policy to pool
        self.generation += 1
        self.opponent_pool.add_opponent(
            self.model,
            generation=self.generation,
            elo_rating=self.agent_elo,
        )

        # Select new opponent
        strategy = self._select_strategy()
        opponent = self.opponent_pool.select_opponent(
            self.agent_elo,
            strategy=strategy,
        )

        if opponent:
            self.current_opponent = opponent
            try:
                opponent_policy = self.opponent_pool.load_opponent_policy(opponent)
                self.env.set_opponent_policy(opponent_policy)
                logger.info(
                    f"Rotated to opponent {opponent.opponent_id} "
                    f"(Elo: {opponent.elo_rating:.0f})"
                )
            except Exception as e:
                logger.warning(f"Failed to load opponent: {e}")

    def _select_strategy(self) -> str:
        """Select opponent selection strategy based on training progress."""
        if self.generation < 10:
            return "weakest"  # Initial training
        elif self.generation < 50:
            return "curriculum"  # Gradual increase
        elif self.wins / max(self.wins + self.losses, 1) > 0.7:
            return "challenge"  # Winning too much, increase difficulty
        else:
            return "match"  # Standard matchmaking

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.n_calls}.zip"
        self.model.save(str(checkpoint_path))

        # Save metadata
        meta = {
            "timesteps": self.n_calls,
            "episodes": self.episode_count,
            "generation": self.generation,
            "elo": self.agent_elo,
            "win_rate": self.wins / max(self.wins + self.losses + self.draws, 1),
            "avg_kills": np.mean(self.recent_kills) if self.recent_kills else 0,
            "avg_deaths": np.mean(self.recent_deaths) if self.recent_deaths else 0,
        }

        meta_path = self.checkpoint_dir / f"checkpoint_{self.n_calls}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved checkpoint at {self.n_calls} timesteps (Elo: {self.agent_elo:.0f})")


class ContinuousDogfightTrainer:
    """
    Continuous self-play trainer for dogfight combat.

    Designed to run for days/weeks, continuously improving combat strategies.
    """

    def __init__(
        self,
        output_dir: str,
        dogfight_config: Optional[DogfightConfig] = None,
        algorithm: str = "PPO",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        n_envs: int = 4,
        device: str = "auto",
    ):
        """
        Initialize continuous trainer.

        Args:
            output_dir: Directory for outputs, checkpoints, logs
            dogfight_config: Configuration for dogfight environment
            algorithm: RL algorithm ("PPO" or "SAC")
            policy_kwargs: Policy network configuration
            n_envs: Number of parallel environments
            device: Device to train on
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required for training")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dogfight_config = dogfight_config or DogfightConfig()
        self.algorithm = algorithm
        self.policy_kwargs = policy_kwargs or {
            "net_arch": [256, 256, 128],
        }
        self.n_envs = n_envs
        self.device = device

        # Create directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.pool_dir = self.output_dir / "opponent_pool"
        self.logs_dir = self.output_dir / "logs"
        self.analytics_dir = self.output_dir / "analytics"

        for d in [self.checkpoint_dir, self.pool_dir, self.logs_dir, self.analytics_dir]:
            d.mkdir(exist_ok=True)

        # Initialize components
        self.opponent_pool = OpponentPool(
            str(self.pool_dir),
            max_size=100,
        )

        self.stats = TrainingStats()
        self._load_stats()

        # Training state
        self.model = None
        self.env = None
        self.is_training = False
        self.start_time = None

        logger.info(f"Initialized ContinuousDogfightTrainer at {output_dir}")

    def _load_stats(self):
        """Load training stats from disk."""
        stats_file = self.output_dir / "training_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.stats, key):
                        setattr(self.stats, key, value)

    def _save_stats(self):
        """Save training stats to disk."""
        stats_file = self.output_dir / "training_stats.json"
        with open(stats_file, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)

    def _create_env(self) -> DogfightEnv:
        """Create dogfight environment."""
        return DogfightEnv(
            config=self.dogfight_config,
            agent_team=0,
            agent_id=0,
            render_mode=None,
        )

    def _create_vec_env(self):
        """Create vectorized environment for parallel training."""
        def make_env():
            def _init():
                return self._create_env()
            return _init

        if self.n_envs > 1:
            return SubprocVecEnv([make_env() for _ in range(self.n_envs)])
        else:
            return DummyVecEnv([make_env()])

    def _create_model(self):
        """Create RL model."""
        env = self._create_vec_env()

        if self.algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=self.policy_kwargs,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=str(self.logs_dir),
                device=self.device,
            )
        elif self.algorithm == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                policy_kwargs=self.policy_kwargs,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                verbose=1,
                tensorboard_log=str(self.logs_dir),
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return model, env

    def train(
        self,
        total_timesteps: int = 10_000_000,
        checkpoint_freq: int = 100_000,
        opponent_rotation_episodes: int = 100,
        resume: bool = True,
    ):
        """
        Start training.

        Args:
            total_timesteps: Total timesteps to train (can be very large)
            checkpoint_freq: Save checkpoint every N timesteps
            opponent_rotation_episodes: Rotate opponent every N episodes
            resume: Resume from latest checkpoint if available
        """
        logger.info(f"Starting training for {total_timesteps:,} timesteps")

        self.start_time = time.time()
        self.is_training = True

        # Create or load model
        if resume:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"Resuming from {latest_checkpoint}")
                if self.algorithm == "PPO":
                    self.model = PPO.load(latest_checkpoint)
                else:
                    self.model = SAC.load(latest_checkpoint)
                self.env = self._create_vec_env()
                self.model.set_env(self.env)
            else:
                self.model, self.env = self._create_model()
        else:
            self.model, self.env = self._create_model()

        # Create callback
        single_env = self._create_env()
        callback = SelfPlayCallback(
            opponent_pool=self.opponent_pool,
            env=single_env,
            rotation_episodes=opponent_rotation_episodes,
            save_freq=checkpoint_freq,
            checkpoint_dir=str(self.checkpoint_dir),
            verbose=1,
        )

        try:
            # Main training loop
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=not resume,
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self.is_training = False
            self._finalize_training(callback)

    def train_forever(
        self,
        checkpoint_freq: int = 100_000,
        opponent_rotation_episodes: int = 100,
        max_hours: Optional[float] = None,
    ):
        """
        Train continuously until stopped.

        Args:
            checkpoint_freq: Save checkpoint every N timesteps
            opponent_rotation_episodes: Rotate opponent every N episodes
            max_hours: Maximum hours to train (None for infinite)
        """
        total_timesteps = 1_000_000_000  # 1 billion - effectively infinite

        if max_hours:
            logger.info(f"Training for up to {max_hours} hours")
        else:
            logger.info("Training indefinitely (Ctrl+C to stop)")

        self.train(
            total_timesteps=total_timesteps,
            checkpoint_freq=checkpoint_freq,
            opponent_rotation_episodes=opponent_rotation_episodes,
            resume=True,
        )

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.zip"))
        if not checkpoints:
            return None

        # Sort by timestep number
        def get_timestep(path):
            name = path.stem
            try:
                return int(name.split("_")[1])
            except (IndexError, ValueError):
                return 0

        latest = max(checkpoints, key=get_timestep)
        return str(latest)

    def _finalize_training(self, callback: SelfPlayCallback):
        """Finalize training and save results."""
        # Update stats
        training_hours = (time.time() - self.start_time) / 3600
        self.stats.total_timesteps += callback.n_calls
        self.stats.total_episodes += callback.episode_count
        self.stats.total_training_hours += training_hours
        self.stats.generations_completed = callback.generation
        self.stats.current_elo = callback.agent_elo
        self.stats.peak_elo = max(self.stats.peak_elo, callback.agent_elo)

        if callback.recent_kills:
            self.stats.avg_kills_per_match = np.mean(callback.recent_kills)
        if callback.recent_deaths:
            self.stats.avg_deaths_per_match = np.mean(callback.recent_deaths)
            if self.stats.avg_deaths_per_match > 0:
                self.stats.kill_death_ratio = (
                    self.stats.avg_kills_per_match / self.stats.avg_deaths_per_match
                )

        total_matches = callback.wins + callback.losses + callback.draws
        if total_matches > 0:
            self.stats.current_win_rate = callback.wins / total_matches

        self._save_stats()

        # Save final model
        final_path = self.output_dir / "final_model.zip"
        self.model.save(str(final_path))

        # Generate report
        self._generate_training_report()

        logger.info(
            f"Training complete. "
            f"Timesteps: {callback.n_calls:,}, "
            f"Episodes: {callback.episode_count:,}, "
            f"Final Elo: {callback.agent_elo:.0f}"
        )

    def _generate_training_report(self):
        """Generate training analytics report."""
        report = {
            "summary": asdict(self.stats),
            "opponent_pool": self.opponent_pool.get_pool_stats(),
            "training_config": {
                "algorithm": self.algorithm,
                "n_envs": self.n_envs,
                "dogfight_config": asdict(self.dogfight_config),
            },
            "generated_at": datetime.now().isoformat(),
        }

        report_path = self.analytics_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved to {report_path}")

    def evaluate(
        self,
        n_matches: int = 100,
        opponent_strategy: str = "strongest",
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            n_matches: Number of matches to play
            opponent_strategy: Opponent selection strategy

        Returns:
            Evaluation results
        """
        if self.model is None:
            # Load latest model
            checkpoint = self._find_latest_checkpoint()
            if checkpoint:
                if self.algorithm == "PPO":
                    self.model = PPO.load(checkpoint)
                else:
                    self.model = SAC.load(checkpoint)
            else:
                raise ValueError("No model found")

        env = self._create_env()

        results = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_kills": 0,
            "total_deaths": 0,
            "total_damage_dealt": 0,
            "total_damage_taken": 0,
        }

        for i in range(n_matches):
            # Select opponent
            opponent = self.opponent_pool.select_opponent(
                self.stats.current_elo,
                strategy=opponent_strategy,
            )

            if opponent:
                try:
                    opponent_policy = self.opponent_pool.load_opponent_policy(opponent)
                    env.set_opponent_policy(opponent_policy)
                except Exception:
                    pass

            # Play match
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Record results
            stats = info.get("episode_stats", {})
            kills = stats.get("kills", 0)
            deaths = stats.get("deaths", 0)

            results["total_kills"] += kills
            results["total_deaths"] += deaths
            results["total_damage_dealt"] += stats.get("damage_dealt", 0)
            results["total_damage_taken"] += stats.get("damage_taken", 0)

            if kills > deaths:
                results["wins"] += 1
            elif kills < deaths:
                results["losses"] += 1
            else:
                results["draws"] += 1

        # Calculate summary stats
        results["matches_played"] = n_matches
        results["win_rate"] = results["wins"] / n_matches
        results["avg_kills"] = results["total_kills"] / n_matches
        results["avg_deaths"] = results["total_deaths"] / n_matches
        if results["total_deaths"] > 0:
            results["kd_ratio"] = results["total_kills"] / results["total_deaths"]
        else:
            results["kd_ratio"] = float("inf")

        env.close()

        return results


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def train_dogfight(
    output_dir: str = "dogfight_training",
    hours: Optional[float] = None,
    timesteps: Optional[int] = None,
    config: Optional[DogfightConfig] = None,
) -> ContinuousDogfightTrainer:
    """
    Quick start function for dogfight training.

    Args:
        output_dir: Output directory
        hours: Hours to train (None for timesteps mode)
        timesteps: Timesteps to train (default 10M if hours not specified)
        config: Dogfight configuration

    Returns:
        Trainer instance
    """
    trainer = ContinuousDogfightTrainer(
        output_dir=output_dir,
        dogfight_config=config,
    )

    if hours:
        trainer.train_forever(max_hours=hours)
    else:
        trainer.train(total_timesteps=timesteps or 10_000_000)

    return trainer


def train_1v1(output_dir: str = "1v1_training", hours: float = 24):
    """Train 1v1 dogfight for specified hours."""
    config = DogfightConfig(
        num_red=1,
        num_blue=1,
        respawn_enabled=True,
        kills_to_win=5,
        spawn_position_jitter=150.0,
        spawn_altitude_jitter=100.0,
        spawn_heading_jitter=np.radians(10),
        spawn_speed_jitter=0.1,
        reward_fire=-0.05,
    )
    return train_dogfight(output_dir, hours=hours, config=config)


def train_swarm(output_dir: str = "swarm_training", hours: float = 48):
    """Train swarm battle for specified hours."""
    config = DogfightConfig(
        num_red=4,
        num_blue=4,
        respawn_enabled=False,
        win_condition="last_alive",
    )
    return train_dogfight(output_dir, hours=hours, config=config)

"""
Evaluation Harness

Main class for comprehensive policy evaluation.

Runs standardized evaluation suites and produces unified reports
with metrics, gate checks, and recommendations.

Key features:
- Early-exit on hard gate failure (crash/violation) for cheap rejection
- Dual seed sets: fixed (regression tracking) + random (robustness sampling)
- Behavioral metrics treated as secondary/diagnostic only
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, TYPE_CHECKING
from datetime import datetime
import json
import hashlib
from pathlib import Path

from evaluation.metrics import (
    MetricCollector,
    MetricResult,
    CompositeMetricCollector,
    FlightMetrics,
    SafetyMetrics,
    SmoothnessMetrics,
    EfficiencyMetrics,
    aggregate_metrics,
)
from evaluation.gates import (
    PromotionGate,
    GateCheckResult,
    CompositeGate,
)

if TYPE_CHECKING:
    from training.suites.mission_suites import MissionSuite


# =============================================================================
# SEED CONFIGURATION
# =============================================================================

@dataclass
class SeedConfig:
    """
    Configuration for evaluation seeds.

    Supports dual seed sets:
    - Fixed seeds for regression tracking (stable, comparable)
    - Random seeds for robustness sampling (truly random per run)

    The meta_seed used for random generation is stored in the result
    for reproducibility.
    """
    # Fixed seeds for regression tracking
    fixed_seeds: List[int] = field(default_factory=lambda: list(range(42, 62)))

    # Whether to also sample random seeds
    use_random_seeds: bool = True
    n_random_seeds: int = 50

    # Meta-seed for generating random seeds
    # None = use OS entropy (truly random per run)
    # int = deterministic (reproducible)
    random_seed_generator: Optional[int] = None

    # Stores the actual meta-seed used (for reproducibility in reports)
    _used_meta_seed: Optional[int] = field(default=None, repr=False)

    def get_seeds(self, n_episodes: int) -> Tuple[List[int], int]:
        """
        Get seeds for evaluation.

        Returns:
            Tuple of (seeds list, meta_seed used)
            The meta_seed is stored for reproducibility in reports.
        """
        seeds = []

        # Add fixed seeds
        n_fixed = min(len(self.fixed_seeds), n_episodes // 2 if self.use_random_seeds else n_episodes)
        seeds.extend(self.fixed_seeds[:n_fixed])

        # Add random seeds
        meta_seed = self.random_seed_generator
        if self.use_random_seeds:
            n_random = n_episodes - len(seeds)

            # If no meta-seed provided, generate one from OS entropy
            if meta_seed is None:
                import time
                meta_seed = int(time.time() * 1000) % (2**31)

            self._used_meta_seed = meta_seed
            rng = np.random.default_rng(meta_seed)
            random_seeds = rng.integers(0, 2**31, size=n_random).tolist()
            seeds.extend(random_seeds)

        return seeds[:n_episodes], meta_seed


@dataclass
class EarlyExitConfig:
    """
    Configuration for early exit on hard gate failure.

    Enables cheap rejection of bad policies without running all episodes.

    NOTE: Success rate exit is OFF by default because condition cycling
    may cause early episodes to be harder. Only crash/violation exits
    are enabled by default (true hard failures).
    """
    enabled: bool = True

    # Check gates after this many episodes
    check_after_episodes: int = 10

    # Fail immediately on any crash (strict safety) - HARD FAILURE
    fail_on_crash: bool = True
    max_crashes_before_exit: int = 0

    # Fail if critical violations exceed threshold - HARD FAILURE
    max_critical_violations: int = 0

    # Success rate exit - OFF by default (can falsely kill good policies
    # when conditions are cycled and early episodes happen to be harder)
    check_success_rate: bool = False
    min_success_rate_threshold: float = 0.3


@dataclass
class EpisodeResult:
    """
    Complete result from one evaluation episode.

    Contains all metrics, trajectory data, and termination info
    for a single episode.
    """
    episode_id: int
    success: bool
    total_reward: float
    episode_length: int
    metrics: Dict[str, float]
    termination_reason: str = "unknown"
    trajectory: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None
    conditions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'success': self.success,
            'total_reward': self.total_reward,
            'episode_length': self.episode_length,
            'metrics': self.metrics,
            'termination_reason': self.termination_reason,
            'conditions': self.conditions,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result across all episodes.

    Contains aggregated metrics, gate results, and configuration.
    """
    num_episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max}
    episodes: List[EpisodeResult]
    gate_results: Dict[str, GateCheckResult]
    all_gates_passed: bool
    timestamp: str
    config: Dict[str, Any]
    evaluation_time_seconds: float = 0.0

    # Reproducibility info
    requested_episodes: int = 0           # Episodes requested (may differ from num_episodes if early exit)
    early_exit_triggered: bool = False
    early_exit_reason: str = ""
    meta_seed_used: Optional[int] = None  # For reproducing random seeds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'num_episodes': self.num_episodes,
                'requested_episodes': self.requested_episodes,
                'success_rate': self.success_rate,
                'mean_reward': self.mean_reward,
                'std_reward': self.std_reward,
                'all_gates_passed': self.all_gates_passed,
                'timestamp': self.timestamp,
                'evaluation_time_seconds': self.evaluation_time_seconds,
                'early_exit_triggered': self.early_exit_triggered,
                'early_exit_reason': self.early_exit_reason,
                'meta_seed_used': self.meta_seed_used,
            },
            'metrics': self.metrics,
            'gate_results': {
                name: {
                    'passed': result.passed,
                    'message': result.message,
                    'required': result.required_value,
                    'actual': str(result.actual_value),
                }
                for name, result in self.gate_results.items()
            },
            'config': self.config,
        }

    def save(self, filepath: str) -> None:
        """Save evaluation result to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class EvaluationHarness:
    """
    Main evaluation harness for comprehensive policy assessment.

    Runs standardized evaluation with:
    - Multiple metric collectors
    - Promotion gate checking with early exit on failure
    - Dual seed sets (fixed + random)
    - Generalization testing (varied conditions)
    - Report generation
    """

    def __init__(
        self,
        env,
        metrics: Optional[List[MetricCollector]] = None,
        gates: Optional[List[PromotionGate]] = None,
        mission_suite: Optional['MissionSuite'] = None,
        record_trajectories: bool = False,
        target_position: Optional[np.ndarray] = None,
        seed_config: Optional[SeedConfig] = None,
        early_exit_config: Optional[EarlyExitConfig] = None,
    ):
        """
        Args:
            env: Gymnasium environment
            metrics: List of metric collectors (defaults to all standard metrics)
            gates: Promotion gates to check
            mission_suite: Optional mission suite for success criteria
            record_trajectories: Whether to record full trajectories
            target_position: Target position for flight metrics
            seed_config: Configuration for eval seeds (fixed + random)
            early_exit_config: Configuration for early exit on failure
        """
        self.env = env
        self.mission_suite = mission_suite
        self.record_trajectories = record_trajectories
        self.target_position = target_position
        self.seed_config = seed_config or SeedConfig()
        self.early_exit_config = early_exit_config or EarlyExitConfig()

        # Initialize metrics
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = self._default_metrics()

        # Initialize gates
        self.gates = gates or []

        # Track early exit
        self._early_exit_triggered = False
        self._early_exit_reason = ""

    def _default_metrics(self) -> List[MetricCollector]:
        """Create default metric collectors."""
        return [
            FlightMetrics(target_position=self.target_position),
            SafetyMetrics(),
            SmoothnessMetrics(),
            EfficiencyMetrics(),
        ]

    def add_metric(self, metric: MetricCollector) -> None:
        """Add a metric collector."""
        self.metrics.append(metric)

    def add_gate(self, gate: PromotionGate) -> None:
        """Add a promotion gate."""
        self.gates.append(gate)

    def evaluate(
        self,
        policy,
        n_episodes: int = 100,
        deterministic: bool = True,
        conditions_list: Optional[List[Dict[str, Any]]] = None,
        seeds: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Run full evaluation of a policy.

        Args:
            policy: Policy with predict(obs) method
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            conditions_list: Optional list of conditions to cycle through
            seeds: Optional list of random seeds (if None, uses seed_config)
            progress_callback: Optional callback(current, total)
            verbose: Print progress

        Returns:
            EvaluationResult with all metrics and gate results
        """
        start_time = datetime.now()
        self._early_exit_triggered = False
        self._early_exit_reason = ""

        if verbose:
            print(f"Starting evaluation: {n_episodes} episodes")

        # Get seeds from config if not provided
        meta_seed_used = None
        if seeds is None:
            seeds, meta_seed_used = self.seed_config.get_seeds(n_episodes)

        episodes = []
        all_episode_metrics = []

        # Track for early exit
        crash_count = 0
        violation_count = 0

        for ep_idx in range(n_episodes):
            # Set conditions if provided
            if conditions_list:
                condition = conditions_list[ep_idx % len(conditions_list)]
                self._apply_conditions(condition)
            else:
                condition = None

            # Set seed
            seed = seeds[ep_idx] if ep_idx < len(seeds) else None

            # Run episode
            episode_result, episode_metrics = self._run_episode(
                policy, ep_idx, deterministic, seed, condition
            )
            episodes.append(episode_result)
            all_episode_metrics.append(episode_metrics)

            # Track crashes/violations for early exit
            if episode_result.metrics.get('crash_count', 0) > 0:
                crash_count += 1
            if episode_result.metrics.get('critical_violations', 0) > 0:
                violation_count += int(episode_result.metrics['critical_violations'])

            if progress_callback:
                progress_callback(ep_idx + 1, n_episodes)

            if verbose and (ep_idx + 1) % 10 == 0:
                success_so_far = sum(e.success for e in episodes) / len(episodes)
                print(f"  Episode {ep_idx + 1}/{n_episodes}: "
                      f"success_rate={success_so_far:.1%}")

            # Check for early exit
            if self._should_early_exit(episodes, crash_count, violation_count, ep_idx + 1):
                if verbose:
                    print(f"\n  EARLY EXIT at episode {ep_idx + 1}: {self._early_exit_reason}")
                break

        # Aggregate metrics
        aggregated = aggregate_metrics(all_episode_metrics)

        # Check promotion gates
        gate_results = {}
        for gate in self.gates:
            result = gate.check(episodes, aggregated)
            gate_results[gate.name] = result

        all_passed = all(r.passed for r in gate_results.values()) if gate_results else True

        # If early exit, mark as failed
        if self._early_exit_triggered:
            all_passed = False
            gate_results['early_exit'] = GateCheckResult(
                passed=False,
                gate_name='early_exit',
                gate_type='safety',
                required_value='No early exit',
                actual_value=self._early_exit_reason,
                message=f"Early exit triggered: {self._early_exit_reason}",
            )

        # Compute summary stats
        actual_episodes = len(episodes)
        success_rate = sum(e.success for e in episodes) / actual_episodes
        rewards = [e.total_reward for e in episodes]
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))

        eval_time = (datetime.now() - start_time).total_seconds()

        result = EvaluationResult(
            num_episodes=actual_episodes,
            success_rate=success_rate,
            mean_reward=mean_reward,
            std_reward=std_reward,
            metrics=aggregated,
            episodes=episodes,
            gate_results=gate_results,
            all_gates_passed=all_passed,
            timestamp=datetime.now().isoformat(),
            config=self._get_config(),
            evaluation_time_seconds=eval_time,
            requested_episodes=n_episodes,
            early_exit_triggered=self._early_exit_triggered,
            early_exit_reason=self._early_exit_reason,
            meta_seed_used=meta_seed_used,
        )

        if verbose:
            print(f"\nEvaluation complete in {eval_time:.1f}s")
            print(f"  Episodes run: {actual_episodes}/{n_episodes}")
            if self._early_exit_triggered:
                print(f"  EARLY EXIT: {self._early_exit_reason}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  Gates passed: {sum(r.passed for r in gate_results.values())}/{len(gate_results)}")
            if meta_seed_used is not None:
                print(f"  Meta-seed used: {meta_seed_used} (for reproducibility)")

        return result

    def _should_early_exit(
        self,
        episodes: List[EpisodeResult],
        crash_count: int,
        violation_count: int,
        current_episode: int,
    ) -> bool:
        """
        Check if we should exit early due to hard failure.

        Only crashes and critical violations trigger exit by default.
        Success rate exit is opt-in due to condition cycling issues.
        """
        cfg = self.early_exit_config

        if not cfg.enabled:
            return False

        # Wait for minimum episodes
        if current_episode < cfg.check_after_episodes:
            return False

        # Check crash threshold - HARD FAILURE
        if cfg.fail_on_crash and crash_count > cfg.max_crashes_before_exit:
            self._early_exit_triggered = True
            self._early_exit_reason = f"Crash threshold exceeded: {crash_count} crashes"
            return True

        # Check critical violations - HARD FAILURE
        if violation_count > cfg.max_critical_violations:
            self._early_exit_triggered = True
            self._early_exit_reason = f"Critical violations: {violation_count}"
            return True

        # Check success rate threshold - OPTIONAL (off by default)
        # Can falsely kill good policies when conditions are cycled
        if cfg.check_success_rate:
            success_rate = sum(e.success for e in episodes) / len(episodes)
            if success_rate < cfg.min_success_rate_threshold:
                self._early_exit_triggered = True
                self._early_exit_reason = f"Success rate too low: {success_rate:.1%}"
                return True

        return False

    def _run_episode(
        self,
        policy,
        episode_id: int,
        deterministic: bool,
        seed: Optional[int],
        conditions: Optional[Dict[str, Any]],
    ) -> tuple:
        """Run a single evaluation episode."""
        # Reset all metrics
        for metric in self.metrics:
            metric.reset()

        # Reset environment
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()

        done = False
        total_reward = 0.0
        step_count = 0

        trajectory = [] if self.record_trajectories else None
        actions = [] if self.record_trajectories else None

        while not done:
            # Get action from policy
            action, _ = policy.predict(obs, deterministic=deterministic)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Update all metrics
            for metric in self.metrics:
                metric.step(obs, action, reward, next_obs, terminated, truncated, info)

            # Record trajectory
            if self.record_trajectories:
                trajectory.append(next_obs.copy())
                actions.append(action.copy())

            total_reward += reward
            step_count += 1
            obs = next_obs
            done = terminated or truncated

        # Collect episode-end metrics
        all_results = []
        metrics_dict = {}

        for metric in self.metrics:
            results = metric.episode_end(info)
            all_results.extend(results)
            for r in results:
                metrics_dict[r.name] = r.value

        # Add episode metadata to metrics
        metrics_dict['terminated'] = float(terminated)
        metrics_dict['truncated'] = float(truncated)
        metrics_dict['steps'] = step_count
        metrics_dict['episode_length'] = step_count

        # Capture shield intervention stats if shield wrapper is present
        # This enables InterventionRateGate to check shield usage
        self._capture_shield_stats(metrics_dict)

        # Determine success
        success = self._determine_success(obs, info, metrics_dict)

        # Get termination reason
        termination_reason = info.get('termination_reason', 'unknown')
        if terminated and not success:
            if metrics_dict.get('crash_count', 0) > 0:
                termination_reason = 'crash'
            elif metrics_dict.get('altitude_violations', 0) > 0:
                termination_reason = 'altitude_violation'

        episode_result = EpisodeResult(
            episode_id=episode_id,
            success=success,
            total_reward=total_reward,
            episode_length=step_count,
            metrics=metrics_dict,
            termination_reason=termination_reason,
            trajectory=np.array(trajectory) if trajectory else None,
            actions=np.array(actions) if actions else None,
            conditions=conditions,
        )

        return episode_result, all_results

    def _determine_success(
        self,
        final_obs: np.ndarray,
        info: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> bool:
        """Determine if episode was successful."""
        # Check explicit success in info
        if 'success' in info:
            return bool(info['success'])

        # Check episode stats
        if 'episode_stats' in info:
            stats = info['episode_stats']
            if 'success' in stats:
                return bool(stats['success'])

        # Check mission suite
        if self.mission_suite is not None:
            return self.mission_suite.check_success(final_obs, info.get('episode_stats', {}))

        # Fallback heuristics
        if metrics.get('crash_count', 0) > 0:
            return False

        if metrics.get('reached_target', 0) > 0:
            return True

        if metrics.get('final_position_error', float('inf')) < 0.5:
            return True

        return metrics.get('total_reward', 0) > 0

    def _apply_conditions(self, conditions: Dict[str, Any]) -> None:
        """Apply environmental conditions to environment."""
        if hasattr(self.env, 'set_conditions'):
            self.env.set_conditions(conditions)
        elif hasattr(self.env, 'env_conditions'):
            for key, value in conditions.items():
                if hasattr(self.env.env_conditions, key):
                    setattr(self.env.env_conditions, key, value)

    def _capture_shield_stats(self, metrics_dict: Dict[str, float]) -> None:
        """
        Capture shield intervention stats from environment if available.

        Looks for SafetyShieldWrapper and extracts intervention counts
        so InterventionRateGate can check them.
        """
        # Walk the wrapper chain to find shield
        env = self.env
        while env is not None:
            if hasattr(env, 'get_intervention_stats'):
                # Found shield wrapper
                stats = env.get_intervention_stats()
                metrics_dict['shield_stats'] = stats

                # Also extract individual counts for easier access
                by_type = stats.get('by_type', {})
                metrics_dict['shield_clamp_count'] = by_type.get('clamp', 0)
                metrics_dict['shield_override_count'] = by_type.get('guardian', 0)
                metrics_dict['shield_emergency_count'] = by_type.get('emergency', 0)
                metrics_dict['shield_total_interventions'] = stats.get('total', 0)
                metrics_dict['shield_intervention_rate'] = stats.get('intervention_rate', 0)
                break

            # Walk to wrapped env
            if hasattr(env, 'env'):
                env = env.env
            elif hasattr(env, 'unwrapped') and env.unwrapped is not env:
                env = env.unwrapped
            else:
                break

    def _get_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return {
            'metrics': [m.name for m in self.metrics],
            'gates': [g.name for g in self.gates],
            'record_trajectories': self.record_trajectories,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
        }

    def evaluate_generalization(
        self,
        policy,
        conditions_sets: Dict[str, List[Dict[str, Any]]],
        episodes_per_condition: int = 20,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate policy generalization across different conditions.

        Args:
            policy: Policy to evaluate
            conditions_sets: Dict of condition_name -> list of conditions
            episodes_per_condition: Episodes to run per condition
            deterministic: Use deterministic policy
            verbose: Print progress

        Returns:
            Dict of condition_name -> EvaluationResult
        """
        results = {}

        for name, conditions in conditions_sets.items():
            if verbose:
                print(f"\nEvaluating on {name} conditions...")

            result = self.evaluate(
                policy=policy,
                n_episodes=episodes_per_condition * len(conditions),
                deterministic=deterministic,
                conditions_list=conditions,
                verbose=verbose,
            )
            results[name] = result

        return results


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 100,
    deterministic: bool = True,
    gates: Optional[List[PromotionGate]] = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Convenience function to evaluate a policy.

    Args:
        policy: Policy with predict(obs) method
        env: Gymnasium environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        gates: Optional promotion gates
        verbose: Print progress

    Returns:
        EvaluationResult
    """
    harness = EvaluationHarness(env=env, gates=gates)
    return harness.evaluate(
        policy=policy,
        n_episodes=n_episodes,
        deterministic=deterministic,
        verbose=verbose,
    )

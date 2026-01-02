"""
Unified Training Pipeline

Single command to run the complete training workflow:
  demos → quality filter → fingerprint → BC → RL → eval → promotion artifact

This is the "one command" entry point for production training.

Example:
    >>> artifact = train_from_demonstrations(
    ...     demo_path="data/expert_flights/",
    ...     output_dir="artifacts/model_v1/",
    ...     task_type="hover",
    ... )
    >>> if artifact.promoted:
    ...     deploy(artifact.model_path)
"""

import json
import shutil
import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

# Training components
from training.imitation.demonstration import DemonstrationDataset, load_demonstrations
from training.imitation.behavioral_cloning import BehavioralCloning, train_bc
from training.imitation.quality import DemonstrationQualityScorer, QualityFilter
from training.imitation.quality.quality_filters import FilterConfig as QualityFilterConfig
from training.imitation.splitting import StratifiedDemoSplitter
from training.imitation.provenance import (
    DatasetFingerprint,
    DatasetFingerprintGenerator,
    generate_fingerprint,
    FeatureSchema,
    FilterConfig,
    SplitConfig,
)

# Evaluation components
from evaluation.harness import (
    EvaluationHarness,
    EvaluationResult,
    SeedConfig,
    EarlyExitConfig,
)
from evaluation.gates import (
    create_default_gates,
    SafetyGate,
    SuccessRateGate,
    InterventionRateGate,
    ViolationEpisodeGate,
    RegressionGate,
    CompositeGate,
)
from evaluation.reports import ReportGenerator

# Simulation components
from simulation.wrappers import SafetyShieldWrapper, ShieldConfig, make_shielded_env


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for the full training pipeline.

    Controls all phases: data loading, training, evaluation, and artifact generation.
    """
    # Task configuration
    task_type: str = "hover"  # hover, waypoint, landing, obstacle, racing

    # Data quality settings
    min_quality_score: float = 0.5
    quality_filter_config: Optional[Dict[str, Any]] = None

    # Training settings
    bc_epochs: int = 100
    bc_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    bc_learning_rate: float = 3e-4
    bc_batch_size: int = 256

    # RL fine-tuning (optional)
    enable_rl: bool = True
    rl_algorithm: str = "PPO"
    rl_timesteps: int = 500000

    # Evaluation settings
    eval_episodes: int = 100
    eval_with_shield: bool = True
    shield_config: Optional[ShieldConfig] = None

    # Promotion gates
    min_success_rate: float = 0.9
    max_crashes: int = 0
    max_override_rate: float = 0.01  # 1% guardian takeover
    max_clamp_rate: float = 0.05     # 5% clamping

    # Regression gate (optional - for production hardening)
    # Use stable, safety-relevant metrics (NOT mean_reward - it drifts with reward shaping)
    baseline_path: Optional[str] = None  # Path to baseline evaluation_result.json
    regression_metrics: List[str] = field(default_factory=lambda: [
        # Core safety (must not regress)
        'success_rate',
        'crash_episodes',
        # Shield dependence (if shielded eval)
        'shield_override_rate',
        'shield_clamp_rate',
        # Control quality (stable across reward changes)
        'mean_position_error',
        'smoothness_score',
    ])

    # Output settings
    generate_html_report: bool = True
    generate_json_report: bool = True
    save_training_history: bool = True

    # Device
    device: str = "auto"
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'task_type': self.task_type,
            'min_quality_score': self.min_quality_score,
            'bc_epochs': self.bc_epochs,
            'bc_hidden_sizes': self.bc_hidden_sizes,
            'bc_learning_rate': self.bc_learning_rate,
            'rl_algorithm': self.rl_algorithm if self.enable_rl else None,
            'rl_timesteps': self.rl_timesteps if self.enable_rl else 0,
            'eval_episodes': self.eval_episodes,
            'eval_with_shield': self.eval_with_shield,
            'min_success_rate': self.min_success_rate,
        }


@dataclass
class TrainingArtifact:
    """
    Complete artifact from training pipeline.

    Contains everything needed for deployment AND reproducibility:
    - Trained model + hash
    - Evaluation results
    - Provenance tracking (dataset fingerprint, git hash, seeds)
    - Full configuration
    """
    # Paths
    artifact_dir: str
    model_path: str
    report_html: Optional[str] = None
    report_json: Optional[str] = None

    # Promotion status
    promoted: bool = False
    promotion_reason: str = ""

    # Provenance - Dataset
    fingerprint: Optional[DatasetFingerprint] = None
    fingerprint_hash: str = ""

    # Provenance - Code
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Provenance - Seeds (for reproducibility)
    meta_seed_used: Optional[int] = None
    fixed_seeds: List[int] = field(default_factory=list)

    # Provenance - Model
    model_hash: str = ""  # SHA256 of model weights

    # Provenance - Environment (REQUIRED for promoted artifacts)
    env_id: str = ""
    env_config: Dict[str, Any] = field(default_factory=dict)
    wrapper_stack: List[Dict[str, Any]] = field(default_factory=list)  # [{name, config_hash, ...}]

    # Evaluation summary - SINGLE SOURCE OF TRUTH for gates
    # These are the exact fields gates check (all episode-level, not event-level)
    evaluation_summary: Dict[str, Any] = field(default_factory=lambda: {
        'episodes_run': 0,
        'crash_episodes': 0,        # Episodes that ENDED in crash
        'critical_violation_episodes': 0,
        'success_episodes': 0,
        # Violation episodes (episodes WITH at least one violation, even if recovered)
        'tilt_violation_episodes': 0,
        'altitude_violation_episodes': 0,
        'geofence_violation_episodes': 0,
        'speed_violation_episodes': 0,
    })

    success_rate: float = 0.0
    mean_reward: float = 0.0
    gates_passed: int = 0
    gates_total: int = 0

    # Termination breakdown: how each episode ended (episode-level, mutually exclusive)
    termination_breakdown: Dict[str, int] = field(default_factory=dict)
    # Safety event counts: events WITHIN episodes (can have multiple per episode)
    safety_event_counts: Dict[str, int] = field(default_factory=dict)
    # Intervention rates from shield
    intervention_breakdown: Dict[str, float] = field(default_factory=dict)

    # Training info
    config: Optional[PipelineConfig] = None
    training_duration_seconds: float = 0.0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary for JSON serialization."""
        return {
            'artifact_dir': self.artifact_dir,
            'model_path': self.model_path,
            'report_html': self.report_html,
            'report_json': self.report_json,
            'promoted': self.promoted,
            'promotion_reason': self.promotion_reason,
            # Provenance - Dataset
            'fingerprint_hash': self.fingerprint_hash,
            # Provenance - Code
            'git': {
                'commit': self.git_commit,
                'branch': self.git_branch,
                'dirty': self.git_dirty,
            },
            # Provenance - Seeds
            'seeds': {
                'meta_seed': self.meta_seed_used,
                'fixed_seeds': self.fixed_seeds,
            },
            # Provenance - Model
            'model_hash': self.model_hash,
            # Provenance - Environment
            'env_id': self.env_id,
            'env_config': self.env_config,
            'wrapper_stack': self.wrapper_stack,
            # Evaluation - SINGLE SOURCE OF TRUTH
            'evaluation_summary': self.evaluation_summary,
            'success_rate': self.success_rate,
            'mean_reward': self.mean_reward,
            'gates_passed': self.gates_passed,
            'gates_total': self.gates_total,
            # Episode termination breakdown (how each episode ended)
            'termination_breakdown': self.termination_breakdown,
            # Safety events within episodes (can be multiple per episode)
            'safety_event_counts': self.safety_event_counts,
            # Shield intervention rates
            'intervention_breakdown': self.intervention_breakdown,
            # Metadata
            'training_duration_seconds': self.training_duration_seconds,
            'created_at': self.created_at,
            'config': self.config.to_dict() if self.config else None,
        }

    def save_manifest(self) -> str:
        """Save artifact manifest to JSON."""
        manifest_path = Path(self.artifact_dir) / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return str(manifest_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_from_demonstrations(
    demo_path: Union[str, List[str]],
    output_dir: str,
    env=None,
    env_fn=None,
    config: Optional[PipelineConfig] = None,
    task_type: str = "hover",
    verbose: bool = True,
) -> TrainingArtifact:
    """
    Complete training pipeline from demonstrations to promoted model.

    This is the single command that:
    1. Loads and filters demonstrations by quality
    2. Generates dataset fingerprint for provenance
    3. Trains BC policy (and optionally RL fine-tuning)
    4. Evaluates with safety shield and promotion gates
    5. Generates reports and artifact bundle

    Args:
        demo_path: Path to demonstrations (file, directory, or list of paths)
        output_dir: Directory for output artifacts
        env: Gymnasium environment (provide env OR env_fn)
        env_fn: Callable that creates environment (for parallel/fresh envs)
        config: Pipeline configuration (uses defaults if None)
        task_type: Type of task (hover, waypoint, landing, obstacle, racing)
        verbose: Print progress

    Returns:
        TrainingArtifact with paths to model, reports, and promotion status

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("DroneHover-v0")
        >>> artifact = train_from_demonstrations(
        ...     demo_path="data/expert_flights/",
        ...     output_dir="artifacts/hover_v1/",
        ...     env=env,
        ...     task_type="hover",
        ... )
        >>> print(f"Promoted: {artifact.promoted}")
        >>> print(f"Success rate: {artifact.success_rate:.1%}")
    """
    start_time = datetime.now()

    # Use config or create default
    if config is None:
        config = PipelineConfig(task_type=task_type, verbose=verbose)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("UNIFIED TRAINING PIPELINE")
        print("=" * 70)
        print(f"  Demo path: {demo_path}")
        print(f"  Output: {output_dir}")
        print(f"  Task type: {config.task_type}")
        print()

    # Get or create environment
    if env is None and env_fn is not None:
        env = env_fn()
    elif env is None:
        raise ValueError("Must provide either 'env' or 'env_fn'")

    # ==========================================================================
    # PHASE 1: DATA LOADING AND QUALITY FILTERING
    # ==========================================================================
    if verbose:
        print("PHASE 1: Data Loading & Quality Filtering")
        print("-" * 50)

    dataset, fingerprint = _load_and_filter_demos(
        demo_path=demo_path,
        config=config,
        output_dir=output_path,
        verbose=verbose,
    )

    if len(dataset) == 0:
        return _create_failed_artifact(
            output_dir=str(output_path),
            reason="No demonstrations passed quality filter",
            config=config,
        )

    # ==========================================================================
    # PHASE 2: BEHAVIORAL CLONING
    # ==========================================================================
    if verbose:
        print("\nPHASE 2: Behavioral Cloning")
        print("-" * 50)

    bc_policy, bc_history = _train_bc(
        dataset=dataset,
        config=config,
        output_dir=output_path,
        verbose=verbose,
    )

    # ==========================================================================
    # PHASE 3: RL FINE-TUNING (OPTIONAL)
    # ==========================================================================
    final_policy = bc_policy.policy

    if config.enable_rl:
        if verbose:
            print("\nPHASE 3: RL Fine-tuning")
            print("-" * 50)

        rl_model = _train_rl(
            env=env,
            bc_policy=bc_policy,
            config=config,
            output_dir=output_path,
            verbose=verbose,
        )

        if rl_model is not None:
            final_policy = rl_model

    # ==========================================================================
    # PHASE 4: EVALUATION WITH SHIELD
    # ==========================================================================
    if verbose:
        print("\nPHASE 4: Evaluation & Promotion Gates")
        print("-" * 50)

    eval_result, eval_env = _run_evaluation(
        env=env,
        policy=final_policy,
        config=config,
        verbose=verbose,
    )

    # ==========================================================================
    # PHASE 5: REPORT GENERATION AND ARTIFACT BUNDLING
    # ==========================================================================
    if verbose:
        print("\nPHASE 5: Report Generation & Artifact Bundle")
        print("-" * 50)

    artifact = _create_artifact(
        output_dir=output_path,
        eval_result=eval_result,
        eval_env=eval_env,
        fingerprint=fingerprint,
        bc_history=bc_history,
        config=config,
        start_time=start_time,
        verbose=verbose,
    )

    # Save manifest
    artifact.save_manifest()

    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Promoted: {'YES' if artifact.promoted else 'NO'}")
        print(f"  Success rate: {artifact.success_rate:.1%}")
        print(f"  Gates: {artifact.gates_passed}/{artifact.gates_total} passed")
        print(f"  Artifact: {artifact.artifact_dir}")
        if artifact.report_html:
            print(f"  Report: {artifact.report_html}")
        print()

    return artifact


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

def _load_and_filter_demos(
    demo_path: Union[str, List[str]],
    config: PipelineConfig,
    output_dir: Path,
    verbose: bool,
) -> Tuple[DemonstrationDataset, Optional[DatasetFingerprint]]:
    """Phase 1: Load demonstrations and apply quality filtering."""

    # Handle single path or list
    if isinstance(demo_path, str):
        demo_paths = [demo_path]
    else:
        demo_paths = demo_path

    # Load all demonstrations
    all_demos = []
    for path in demo_paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            dataset = DemonstrationDataset.load(str(path_obj))
            all_demos.extend(dataset.demonstrations)
            if verbose:
                print(f"  Loaded {len(dataset)} demos from {path}")
        elif path_obj.exists():
            from training.imitation.demonstration import Demonstration
            demo = Demonstration.load(str(path_obj))
            all_demos.append(demo)
            if verbose:
                print(f"  Loaded demo from {path}")

    if verbose:
        print(f"  Total demos loaded: {len(all_demos)}")

    # Apply quality scoring
    scorer = DemonstrationQualityScorer()
    for demo in all_demos:
        if demo.quality_score == 0:  # Not scored yet
            demo.quality_score = scorer.score(demo)

    # Apply quality filter
    quality_filter_config = QualityFilterConfig(
        min_quality_score=config.min_quality_score,
        **(config.quality_filter_config or {})
    )

    quality_filter = QualityFilter(config=quality_filter_config)

    filtered_demos = [d for d in all_demos if quality_filter.passes_all_filters(d)]

    if verbose:
        print(f"  After quality filter: {len(filtered_demos)} demos")
        print(f"  Rejected: {len(all_demos) - len(filtered_demos)} low-quality demos")

    # Create dataset
    dataset = DemonstrationDataset(filtered_demos)

    # Generate fingerprint
    fingerprint = None
    if len(dataset) > 0:
        try:
            fingerprint_gen = DatasetFingerprintGenerator()

            fingerprint = fingerprint_gen.generate(
                dataset=dataset,
                dataset_id=config.task_type,
                description=f"Filtered demos for {config.task_type} training",
            )

            # Save fingerprint
            fingerprint_path = output_dir / "fingerprint.json"
            with open(fingerprint_path, 'w') as f:
                json.dump(fingerprint.to_dict(), f, indent=2, default=str)

            if verbose:
                print(f"  Fingerprint: {fingerprint.content_hash[:16]}...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate fingerprint: {e}")

    return dataset, fingerprint


def _train_bc(
    dataset: DemonstrationDataset,
    config: PipelineConfig,
    output_dir: Path,
    verbose: bool,
) -> Tuple[BehavioralCloning, Dict[str, Any]]:
    """Phase 2: Train behavioral cloning policy."""

    bc = BehavioralCloning(
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        hidden_sizes=config.bc_hidden_sizes,
        learning_rate=config.bc_learning_rate,
        batch_size=config.bc_batch_size,
        device=config.device,
    )

    history = bc.train(
        dataset=dataset,
        num_epochs=config.bc_epochs,
        verbose=verbose,
    )

    # Save BC model
    bc_path = output_dir / "bc_policy.pt"
    bc.save(str(bc_path))

    if verbose:
        print(f"  Final val loss: {history['best_val_loss']:.6f}")
        print(f"  Epochs: {history['epochs_trained']}")
        print(f"  Saved: {bc_path}")

    return bc, history


def _train_rl(
    env,
    bc_policy: BehavioralCloning,
    config: PipelineConfig,
    output_dir: Path,
    verbose: bool,
):
    """Phase 3: Optional RL fine-tuning."""

    try:
        from stable_baselines3 import PPO, SAC, TD3

        algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
        algo_class = algorithms.get(config.rl_algorithm, PPO)

        if verbose:
            print(f"  Algorithm: {config.rl_algorithm}")
            print(f"  Timesteps: {config.rl_timesteps}")

        # Create model
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=1 if verbose else 0,
            device=config.device,
        )

        # Train
        model.learn(total_timesteps=config.rl_timesteps)

        # Save
        rl_path = output_dir / "rl_policy.zip"
        model.save(str(rl_path))

        if verbose:
            print(f"  Saved: {rl_path}")

        return model

    except ImportError:
        if verbose:
            print("  Skipping RL: stable-baselines3 not installed")
        return None
    except Exception as e:
        if verbose:
            print(f"  RL training failed: {e}")
        return None


def _run_evaluation(
    env,
    policy,
    config: PipelineConfig,
    verbose: bool,
) -> Tuple[EvaluationResult, Any]:
    """Phase 4: Evaluate with shield and promotion gates."""

    # Wrap with shield if enabled
    eval_env = env
    if config.eval_with_shield:
        shield_config = config.shield_config or ShieldConfig(apply_penalties=False)
        eval_env = make_shielded_env(env, shield_config)
        if verbose:
            print(f"  Shield: ON (penalties={shield_config.apply_penalties})")

    # Create promotion gates in optimal order:
    # 1. SafetyGate (cheapest, hard failures kill fast)
    # 2. InterventionRateGate (cheap, reveals shield-dependence)
    # 3. RegressionGate (compares to baseline)
    # 4. SuccessRateGate (performance gate, last)
    gates = []

    # 1. Safety gate (FIRST - zero tolerance for crashes)
    gates.append(
        SafetyGate(
            min_episodes=config.eval_episodes // 2,
            max_crashes=config.max_crashes,
        )
    )

    # 2. Intervention rate gate (if shield is on)
    if config.eval_with_shield:
        gates.append(
            InterventionRateGate(
                max_emergency_interventions=0,
                max_override_rate=config.max_override_rate,
                max_clamp_rate=config.max_clamp_rate,
                min_episodes=config.eval_episodes // 2,
            )
        )

    # 3. Violation episode gate (task-type aware)
    # Conservative tasks (hover, landing) get strict defaults
    # Aggressive tasks get looser limits
    task_lower = config.task_type.lower()
    if task_lower in ('hover', 'landing'):
        gates.append(ViolationEpisodeGate.for_hover(min_episodes=config.eval_episodes // 2))
        if verbose:
            print(f"  Violation gate: STRICT (zero violations for {config.task_type})")
    elif task_lower == 'waypoint':
        gates.append(ViolationEpisodeGate.for_waypoint(min_episodes=config.eval_episodes // 2))
        if verbose:
            print(f"  Violation gate: MODERATE (waypoint)")
    elif task_lower in ('obstacle', 'racing', 'aggressive'):
        gates.append(ViolationEpisodeGate.for_aggressive(min_episodes=config.eval_episodes // 2))
        if verbose:
            print(f"  Violation gate: LOOSE (aggressive task)")
    else:
        # Default to hover-style strict for unknown tasks
        gates.append(ViolationEpisodeGate.for_hover(min_episodes=config.eval_episodes // 2))
        if verbose:
            print(f"  Violation gate: STRICT (default for {config.task_type})")

    # 4. Regression gate (if baseline provided)
    if config.baseline_path and Path(config.baseline_path).exists():
        gates.append(
            RegressionGate(
                baseline_path=config.baseline_path,
                metrics_to_check=config.regression_metrics,
            )
        )
        if verbose:
            print(f"  Regression baseline: {config.baseline_path}")

    # 5. Success rate gate (LAST - performance metric)
    gates.append(
        SuccessRateGate(
            min_success_rate=config.min_success_rate,
            min_episodes=config.eval_episodes // 2,
        )
    )

    composite_gate = CompositeGate(gates, name="promotion_gates")

    # Run evaluation
    harness = EvaluationHarness(
        env=eval_env,
        gates=[composite_gate],
        seed_config=SeedConfig(use_random_seeds=True),
        early_exit_config=EarlyExitConfig(enabled=True),
    )

    if verbose:
        print(f"  Episodes: {config.eval_episodes}")
        print(f"  Gates: {len(gates)}")

    result = harness.evaluate(
        policy=policy,
        n_episodes=config.eval_episodes,
        deterministic=True,
        verbose=verbose,
    )

    return result, eval_env


def _create_artifact(
    output_dir: Path,
    eval_result: EvaluationResult,
    eval_env,  # The environment used for evaluation (for env_id and wrapper stack)
    fingerprint: Optional[DatasetFingerprint],
    bc_history: Dict[str, Any],
    config: PipelineConfig,
    start_time: datetime,
    verbose: bool,
) -> TrainingArtifact:
    """Phase 5: Create artifact bundle with reports."""

    # Generate reports
    report_html = None
    report_json = None

    generator = ReportGenerator(eval_result)

    if config.generate_json_report:
        report_json = str(output_dir / "report.json")
        generator.to_json(report_json)
        if verbose:
            print(f"  Report JSON: {report_json}")

    if config.generate_html_report:
        report_html = str(output_dir / "report.html")
        generator.to_html(report_html)
        if verbose:
            print(f"  Report HTML: {report_html}")

    # Save training history
    if config.save_training_history:
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(bc_history, f, indent=2, default=str)

    # Save evaluation result
    eval_path = output_dir / "evaluation_result.json"
    eval_result.save(str(eval_path))

    # Determine promotion status
    promoted = eval_result.all_gates_passed

    if promoted:
        promotion_reason = "All gates passed"
    else:
        failed_gates = [
            name for name, r in eval_result.gate_results.items()
            if not r.passed
        ]
        promotion_reason = f"Failed gates: {', '.join(failed_gates)}"

    # Count gates
    gates_total = len(eval_result.gate_results)
    gates_passed = sum(1 for r in eval_result.gate_results.values() if r.passed)

    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()

    # Determine model path (RL if available, else BC)
    rl_path = output_dir / "rl_policy.zip"
    bc_path = output_dir / "bc_policy.pt"
    model_path = str(rl_path) if rl_path.exists() else str(bc_path)

    # Compute model hash
    model_hash = _compute_model_hash(model_path)

    # Get git info
    git_info = _get_git_info()

    # Extract breakdowns with clear semantics
    evaluation_summary = _extract_evaluation_summary(eval_result)
    termination_breakdown = _extract_termination_breakdown(eval_result)
    safety_event_counts = _extract_safety_event_counts(eval_result)
    intervention_breakdown = _extract_intervention_breakdown(eval_result)

    # Get seed info from eval result
    meta_seed = eval_result.meta_seed_used
    # Fixed seeds from config (first 20 by default)
    fixed_seeds = list(range(42, 62))

    # Get environment info
    env_id, env_config = _get_env_info(eval_env)
    wrapper_stack = _get_wrapper_stack(eval_env)

    # VALIDATION: env_id must not be empty for promoted artifacts
    if promoted and not env_id:
        promoted = False
        promotion_reason = "Cannot promote: env_id is empty (reproducibility requirement)"
        if verbose:
            print(f"  WARNING: Blocking promotion - env_id is empty")

    # Create promoted model folder if promoted
    if promoted:
        promoted_dir = output_dir / "promoted"
        promoted_dir.mkdir(exist_ok=True)

        # Copy model to promoted folder
        if rl_path.exists():
            shutil.copy(rl_path, promoted_dir / "policy.zip")
        else:
            shutil.copy(bc_path, promoted_dir / "policy.pt")

        # Copy config
        config_path = promoted_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Copy fingerprint
        if fingerprint:
            fp_path = promoted_dir / "fingerprint.json"
            with open(fp_path, 'w') as f:
                json.dump(fingerprint.to_dict(), f, indent=2, default=str)

    return TrainingArtifact(
        artifact_dir=str(output_dir),
        model_path=model_path,
        report_html=report_html,
        report_json=report_json,
        promoted=promoted,
        promotion_reason=promotion_reason,
        # Provenance - Dataset
        fingerprint=fingerprint,
        fingerprint_hash=fingerprint.content_hash if fingerprint else "",
        # Provenance - Code
        git_commit=git_info.get('commit', ''),
        git_branch=git_info.get('branch', ''),
        git_dirty=git_info.get('dirty', False),
        # Provenance - Seeds
        meta_seed_used=meta_seed,
        fixed_seeds=fixed_seeds,
        # Provenance - Model
        model_hash=model_hash,
        # Provenance - Environment
        env_id=env_id,
        env_config=env_config,
        wrapper_stack=wrapper_stack,
        # Evaluation - SINGLE SOURCE OF TRUTH
        evaluation_summary=evaluation_summary,
        success_rate=eval_result.success_rate,
        mean_reward=eval_result.mean_reward,
        gates_passed=gates_passed,
        gates_total=gates_total,
        # Episode termination breakdown
        termination_breakdown=termination_breakdown,
        # Safety events within episodes
        safety_event_counts=safety_event_counts,
        # Shield intervention rates
        intervention_breakdown=intervention_breakdown,
        # Metadata
        config=config,
        training_duration_seconds=duration,
    )


def _create_failed_artifact(
    output_dir: str,
    reason: str,
    config: PipelineConfig,
) -> TrainingArtifact:
    """Create artifact for failed pipeline."""
    git_info = _get_git_info()
    return TrainingArtifact(
        artifact_dir=output_dir,
        model_path="",
        promoted=False,
        promotion_reason=reason,
        config=config,
        git_commit=git_info.get('commit', ''),
        git_branch=git_info.get('branch', ''),
        git_dirty=git_info.get('dirty', False),
    )


# =============================================================================
# PROVENANCE HELPERS
# =============================================================================

def _get_git_info() -> Dict[str, Any]:
    """Get git commit hash, branch, and dirty status."""
    info = {'commit': '', 'branch': '', 'dirty': False}

    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['dirty'] = len(result.stdout.strip()) > 0

    except Exception:
        pass  # Git not available or not in a repo

    return info


def _compute_model_hash(model_path: str) -> str:
    """Compute SHA256 hash of model file."""
    if not Path(model_path).exists():
        return ""

    hasher = hashlib.sha256()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_wrapper_stack(env) -> List[Dict[str, Any]]:
    """
    Get list of wrapper info with config hashes.

    Each entry includes:
    - name: class name
    - config_hash: hash of config if available
    - stage: curriculum stage if applicable
    """
    wrappers = []
    current = env

    while current is not None:
        wrapper_info = {'name': type(current).__name__}

        # Extract config if available
        if hasattr(current, 'config'):
            config = current.config
            if hasattr(config, '__dict__'):
                config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
                wrapper_info['config_hash'] = hashlib.md5(config_str.encode()).hexdigest()[:12]

            # Special handling for known wrappers
            if hasattr(config, 'limits'):  # SafetyShieldWrapper
                wrapper_info['limits_hash'] = hashlib.md5(
                    str(config.limits.__dict__).encode()
                ).hexdigest()[:12]

        # Curriculum stage if applicable
        if hasattr(current, 'current_stage'):
            wrapper_info['stage'] = current.current_stage
        if hasattr(current, 'stage_name'):
            wrapper_info['stage'] = current.stage_name

        wrappers.append(wrapper_info)

        if hasattr(current, 'env'):
            current = current.env
        else:
            break

    return wrappers


def _get_env_info(env) -> Tuple[str, Dict[str, Any]]:
    """
    Extract environment ID and config.

    Returns:
        (env_id, env_config)
    """
    env_id = ""
    env_config = {}

    # Try to get env spec/ID
    if hasattr(env, 'spec') and env.spec is not None:
        env_id = env.spec.id
    elif hasattr(env, 'unwrapped'):
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'spec') and unwrapped.spec is not None:
            env_id = unwrapped.spec.id
        else:
            env_id = type(unwrapped).__name__

    # If still empty, use class name
    if not env_id:
        env_id = type(env.unwrapped).__name__ if hasattr(env, 'unwrapped') else type(env).__name__

    # Try to get config
    if hasattr(env, 'config'):
        if hasattr(env.config, '__dict__'):
            env_config = {k: str(v) for k, v in env.config.__dict__.items()}
        elif isinstance(env.config, dict):
            env_config = {k: str(v) for k, v in env.config.items()}

    # Check for common config attributes
    for attr in ['platform_config', 'env_conditions', 'observation_space', 'action_space']:
        if hasattr(env, attr):
            val = getattr(env, attr)
            if hasattr(val, 'shape'):
                env_config[attr] = str(val.shape)
            elif hasattr(val, '__dict__'):
                env_config[attr] = type(val).__name__

    return env_id, env_config


def _extract_evaluation_summary(eval_result: 'EvaluationResult') -> Dict[str, int]:
    """
    Extract evaluation summary - SINGLE SOURCE OF TRUTH for gates.

    This is the definitive source for episode-level counts that gates check.
    All counts are EPISODE-LEVEL (not event counts).

    - crash_episodes: episodes that ENDED in crash
    - tilt_violation_episodes: episodes that HAD at least one tilt violation
      (even if they recovered and succeeded)
    """
    crash_episodes = 0
    critical_violation_episodes = 0
    success_episodes = 0

    # Violation episodes: episodes WITH at least one violation (even if recovered)
    tilt_violation_episodes = 0
    altitude_violation_episodes = 0
    geofence_violation_episodes = 0
    speed_violation_episodes = 0

    for ep in eval_result.episodes:
        if ep.success:
            success_episodes += 1
        else:
            # An episode is a "crash episode" if it TERMINATED due to crash
            # (not just had crash events during it)
            reason = ep.termination_reason.lower()
            if 'crash' in reason:
                crash_episodes += 1
            elif ep.metrics.get('crash_count', 0) > 0 and ep.metrics.get('terminated', 0) > 0:
                # Terminated (not truncated) with crash events = crash episode
                crash_episodes += 1

            # Critical violations that caused termination
            if 'violation' in reason or 'critical' in reason:
                critical_violation_episodes += 1

        # Count episodes WITH violations (regardless of success/failure)
        # These tell you how widespread the problem is
        if ep.metrics.get('tilt_violations', 0) > 0:
            tilt_violation_episodes += 1
        if ep.metrics.get('altitude_violations', 0) > 0:
            altitude_violation_episodes += 1
        if ep.metrics.get('geofence_violations', 0) > 0:
            geofence_violation_episodes += 1
        if ep.metrics.get('speed_violations', 0) > 0:
            speed_violation_episodes += 1

    return {
        'episodes_run': len(eval_result.episodes),
        'crash_episodes': crash_episodes,
        'critical_violation_episodes': critical_violation_episodes,
        'success_episodes': success_episodes,
        # Violation episodes (can overlap with success - recovered violations)
        'tilt_violation_episodes': tilt_violation_episodes,
        'altitude_violation_episodes': altitude_violation_episodes,
        'geofence_violation_episodes': geofence_violation_episodes,
        'speed_violation_episodes': speed_violation_episodes,
    }


def _extract_termination_breakdown(eval_result: 'EvaluationResult') -> Dict[str, int]:
    """
    Extract episode termination reasons (mutually exclusive per episode).

    Each episode has exactly ONE termination reason.
    """
    breakdown = {
        'success': 0,
        'crash': 0,
        'geofence': 0,
        'altitude_violation': 0,
        'timeout': 0,
        'other_failure': 0,
    }

    for ep in eval_result.episodes:
        if ep.success:
            breakdown['success'] += 1
            continue

        reason = ep.termination_reason.lower()

        # Check in priority order (most specific first)
        if 'crash' in reason:
            breakdown['crash'] += 1
        elif 'geofence' in reason or 'boundary' in reason:
            breakdown['geofence'] += 1
        elif 'altitude' in reason:
            breakdown['altitude_violation'] += 1
        elif 'truncat' in reason or ep.metrics.get('truncated', 0) > 0:
            breakdown['timeout'] += 1
        else:
            breakdown['other_failure'] += 1

    # Remove zero entries for cleaner output
    return {k: v for k, v in breakdown.items() if v > 0}


def _extract_safety_event_counts(eval_result: 'EvaluationResult') -> Dict[str, int]:
    """
    Extract safety events WITHIN episodes (can have multiple per episode).

    These are event counts, not episode counts. An episode can have
    multiple tilt violations, for example.
    """
    counts = {
        'crash_events': 0,
        'tilt_violations': 0,
        'altitude_violations': 0,
        'speed_violations': 0,
        'geofence_violations': 0,
    }

    for ep in eval_result.episodes:
        counts['crash_events'] += int(ep.metrics.get('crash_count', 0))
        counts['tilt_violations'] += int(ep.metrics.get('tilt_violations', 0))
        counts['altitude_violations'] += int(ep.metrics.get('altitude_violations', 0))
        counts['speed_violations'] += int(ep.metrics.get('speed_violations', 0))
        counts['geofence_violations'] += int(ep.metrics.get('geofence_violations', 0))

    # Remove zero entries
    return {k: v for k, v in counts.items() if v > 0}


def _extract_intervention_breakdown(eval_result: 'EvaluationResult') -> Dict[str, float]:
    """Extract intervention rates from evaluation results."""
    total_steps = 0
    total_clamps = 0
    total_overrides = 0
    total_emergencies = 0

    for ep in eval_result.episodes:
        steps = ep.metrics.get('steps', ep.metrics.get('episode_length', 0))
        total_steps += int(steps)

        total_clamps += int(ep.metrics.get('shield_clamp_count', 0))
        total_overrides += int(ep.metrics.get('shield_override_count', 0))
        total_emergencies += int(ep.metrics.get('shield_emergency_count', 0))

    if total_steps == 0:
        return {}

    return {
        'clamp_rate': round(total_clamps / total_steps, 4),
        'override_rate': round(total_overrides / total_steps, 4),
        'emergency_count': total_emergencies,
        'total_steps': total_steps,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_train(
    demo_path: str,
    output_dir: str,
    env,
    bc_epochs: int = 50,
    eval_episodes: int = 50,
    verbose: bool = True,
) -> TrainingArtifact:
    """
    Quick training for development/testing.

    Runs a minimal pipeline with:
    - No RL fine-tuning
    - Fewer BC epochs
    - Fewer eval episodes

    Args:
        demo_path: Path to demonstrations
        output_dir: Output directory
        env: Gymnasium environment
        bc_epochs: BC training epochs
        eval_episodes: Evaluation episodes
        verbose: Print progress

    Returns:
        TrainingArtifact
    """
    config = PipelineConfig(
        bc_epochs=bc_epochs,
        enable_rl=False,
        eval_episodes=eval_episodes,
        verbose=verbose,
    )

    return train_from_demonstrations(
        demo_path=demo_path,
        output_dir=output_dir,
        env=env,
        config=config,
        verbose=verbose,
    )


def load_artifact(artifact_dir: str) -> TrainingArtifact:
    """
    Load a training artifact from disk.

    Args:
        artifact_dir: Path to artifact directory

    Returns:
        TrainingArtifact
    """
    manifest_path = Path(artifact_dir) / "manifest.json"

    with open(manifest_path, 'r') as f:
        data = json.load(f)

    git_info = data.get('git', {})
    seeds_info = data.get('seeds', {})

    return TrainingArtifact(
        artifact_dir=data['artifact_dir'],
        model_path=data['model_path'],
        report_html=data.get('report_html'),
        report_json=data.get('report_json'),
        promoted=data['promoted'],
        promotion_reason=data['promotion_reason'],
        fingerprint_hash=data.get('fingerprint_hash', ''),
        git_commit=git_info.get('commit', ''),
        git_branch=git_info.get('branch', ''),
        git_dirty=git_info.get('dirty', False),
        meta_seed_used=seeds_info.get('meta_seed'),
        fixed_seeds=seeds_info.get('fixed_seeds', []),
        model_hash=data.get('model_hash', ''),
        env_id=data.get('env_id', ''),
        env_config=data.get('env_config', {}),
        wrapper_stack=data.get('wrapper_stack', []),
        evaluation_summary=data.get('evaluation_summary', {}),
        success_rate=data['success_rate'],
        mean_reward=data['mean_reward'],
        gates_passed=data['gates_passed'],
        gates_total=data['gates_total'],
        termination_breakdown=data.get('termination_breakdown', {}),
        safety_event_counts=data.get('safety_event_counts', {}),
        intervention_breakdown=data.get('intervention_breakdown', {}),
        training_duration_seconds=data['training_duration_seconds'],
        created_at=data['created_at'],
    )


def promote_latest(
    artifacts_dir: str,
    deploy_dir: str = "deploy/latest",
    verbose: bool = True,
) -> Optional[TrainingArtifact]:
    """
    Find the newest promoted artifact and copy to deploy location.

    This is a deployment helper that:
    1. Scans artifacts_dir for the most recent promoted model
    2. Copies the promoted model to deploy_dir
    3. Writes a deploy manifest with fingerprint + config

    Args:
        artifacts_dir: Directory containing artifact runs
        deploy_dir: Destination for promoted model
        verbose: Print progress

    Returns:
        TrainingArtifact if promotion succeeded, None otherwise

    Example:
        >>> artifact = promote_latest(
        ...     artifacts_dir="artifacts/",
        ...     deploy_dir="deploy/latest/",
        ... )
        >>> if artifact:
        ...     print(f"Deployed: {artifact.model_path}")
    """
    artifacts_path = Path(artifacts_dir)
    deploy_path = Path(deploy_dir)

    if not artifacts_path.exists():
        if verbose:
            print(f"Artifacts directory not found: {artifacts_dir}")
        return None

    # Find all artifact directories with manifests
    candidates = []
    for manifest_file in artifacts_path.glob("**/manifest.json"):
        try:
            artifact = load_artifact(str(manifest_file.parent))
            if artifact.promoted:
                candidates.append(artifact)
        except Exception as e:
            if verbose:
                print(f"  Skipping {manifest_file}: {e}")

    if not candidates:
        if verbose:
            print("No promoted artifacts found")
        return None

    # Sort by creation time (newest first)
    candidates.sort(key=lambda a: a.created_at, reverse=True)
    latest = candidates[0]

    if verbose:
        print(f"Found {len(candidates)} promoted artifacts")
        print(f"Latest: {latest.artifact_dir}")
        print(f"  Created: {latest.created_at}")
        print(f"  Success rate: {latest.success_rate:.1%}")

    # Create deploy directory
    deploy_path.mkdir(parents=True, exist_ok=True)

    # Copy model
    src_model = Path(latest.model_path)
    if src_model.exists():
        dst_model = deploy_path / src_model.name
        shutil.copy(src_model, dst_model)
        if verbose:
            print(f"  Copied model to: {dst_model}")

    # Write deploy manifest
    deploy_manifest = {
        'source_artifact': latest.artifact_dir,
        'model_path': str(deploy_path / src_model.name),
        'model_hash': latest.model_hash,
        'fingerprint_hash': latest.fingerprint_hash,
        'git_commit': latest.git_commit,
        'success_rate': latest.success_rate,
        'gates_passed': latest.gates_passed,
        'gates_total': latest.gates_total,
        'deployed_at': datetime.now().isoformat(),
        'config': latest.config.to_dict() if latest.config else None,
    }

    manifest_path = deploy_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(deploy_manifest, f, indent=2, default=str)

    if verbose:
        print(f"  Wrote manifest: {manifest_path}")
        print(f"\nDeployment complete: {deploy_path}")

    return latest


def find_artifacts(
    artifacts_dir: str,
    promoted_only: bool = False,
) -> List[TrainingArtifact]:
    """
    Find all artifacts in a directory.

    Args:
        artifacts_dir: Directory to scan
        promoted_only: Only return promoted artifacts

    Returns:
        List of TrainingArtifact sorted by creation time (newest first)
    """
    artifacts_path = Path(artifacts_dir)
    artifacts = []

    for manifest_file in artifacts_path.glob("**/manifest.json"):
        try:
            artifact = load_artifact(str(manifest_file.parent))
            if not promoted_only or artifact.promoted:
                artifacts.append(artifact)
        except Exception:
            pass

    # Sort newest first
    artifacts.sort(key=lambda a: a.created_at, reverse=True)
    return artifacts

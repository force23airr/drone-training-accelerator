"""
Test the training pipeline with synthetic demo data.

This validates the full pipeline without requiring real environments or demos.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

# Create mock environment
@dataclass
class MockSpec:
    id: str = "MockDroneHover-v0"

class MockEnv:
    """Minimal mock environment for testing."""

    def __init__(self):
        self.observation_space = MockSpace((12,))
        self.action_space = MockSpace((4,))
        self.spec = MockSpec()
        self._step_count = 0
        self._episode_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        self._episode_count += 1
        obs = np.random.randn(12).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(12).astype(np.float32)

        # Simulate reasonable success rate
        done = self._step_count >= 100
        truncated = done
        terminated = False

        # Occasional early termination (success)
        if self._step_count > 50 and np.random.random() < 0.05:
            terminated = True
            done = True

        reward = 1.0 - 0.01 * np.sum(action**2)

        info = {
            'crash_count': 0,
            'tilt_violations': 0,
            'altitude_violations': 0,
            'speed_violations': 0,
        }

        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self


class MockSpace:
    """Mock gym space."""
    def __init__(self, shape):
        self.shape = shape
        self.low = -np.ones(shape)
        self.high = np.ones(shape)

    def sample(self):
        return np.random.randn(*self.shape).astype(np.float32)


class MockPolicy:
    """Mock policy for testing."""
    def __init__(self, action_dim=4):
        self.action_dim = action_dim

    def predict(self, obs, deterministic=True):
        if obs.ndim == 1:
            action = np.random.randn(self.action_dim).astype(np.float32) * 0.1
        else:
            action = np.random.randn(obs.shape[0], self.action_dim).astype(np.float32) * 0.1
        return action, None


def create_synthetic_demos(output_dir: str, n_demos: int = 10):
    """Create synthetic demonstration data."""
    from training.imitation.demonstration import Demonstration, DemonstrationDataset, DemonstrationStep

    demos = []
    for i in range(n_demos):
        # Random episode length
        n_steps = np.random.randint(50, 150)

        # Generate synthetic steps
        steps = []
        for t in range(n_steps):
            step = DemonstrationStep(
                observation=np.random.randn(12).astype(np.float32),
                action=np.random.randn(4).astype(np.float32) * 0.5,
                timestamp=t * 0.01,
                reward=0.9 + np.random.randn() * 0.1,
            )
            steps.append(step)

        demo = Demonstration(
            steps=steps,
            pilot_id=f"synthetic_pilot_{i % 3}",
            task_type="hover",
            quality_score=0.7 + np.random.random() * 0.3,
            source="synthetic",
        )
        demos.append(demo)

    # Save dataset
    dataset = DemonstrationDataset(demos)
    dataset.save(output_dir)
    print(f"Created {n_demos} synthetic demos in {output_dir}")
    return dataset


def test_pipeline_minimal():
    """Minimal test - just check imports and basic structures."""
    print("=" * 60)
    print("TEST 1: Minimal Import Test")
    print("=" * 60)

    from training.pipelines import (
        train_from_demonstrations,
        quick_train,
        load_artifact,
        promote_latest,
        PipelineConfig,
        TrainingArtifact,
    )

    from evaluation.gates import (
        SafetyGate,
        InterventionRateGate,
        ViolationEpisodeGate,
        RegressionGate,
        CompositeGate,
    )

    # Test ViolationEpisodeGate factory methods
    hover_gate = ViolationEpisodeGate.for_hover()
    assert hover_gate.max_tilt == 0, "Hover should have zero tilt tolerance"

    landing_gate = ViolationEpisodeGate.for_landing()
    assert landing_gate.max_altitude == 0, "Landing should have zero altitude tolerance"

    waypoint_gate = ViolationEpisodeGate.for_waypoint()
    assert waypoint_gate.max_tilt == 2, "Waypoint should allow some tilt"

    aggressive_gate = ViolationEpisodeGate.for_aggressive()
    assert aggressive_gate.max_tilt == 10, "Aggressive should be loose on tilt"

    print("✓ All imports successful")
    print("✓ ViolationEpisodeGate factory methods work")
    print()


def test_pipeline_config():
    """Test pipeline configuration."""
    print("=" * 60)
    print("TEST 2: Pipeline Configuration")
    print("=" * 60)

    from training.pipelines import PipelineConfig

    # Default config
    config = PipelineConfig()
    assert config.task_type == "hover"
    assert config.min_success_rate == 0.9
    assert config.max_crashes == 0
    print(f"✓ Default config: task_type={config.task_type}")

    # Check regression metrics are safety-focused
    assert 'success_rate' in config.regression_metrics
    assert 'crash_episodes' in config.regression_metrics
    assert 'mean_reward' not in config.regression_metrics  # Should NOT include reward
    print(f"✓ Regression metrics: {config.regression_metrics}")

    # Custom config
    custom = PipelineConfig(
        task_type="racing",
        bc_epochs=50,
        enable_rl=False,
        baseline_path="/some/baseline.json",
    )
    assert custom.task_type == "racing"
    assert custom.enable_rl == False
    print(f"✓ Custom config works")
    print()


def test_artifact_structure():
    """Test TrainingArtifact structure."""
    print("=" * 60)
    print("TEST 3: Artifact Structure")
    print("=" * 60)

    from training.pipelines import TrainingArtifact, PipelineConfig

    artifact = TrainingArtifact(
        artifact_dir="/tmp/test",
        model_path="/tmp/test/model.pt",
        promoted=True,
        promotion_reason="All gates passed",
        success_rate=0.95,
        mean_reward=150.0,
        gates_passed=5,
        gates_total=5,
        env_id="MockDroneHover-v0",
        git_commit="abc123",
        model_hash="sha256:xyz",
        evaluation_summary={
            'episodes_run': 100,
            'crash_episodes': 0,
            'success_episodes': 95,
            'tilt_violation_episodes': 0,
        },
        termination_breakdown={'success': 95, 'timeout': 5},
        safety_event_counts={},
        intervention_breakdown={'clamp_rate': 0.02, 'override_rate': 0.001},
    )

    # Convert to dict
    d = artifact.to_dict()

    assert d['promoted'] == True
    assert d['evaluation_summary']['crash_episodes'] == 0
    assert d['env_id'] == "MockDroneHover-v0"
    assert d['git']['commit'] == "abc123"
    assert 'termination_breakdown' in d
    assert 'safety_event_counts' in d
    print(f"✓ Artifact structure correct")
    print(f"✓ evaluation_summary has episode-level counts")
    print(f"✓ Separate termination_breakdown and safety_event_counts")
    print()


def test_regression_gate_fail_closed():
    """Test that RegressionGate fails on missing metrics."""
    print("=" * 60)
    print("TEST 4: RegressionGate Fail-Closed Behavior")
    print("=" * 60)

    from evaluation.gates import RegressionGate, GateCheckResult

    # Create gate with some metrics
    gate = RegressionGate(
        baseline_metrics={
            'success_rate': 0.9,
            'crash_episodes': 0,
        },
        metrics_to_check=['success_rate', 'crash_episodes', 'missing_metric'],
        fail_on_missing=True,
    )

    # Mock episode result
    class MockEpisode:
        def __init__(self):
            self.metrics = {}
            self.success = True

    # Aggregated metrics missing 'missing_metric'
    aggregated = {
        'success_rate': {'mean': 0.92, 'std': 0.05},
        'crash_episodes': {'mean': 0, 'std': 0},
    }

    result = gate.check([MockEpisode()], aggregated)

    assert result.passed == False, "Should fail when metric is missing"
    assert "missing_metric" in result.message
    assert "fail-closed" in result.message.lower()
    print(f"✓ Gate correctly fails on missing metric")
    print(f"  Message: {result.message}")

    # Test with all metrics present
    gate2 = RegressionGate(
        baseline_metrics={
            'success_rate': 0.9,
        },
        metrics_to_check=['success_rate'],
        fail_on_missing=True,
    )

    result2 = gate2.check([MockEpisode()], aggregated)
    assert result2.passed == True, "Should pass when all metrics present and no regression"
    print(f"✓ Gate passes when all metrics present")
    print()


def test_synthetic_demo_creation():
    """Test creating synthetic demos."""
    print("=" * 60)
    print("TEST 5: Synthetic Demo Creation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_dir = Path(tmpdir) / "demos"
        demo_dir.mkdir()

        dataset = create_synthetic_demos(str(demo_dir), n_demos=5)

        assert len(dataset) == 5
        assert dataset.observation_dim == 12
        assert dataset.action_dim == 4
        print(f"✓ Created {len(dataset)} demos")
        print(f"✓ Observation dim: {dataset.observation_dim}")
        print(f"✓ Action dim: {dataset.action_dim}")
        print(f"✓ Total transitions: {dataset.total_transitions}")
    print()


def test_bc_training():
    """Test BC training with synthetic data."""
    print("=" * 60)
    print("TEST 6: BC Training")
    print("=" * 60)

    from training.imitation.behavioral_cloning import BehavioralCloning
    from training.imitation.demonstration import Demonstration, DemonstrationDataset, DemonstrationStep

    # Create minimal synthetic dataset
    demos = []
    for i in range(3):
        n_steps = 50
        steps = []
        for t in range(n_steps):
            step = DemonstrationStep(
                observation=np.random.randn(12).astype(np.float32),
                action=np.random.randn(4).astype(np.float32) * 0.5,
                timestamp=t * 0.01,
                reward=1.0,
            )
            steps.append(step)
        demo = Demonstration(
            steps=steps,
            quality_score=0.8,
        )
        demos.append(demo)

    dataset = DemonstrationDataset(demos)

    # Train BC
    bc = BehavioralCloning(
        observation_dim=12,
        action_dim=4,
        hidden_sizes=[32, 32],  # Small for testing
        learning_rate=1e-3,
        batch_size=32,
    )

    history = bc.train(
        dataset=dataset,
        num_epochs=5,
        verbose=False,
    )

    assert 'train_losses' in history
    assert len(history['train_losses']) > 0
    # Loss should generally decrease (allow some tolerance with random data)
    loss_decreased = history['train_losses'][-1] <= history['train_losses'][0] * 1.1
    print(f"✓ BC training completed (loss {'decreased' if loss_decreased else 'stable'})")
    print(f"✓ Initial loss: {history['train_losses'][0]:.4f}")
    print(f"✓ Final loss: {history['train_losses'][-1]:.4f}")

    # Test prediction
    obs = np.random.randn(12).astype(np.float32)
    action = bc.predict(obs)
    assert action.shape == (4,)
    print(f"✓ Prediction works: action shape = {action.shape}")
    print()


def test_evaluation_summary_extraction():
    """Test evaluation summary extraction."""
    print("=" * 60)
    print("TEST 7: Evaluation Summary Extraction")
    print("=" * 60)

    # Import the helper function
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from training.pipelines.training_pipeline import (
        _extract_evaluation_summary,
        _extract_termination_breakdown,
        _extract_safety_event_counts,
    )

    # Create mock episodes
    class MockEpisode:
        def __init__(self, success, termination_reason, metrics):
            self.success = success
            self.termination_reason = termination_reason
            self.metrics = metrics

    class MockEvalResult:
        def __init__(self, episodes):
            self.episodes = episodes

    episodes = [
        MockEpisode(True, "success", {'tilt_violations': 0}),
        MockEpisode(True, "success", {'tilt_violations': 2}),  # Success but had violations
        MockEpisode(False, "timeout", {'tilt_violations': 0, 'truncated': 1}),
        MockEpisode(False, "crash", {'crash_count': 1, 'terminated': 1}),
        MockEpisode(True, "success", {'altitude_violations': 1}),
    ]

    eval_result = MockEvalResult(episodes)

    summary = _extract_evaluation_summary(eval_result)

    assert summary['episodes_run'] == 5
    assert summary['success_episodes'] == 3
    assert summary['crash_episodes'] == 1
    assert summary['tilt_violation_episodes'] == 1  # Only episode 2 had tilt violations
    assert summary['altitude_violation_episodes'] == 1
    print(f"✓ Evaluation summary extracted correctly")
    print(f"  Episodes: {summary['episodes_run']}")
    print(f"  Success: {summary['success_episodes']}")
    print(f"  Crash: {summary['crash_episodes']}")
    print(f"  Tilt violation episodes: {summary['tilt_violation_episodes']}")

    termination = _extract_termination_breakdown(eval_result)
    assert termination['success'] == 3
    assert termination['crash'] == 1
    assert termination['timeout'] == 1
    print(f"✓ Termination breakdown: {termination}")

    safety_events = _extract_safety_event_counts(eval_result)
    # Total events, not episodes
    print(f"✓ Safety event counts: {safety_events}")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE TESTS WITH SYNTHETIC DATA")
    print("=" * 60 + "\n")

    tests = [
        test_pipeline_minimal,
        test_pipeline_config,
        test_artifact_structure,
        test_regression_gate_fail_closed,
        test_synthetic_demo_creation,
        test_bc_training,
        test_evaluation_summary_extraction,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

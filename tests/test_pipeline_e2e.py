"""
End-to-end test of the full training pipeline.

Tests train_from_demonstrations() with synthetic data through all phases:
1. Demo loading + quality filtering
2. BC training
3. Evaluation with gates
4. Artifact generation
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import json
import gymnasium as gym


# =============================================================================
# MOCK ENVIRONMENT
# =============================================================================

@dataclass
class MockSpec:
    id: str = "MockDroneHover-v0"


class MockEnv(gym.Env):
    """Minimal mock environment for testing."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.ones(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32
        )
        self.spec = MockSpec()
        self._step_count = 0
        self._episode_count = 0
        self.dt = 0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_count += 1
        obs = np.random.randn(12).astype(np.float32) * 0.1
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(12).astype(np.float32) * 0.1

        # Simulate reasonable success rate (high for testing)
        done = self._step_count >= 100
        truncated = done
        terminated = False

        # 10% chance of early success after step 50
        if self._step_count > 50 and np.random.random() < 0.1:
            terminated = True
            done = True

        reward = 1.0 - 0.01 * np.sum(action**2)

        info = {
            'crash_count': 0,
            'tilt_violations': 0,
            'altitude_violations': 0,
            'speed_violations': 0,
            'position_error': np.random.random() * 0.5,
        }

        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_synthetic_demos(output_dir: str, n_demos: int = 10):
    """Create synthetic demonstration data."""
    from training.imitation.demonstration import Demonstration, DemonstrationDataset, DemonstrationStep

    demos = []
    for i in range(n_demos):
        # Random episode length
        n_steps = np.random.randint(80, 120)

        # Generate synthetic steps
        steps = []
        for t in range(n_steps):
            step = DemonstrationStep(
                observation=np.random.randn(12).astype(np.float32) * 0.1,
                action=np.random.randn(4).astype(np.float32) * 0.3,
                timestamp=t * 0.01,
                reward=0.95 + np.random.randn() * 0.05,
            )
            steps.append(step)

        demo = Demonstration(
            steps=steps,
            pilot_id=f"synthetic_pilot_{i % 3}",
            task_type="hover",
            quality_score=0.8 + np.random.random() * 0.2,
            source="synthetic",
            success=True,
            sample_rate_hz=100.0,
            duration_seconds=(n_steps - 1) * 0.01,  # Must set explicitly
        )
        demos.append(demo)

    # Save dataset
    dataset = DemonstrationDataset(demos)
    dataset.save(output_dir)
    return dataset


def create_baseline_metrics(output_path: str):
    """Create baseline metrics JSON for regression testing."""
    baseline = {
        'success_rate': {'mean': 0.85, 'std': 0.05},
        'crash_episodes': {'mean': 0, 'std': 0},
        'shield_override_rate': {'mean': 0.001, 'std': 0.001},
        'shield_clamp_rate': {'mean': 0.02, 'std': 0.01},
        'mean_position_error': {'mean': 0.3, 'std': 0.1},
        'smoothness_score': {'mean': 0.8, 'std': 0.1},
    }

    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    return baseline


# =============================================================================
# END-TO-END TEST
# =============================================================================

def test_full_pipeline():
    """Test the complete train_from_demonstrations pipeline."""
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE TEST")
    print("=" * 70 + "\n")

    from training.pipelines import (
        train_from_demonstrations,
        quick_train,
        load_artifact,
        PipelineConfig,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # =================================================================
        # PHASE 0: Setup
        # =================================================================
        print("PHASE 0: Setup")
        print("-" * 40)

        demo_dir = tmpdir / "demos"
        demo_dir.mkdir()
        output_dir = tmpdir / "artifacts"
        output_dir.mkdir()
        baseline_path = tmpdir / "baseline.json"

        # Create synthetic demos
        dataset = create_synthetic_demos(str(demo_dir), n_demos=5)
        print(f"  Created {len(dataset)} synthetic demos")
        print(f"  Total transitions: {dataset.total_transitions}")

        # Create baseline for regression testing
        create_baseline_metrics(str(baseline_path))
        print(f"  Created baseline metrics at {baseline_path}")

        # Create mock environment
        env = MockEnv()
        print(f"  Created mock environment: {env.spec.id}")
        print()

        # =================================================================
        # PHASE 1: Run Pipeline with BC only (no RL)
        # =================================================================
        print("PHASE 1: Running Pipeline (BC only)")
        print("-" * 40)

        config = PipelineConfig(
            task_type="hover",
            bc_epochs=10,  # Small for testing
            bc_batch_size=32,
            bc_hidden_sizes=[64, 64],
            enable_rl=False,  # BC only for speed
            eval_episodes=20,  # Reduced for testing
            eval_with_shield=False,  # No shield for mock env
            min_success_rate=0.5,  # Relaxed for synthetic data
            max_crashes=2,  # Allow some crashes in mock
            baseline_path=str(baseline_path),
        )

        print(f"  Config: task_type={config.task_type}")
        print(f"  Config: bc_epochs={config.bc_epochs}")
        print(f"  Config: eval_episodes={config.eval_episodes}")
        print(f"  Config: min_success_rate={config.min_success_rate}")
        print()

        try:
            artifact = train_from_demonstrations(
                demo_path=str(demo_dir),
                output_dir=str(output_dir),
                env=env,
                config=config,
                verbose=True,
            )

            print()
            print("PHASE 2: Pipeline Completed")
            print("-" * 40)

            # =================================================================
            # PHASE 2: Validate Artifact
            # =================================================================
            print(f"  Artifact dir: {artifact.artifact_dir}")
            print(f"  Model path: {artifact.model_path}")
            print(f"  Promoted: {artifact.promoted}")
            print(f"  Promotion reason: {artifact.promotion_reason}")
            print()

            # Check artifact structure
            print("PHASE 3: Validating Artifact Structure")
            print("-" * 40)

            # Check evaluation summary
            summary = artifact.evaluation_summary
            print(f"  Episodes run: {summary.get('episodes_run', 'N/A')}")
            print(f"  Success episodes: {summary.get('success_episodes', 'N/A')}")
            print(f"  Crash episodes: {summary.get('crash_episodes', 'N/A')}")
            print(f"  Tilt violation episodes: {summary.get('tilt_violation_episodes', 'N/A')}")

            assert 'episodes_run' in summary, "Missing episodes_run"
            assert 'success_episodes' in summary, "Missing success_episodes"
            assert 'crash_episodes' in summary, "Missing crash_episodes"
            print("  ✓ evaluation_summary has required fields")

            # Check termination breakdown
            term = artifact.termination_breakdown
            print(f"  Termination breakdown: {term}")
            assert isinstance(term, dict), "termination_breakdown should be dict"
            print("  ✓ termination_breakdown present")

            # Check safety event counts
            safety = artifact.safety_event_counts
            print(f"  Safety event counts: {safety}")
            assert isinstance(safety, dict), "safety_event_counts should be dict"
            print("  ✓ safety_event_counts present")

            # Check intervention breakdown
            interv = artifact.intervention_breakdown
            print(f"  Intervention breakdown: {interv}")
            assert isinstance(interv, dict), "intervention_breakdown should be dict"
            print("  ✓ intervention_breakdown present")

            # Check provenance
            print()
            print("PHASE 4: Validating Provenance")
            print("-" * 40)

            print(f"  Git commit: {artifact.git_commit or 'N/A'}")
            print(f"  Git branch: {artifact.git_branch or 'N/A'}")
            print(f"  Model hash: {artifact.model_hash[:20] if artifact.model_hash else 'N/A'}...")
            print(f"  Env ID: {artifact.env_id}")

            # Check env_id is set
            assert artifact.env_id, "env_id should not be empty"
            print("  ✓ env_id is set")

            # Check model exists
            model_path = Path(artifact.model_path)
            assert model_path.exists(), f"Model file not found: {model_path}"
            print(f"  ✓ Model file exists ({model_path.stat().st_size} bytes)")

            # =================================================================
            # PHASE 5: Test Artifact Serialization
            # =================================================================
            print()
            print("PHASE 5: Testing Artifact Serialization")
            print("-" * 40)

            artifact_dict = artifact.to_dict()

            # Check key fields in serialized form
            assert 'evaluation_summary' in artifact_dict
            assert 'termination_breakdown' in artifact_dict
            assert 'safety_event_counts' in artifact_dict
            assert 'git' in artifact_dict
            assert 'commit' in artifact_dict['git']
            print("  ✓ Artifact serializes correctly")

            # Save and reload
            manifest_path = output_dir / "test_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(artifact_dict, f, indent=2)
            print(f"  ✓ Manifest saved to {manifest_path}")

            # =================================================================
            # PHASE 6: Test load_artifact
            # =================================================================
            print()
            print("PHASE 6: Testing load_artifact")
            print("-" * 40)

            loaded = load_artifact(str(artifact.artifact_dir))
            assert loaded is not None, "Failed to load artifact"
            assert loaded.model_path == artifact.model_path
            assert loaded.promoted == artifact.promoted
            print("  ✓ Artifact loads correctly")

            # =================================================================
            # PHASE 7: Test quick_train
            # =================================================================
            print()
            print("PHASE 7: Testing quick_train")
            print("-" * 40)

            quick_output = tmpdir / "quick_artifacts"
            quick_artifact = quick_train(
                demo_path=str(demo_dir),
                output_dir=str(quick_output),
                env=env,
                bc_epochs=5,
            )

            assert quick_artifact is not None
            assert Path(quick_artifact.model_path).exists()
            print(f"  ✓ quick_train completed")
            print(f"  Model: {quick_artifact.model_path}")

            # =================================================================
            # SUMMARY
            # =================================================================
            print()
            print("=" * 70)
            print("END-TO-END TEST PASSED")
            print("=" * 70)
            print()
            print("Summary:")
            print(f"  - Pipeline ran successfully")
            print(f"  - Artifact promoted: {artifact.promoted}")
            print(f"  - Gates passed: {artifact.gates_passed}/{artifact.gates_total}")
            print(f"  - Success rate: {artifact.success_rate:.1%}")
            print(f"  - Crashes: {summary.get('crash_episodes', 0)}")
            print()

            return True

        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_pipeline_with_failing_gates():
    """Test that pipeline correctly fails gates when thresholds aren't met."""
    print("\n" + "=" * 70)
    print("FAILING GATES TEST")
    print("=" * 70 + "\n")

    from training.pipelines import train_from_demonstrations, PipelineConfig
    from training.imitation.demonstration import Demonstration, DemonstrationDataset, DemonstrationStep

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal demos
        demo_dir = tmpdir / "demos"
        demo_dir.mkdir()
        output_dir = tmpdir / "artifacts"
        output_dir.mkdir()

        # Create just 2 demos with enough steps/duration to pass filters
        demos = []
        for i in range(2):
            n_steps = 60
            steps = [
                DemonstrationStep(
                    observation=np.random.randn(12).astype(np.float32) * 0.1,
                    action=np.random.randn(4).astype(np.float32) * 0.3,
                    timestamp=t * 0.01,
                )
                for t in range(n_steps)
            ]
            demo = Demonstration(
                steps=steps,
                quality_score=0.9,
                sample_rate_hz=100.0,
                duration_seconds=(n_steps - 1) * 0.01,
            )
            demos.append(demo)

        dataset = DemonstrationDataset(demos)
        dataset.save(str(demo_dir))

        # Create env that always "crashes"
        class CrashyEnv(MockEnv):
            def step(self, action):
                self._step_count += 1
                obs = np.random.randn(12).astype(np.float32)

                # Always crash after 10 steps
                if self._step_count >= 10:
                    return obs, -10.0, True, False, {
                        'crash_count': 1,
                        'tilt_violations': 5,
                    }

                return obs, 1.0, False, False, {'crash_count': 0}

        env = CrashyEnv()

        # Config with strict thresholds
        config = PipelineConfig(
            task_type="hover",
            bc_epochs=3,
            eval_episodes=5,
            eval_with_shield=False,
            min_success_rate=0.95,  # Very strict
            max_crashes=0,  # Zero tolerance
        )

        try:
            artifact = train_from_demonstrations(
                demo_path=str(demo_dir),
                output_dir=str(output_dir),
                env=env,
                config=config,
                verbose=False,
            )

            print(f"  Promoted: {artifact.promoted}")
            print(f"  Reason: {artifact.promotion_reason}")
            print(f"  Gates passed: {artifact.gates_passed}/{artifact.gates_total}")

            # Should NOT be promoted due to crashes
            assert artifact.promoted == False, "Should not be promoted with crashes"
            assert artifact.gates_passed < artifact.gates_total
            print("  ✓ Correctly rejected promotion due to safety failures")

            return True

        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    import sys

    results = []

    # Run full pipeline test
    results.append(("Full Pipeline E2E", test_full_pipeline()))

    # Run failing gates test
    results.append(("Failing Gates", test_pipeline_with_failing_gates()))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    sys.exit(0 if all_passed else 1)

"""
End-to-end test of the training pipeline with a REAL drone environment.

Uses BaseDroneEnv with PyBullet physics for actual drone simulation.
"""

import numpy as np
import tempfile
from pathlib import Path
import json

# Import the real drone environment
from simulation.environments import BaseDroneEnv, create_clear_day

# Import training pipeline
from training.pipelines import (
    train_from_demonstrations,
    PipelineConfig,
)
from training.imitation.demonstration import (
    Demonstration,
    DemonstrationDataset,
    DemonstrationStep,
)


def create_default_platform_config():
    """Create a default quadcopter platform configuration."""
    return {
        "name": "TestQuadcopter",
        "platform_type": "quadcopter",
        "num_motors": 4,
        "mass": 1.0,  # kg
        "arm_length": 0.25,  # meters
        "max_thrust_per_motor": 10.0,  # N
        "max_rpm": 20000,
        "physics_params": {
            "ixx": 0.01,
            "iyy": 0.01,
            "izz": 0.02,
        },
    }


def create_synthetic_demos_for_env(env, output_dir: str, n_demos: int = 5):
    """Create synthetic demonstrations matching the environment's spaces."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    demos = []
    for i in range(n_demos):
        n_steps = np.random.randint(80, 120)

        steps = []
        for t in range(n_steps):
            # Generate observations within bounds
            obs = np.random.uniform(
                low=-1.0, high=1.0, size=(obs_dim,)
            ).astype(np.float32)

            # Generate actions within action space bounds
            action = np.random.uniform(
                low=env.action_space.low,
                high=env.action_space.high,
            ).astype(np.float32)

            step = DemonstrationStep(
                observation=obs,
                action=action,
                timestamp=t * 0.02,  # 50Hz control
                reward=1.0 - 0.01 * np.sum(action ** 2),
            )
            steps.append(step)

        demo = Demonstration(
            steps=steps,
            pilot_id=f"synthetic_pilot_{i % 3}",
            task_type="hover",
            quality_score=0.85 + np.random.random() * 0.15,
            source="synthetic",
            success=True,
            sample_rate_hz=50.0,
            duration_seconds=(n_steps - 1) * 0.02,
        )
        demos.append(demo)

    dataset = DemonstrationDataset(demos)
    dataset.save(output_dir)
    return dataset


def test_pipeline_with_real_env():
    """Test the full pipeline with a real PyBullet drone environment."""
    print("\n" + "=" * 70)
    print("PIPELINE TEST WITH REAL DRONE ENVIRONMENT")
    print("=" * 70 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        demo_dir = tmpdir / "demos"
        demo_dir.mkdir()
        output_dir = tmpdir / "artifacts"
        output_dir.mkdir()

        # =================================================================
        # Create REAL drone environment
        # =================================================================
        print("Creating real drone environment...")
        platform_config = create_default_platform_config()
        env_conditions = create_clear_day()

        env = BaseDroneEnv(
            platform_config=platform_config,
            environmental_conditions=env_conditions,
            render_mode=None,  # Headless for testing
            physics_hz=240,
            control_hz=48,
        )

        print(f"  Environment: {env.__class__.__name__}")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Physics Hz: {env.physics_hz}")
        print(f"  Control Hz: {env.control_hz}")
        print()

        # =================================================================
        # Create synthetic demos matching the environment
        # =================================================================
        print("Creating synthetic demonstrations...")
        dataset = create_synthetic_demos_for_env(env, str(demo_dir), n_demos=5)
        print(f"  Created {len(dataset)} demos")
        print(f"  Observation dim: {dataset.observation_dim}")
        print(f"  Action dim: {dataset.action_dim}")
        print(f"  Total transitions: {dataset.total_transitions}")
        print()

        # =================================================================
        # Run the pipeline
        # =================================================================
        print("Running training pipeline...")
        print("-" * 40)

        config = PipelineConfig(
            task_type="hover",
            bc_epochs=10,
            bc_batch_size=32,
            bc_hidden_sizes=[64, 64],
            enable_rl=False,  # BC only for speed
            eval_episodes=10,  # Fewer episodes for faster test
            eval_with_shield=True,  # Test with shield
            min_success_rate=0.3,  # Relaxed for synthetic data
            max_crashes=5,  # Allow some crashes
        )

        try:
            artifact = train_from_demonstrations(
                demo_path=str(demo_dir),
                output_dir=str(output_dir),
                env=env,
                config=config,
                verbose=True,
            )

            print()
            print("=" * 70)
            print("PIPELINE COMPLETED")
            print("=" * 70)
            print()

            # =================================================================
            # Validate results
            # =================================================================
            print("Artifact Summary:")
            print(f"  Promoted: {artifact.promoted}")
            print(f"  Reason: {artifact.promotion_reason}")
            print(f"  Gates passed: {artifact.gates_passed}/{artifact.gates_total}")
            print(f"  Success rate: {artifact.success_rate:.1%}")
            print()

            print("Evaluation Summary:")
            summary = artifact.evaluation_summary
            print(f"  Episodes run: {summary.get('episodes_run', 'N/A')}")
            print(f"  Success episodes: {summary.get('success_episodes', 'N/A')}")
            print(f"  Crash episodes: {summary.get('crash_episodes', 'N/A')}")
            print(f"  Tilt violations: {summary.get('tilt_violation_episodes', 'N/A')}")
            print()

            print("Termination Breakdown:")
            for reason, count in artifact.termination_breakdown.items():
                print(f"  {reason}: {count}")
            print()

            print("Safety Event Counts:")
            for event, count in artifact.safety_event_counts.items():
                print(f"  {event}: {count}")
            print()

            print("Intervention Stats:")
            interv = artifact.intervention_breakdown
            print(f"  Clamp rate: {interv.get('clamp_rate', 0):.2%}")
            print(f"  Override rate: {interv.get('override_rate', 0):.2%}")
            print(f"  Emergency count: {interv.get('emergency_count', 0)}")
            print()

            print("Provenance:")
            print(f"  Git commit: {artifact.git_commit[:12]}...")
            print(f"  Model hash: {artifact.model_hash[:20]}...")
            print(f"  Env ID: {artifact.env_id}")
            print()

            # Check model file exists
            model_path = Path(artifact.model_path)
            assert model_path.exists(), f"Model file not found: {model_path}"
            print(f"  Model file: {model_path.name} ({model_path.stat().st_size} bytes)")

            # Check report files
            report_json = output_dir / "report.json"
            report_html = output_dir / "report.html"
            assert report_json.exists(), "report.json not found"
            assert report_html.exists(), "report.html not found"
            print(f"  Report JSON: {report_json.stat().st_size} bytes")
            print(f"  Report HTML: {report_html.stat().st_size} bytes")

            print()
            print("=" * 70)
            print("TEST PASSED - Pipeline works with real drone environment!")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Clean up environment
            env.close()


if __name__ == "__main__":
    import sys
    success = test_pipeline_with_real_env()
    sys.exit(0 if success else 1)

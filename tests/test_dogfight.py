"""
Tests for Dogfight Environment and Combat Training System.

Verifies:
- DogfightEnv basic functionality
- Self-play training components
- Combat analytics
"""

import numpy as np
import tempfile
from pathlib import Path

import pytest


def test_dogfight_env_creation():
    """Test creating a dogfight environment."""
    from simulation.environments.combat import (
        DogfightEnv,
        DogfightConfig,
        create_1v1_dogfight,
        create_2v2_dogfight,
    )

    # Default config
    env = DogfightEnv()
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None

    # Check spaces
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)

    env.close()
    print("Basic dogfight env creation: PASSED")


def test_dogfight_1v1():
    """Test 1v1 dogfight environment."""
    from simulation.environments.combat import create_1v1_dogfight

    env = create_1v1_dogfight()
    obs, info = env.reset()

    # Run a few steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("1v1 dogfight simulation: PASSED")


def test_dogfight_2v2():
    """Test 2v2 team dogfight."""
    from simulation.environments.combat import create_2v2_dogfight

    env = create_2v2_dogfight()
    obs, info = env.reset()

    # Run a few steps
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()
    print("2v2 team dogfight: PASSED")


def test_dogfight_swarm():
    """Test swarm battle."""
    from simulation.environments.combat import create_swarm_battle

    env = create_swarm_battle(red_count=3, blue_count=3)
    obs, info = env.reset()

    # Run a few steps
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()
    print("Swarm battle (3v3): PASSED")


def test_dogfight_combat_events():
    """Test that combat events are generated."""
    from simulation.environments.combat import DogfightEnv, DogfightConfig

    # Short match with respawn
    config = DogfightConfig(
        num_red=1,
        num_blue=1,
        respawn_enabled=True,
        max_match_time=60.0,
        kills_to_win=3,
    )

    env = DogfightEnv(config=config)
    obs, info = env.reset()

    total_combat_events = 0
    total_rewards = 0

    for _ in range(1000):
        # Always try to fire
        action = np.array([0.0, 0.0, 0.7, 0, 1.0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        total_rewards += reward

        if "combat_events" in info:
            total_combat_events += len(info["combat_events"])

        if terminated or truncated:
            break

    print(f"Combat events generated: {total_combat_events}")
    print(f"Total reward: {total_rewards:.2f}")

    env.close()
    print("Combat events generation: PASSED")


def test_dogfight_episode_stats():
    """Test episode statistics tracking."""
    from simulation.environments.combat import DogfightEnv

    env = DogfightEnv()
    obs, info = env.reset()

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Check stats are tracked
    assert "episode_stats" in info
    stats = info["episode_stats"]

    expected_keys = [
        "kills", "deaths", "damage_dealt", "damage_taken",
        "shots_fired", "shots_hit", "time_alive"
    ]

    for key in expected_keys:
        assert key in stats, f"Missing stat: {key}"

    print(f"Episode stats: {stats}")

    env.close()
    print("Episode stats tracking: PASSED")


def test_opponent_pool():
    """Test opponent pool management."""
    from training.combat import OpponentPool, OpponentRecord

    with tempfile.TemporaryDirectory() as tmpdir:
        pool = OpponentPool(tmpdir, max_size=10)

        # Pool should be empty initially
        assert len(pool.opponents) == 0

        # Test selection on empty pool
        selected = pool.select_opponent(1500, strategy="match")
        assert selected is None

        # Create mock opponent records directly
        for i in range(5):
            record = OpponentRecord(
                opponent_id=f"test_{i}",
                checkpoint_path=f"{tmpdir}/test_{i}.zip",
                generation=i,
                elo_rating=1400 + i * 50,
            )
            pool.opponents[record.opponent_id] = record

        # Test selection strategies
        weakest = pool.select_opponent(1500, strategy="weakest")
        assert weakest is not None
        assert weakest.elo_rating == 1400

        strongest = pool.select_opponent(1500, strategy="strongest")
        assert strongest is not None
        assert strongest.elo_rating == 1600

        matched = pool.select_opponent(1500, strategy="match")
        assert matched is not None

        pool.save_pool()

        # Test loading
        pool2 = OpponentPool(tmpdir, max_size=10)
        assert len(pool2.opponents) == 5

    print("Opponent pool management: PASSED")


def test_elo_updates():
    """Test Elo rating updates."""
    from training.combat import OpponentPool, OpponentRecord

    with tempfile.TemporaryDirectory() as tmpdir:
        pool = OpponentPool(tmpdir)

        # Add opponent
        record = OpponentRecord(
            opponent_id="test_opp",
            checkpoint_path=f"{tmpdir}/test.zip",
            generation=0,
            elo_rating=1500,
        )
        pool.opponents[record.opponent_id] = record

        # Agent wins
        agent_delta, opp_delta = pool.update_elo(
            "agent",
            "test_opp",
            agent_score=1.0,  # Win
        )

        assert agent_delta > 0, "Agent should gain Elo on win"
        assert opp_delta < 0, "Opponent should lose Elo on loss"

        # Check opponent record updated
        assert pool.opponents["test_opp"].losses == 1

    print("Elo rating updates: PASSED")


def test_combat_analyzer():
    """Test combat analytics."""
    from training.combat import (
        CombatAnalyzer,
        EngagementType,
        ManeuverType,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = CombatAnalyzer(tmpdir)

        # Test engagement classification
        attacker_pos = np.array([0, 0, 500])
        attacker_vel = np.array([100, 0, 0])
        defender_pos = np.array([200, 0, 500])
        defender_vel = np.array([-100, 0, 0])

        eng_type = analyzer.classify_engagement(
            attacker_pos, attacker_vel,
            defender_pos, defender_vel
        )

        assert eng_type == EngagementType.HEAD_ON

        # Tail chase
        defender_vel = np.array([100, 0, 0])  # Same direction
        eng_type = analyzer.classify_engagement(
            attacker_pos, attacker_vel,
            defender_pos, defender_vel
        )

        assert eng_type == EngagementType.TAIL_CHASE

    print("Combat analyzer: PASSED")


def test_maneuver_detection():
    """Test maneuver detection from trajectory."""
    from training.combat import CombatAnalyzer, ManeuverType

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = CombatAnalyzer(tmpdir)

        # Create a climbing trajectory
        n_points = 60
        positions = np.zeros((n_points, 3))
        velocities = np.zeros((n_points, 3))

        for i in range(n_points):
            t = i / 60
            positions[i] = [100 * t, 0, 500 + 300 * t]  # Climbing
            velocities[i] = [100, 0, 300]

        maneuver = analyzer.detect_maneuver(positions, velocities)

        # Should detect a climb
        assert maneuver in [ManeuverType.CLIMB, ManeuverType.VERTICAL_EXTENSION]

    print("Maneuver detection: PASSED")


def test_dogfight_config_variations():
    """Test various dogfight configurations."""
    from simulation.environments.combat import DogfightEnv, DogfightConfig

    # Test different respawn settings
    config_no_respawn = DogfightConfig(
        respawn_enabled=False,
        win_condition="last_alive",
    )
    env = DogfightEnv(config=config_no_respawn)
    obs, _ = env.reset()
    env.close()

    # Test different weapon loadouts
    config_guns_only = DogfightConfig(
        weapons_config=[
            {"type": "gun", "ammo": 1000, "range": 400, "damage": 5, "cooldown": 0.05},
        ]
    )
    env = DogfightEnv(config=config_guns_only)
    obs, _ = env.reset()
    env.close()

    # Test large arena
    config_large = DogfightConfig(
        arena_size=10000.0,
        arena_height_max=5000.0,
    )
    env = DogfightEnv(config=config_large)
    obs, _ = env.reset()
    env.close()

    print("Config variations: PASSED")


def test_set_opponent_policy():
    """Test setting opponent policy."""
    from simulation.environments.combat import DogfightEnv

    env = DogfightEnv()

    # Create a simple policy function
    class SimplePolicy:
        def predict(self, obs, deterministic=False):
            # Always go straight and shoot
            return np.array([0.0, 0.0, 0.5, 0, 1.0, 0]), None

    policy = SimplePolicy()
    env.set_opponent_policy(policy)

    # Run with the policy
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print("Opponent policy setting: PASSED")


def run_all_tests():
    """Run all dogfight tests."""
    print("\n" + "=" * 60)
    print("DOGFIGHT ENVIRONMENT TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_dogfight_env_creation,
        test_dogfight_1v1,
        test_dogfight_2v2,
        test_dogfight_swarm,
        test_dogfight_combat_events,
        test_dogfight_episode_stats,
        test_opponent_pool,
        test_elo_updates,
        test_combat_analyzer,
        test_maneuver_detection,
        test_dogfight_config_variations,
        test_set_opponent_policy,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

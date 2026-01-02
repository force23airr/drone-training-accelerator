#!/usr/bin/env python3
"""
Test the 3D Visualizer with a Dogfight Training Run

This script:
1. Runs a dogfight simulation with AI opponents
2. Streams state via ZeroMQ for visualization
3. Optionally records a replay file

Usage:
    # Terminal 1: Run this script
    python scripts/test_visualizer.py

    # Terminal 2: Launch the visualizer
    python -m visualization.renderer.dogfight_viewer --connect localhost:5555
"""

import sys
import time
import argparse
import numpy as np

sys.path.insert(0, '.')

from simulation.environments.combat.dogfight_env import (
    DogfightEnv,
    DogfightConfig,
    create_1v1_dogfight,
    create_2v2_dogfight,
)
from visualization.streaming import CombatEvent, EventType, StateStreamer, WeaponType


EVENT_TYPE_MAP = {
    "hit": EventType.HIT,
    "kill": EventType.KILL,
    "respawn": EventType.RESPAWN,
    "out_of_bounds": EventType.OUT_OF_BOUNDS,
    "crash": EventType.CRASH,
}

WEAPON_TYPE_MAP = {
    "gun": WeaponType.GUN,
    "missile_ir": WeaponType.MISSILE_IR,
    "missile_radar": WeaponType.MISSILE_RADAR,
    "laser": WeaponType.LASER,
}


def _coerce_weapon_type(raw_weapon):
    if isinstance(raw_weapon, WeaponType):
        return raw_weapon
    if hasattr(raw_weapon, "value"):
        return _coerce_weapon_type(raw_weapon.value)
    if isinstance(raw_weapon, str):
        return WEAPON_TYPE_MAP.get(raw_weapon, WeaponType.GUN)
    if isinstance(raw_weapon, int):
        try:
            return WeaponType(raw_weapon)
        except ValueError:
            return WeaponType.GUN
    return WeaponType.GUN


def _build_protocol_event(event, env):
    event_type = EVENT_TYPE_MAP.get(event.get("type"))
    if event_type is None:
        return None

    attacker_id = event.get("attacker")
    target_id = event.get("target")
    if attacker_id is None or target_id is None:
        return None

    weapon = _coerce_weapon_type(event.get("weapon"))
    damage = float(event.get("damage", 0.0))

    position = (0.0, 0.0, 0.0)
    target = env.drones.get(target_id)
    if target is not None:
        position = tuple(float(x) for x in target.position)
    else:
        attacker = env.drones.get(attacker_id)
        if attacker is not None:
            position = tuple(float(x) for x in attacker.position)

    return CombatEvent(
        event_type=event_type,
        timestamp=time.time(),
        attacker_id=int(attacker_id),
        target_id=int(target_id),
        weapon=weapon,
        damage=damage,
        position=position,
    )


def run_demo_match(
    num_red: int = 2,
    num_blue: int = 2,
    port: int = 5555,
    record_replay: bool = True,
    max_episodes: int = 5,
):
    """
    Run a demo dogfight match with visualization streaming.

    Args:
        num_red: Number of red team drones
        num_blue: Number of blue team drones
        port: ZeroMQ streaming port
        record_replay: Whether to record replay file
        max_episodes: Number of episodes to run
    """
    print("=" * 60)
    print("DOGFIGHT VISUALIZATION TEST")
    print("=" * 60)
    print()
    print(f"Teams: {num_red}v{num_blue}")
    print(f"Streaming on port: {port}")
    print(f"Recording replay: {record_replay}")
    print()
    print("Start the visualizer in another terminal:")
    print(f"  python -m visualization.renderer.dogfight_viewer --connect localhost:{port}")
    print()
    print("Press Ctrl+C to stop")
    print()

    # Create environment
    config = DogfightConfig(
        num_red=num_red,
        num_blue=num_blue,
        arena_size=2000.0,
        arena_height_min=100.0,
        arena_height_max=1500.0,
        respawn_enabled=True,
        max_match_time=180.0,  # 3 minutes per match
        kills_to_win=10,
    )
    env = DogfightEnv(config=config)

    # Create state streamer
    replay_path = "dogfight_demo.dfrp" if record_replay else None
    streamer = StateStreamer(
        port=port,
        enable_replay=record_replay,
        replay_path=replay_path,
    )
    streamer.set_match_info(env)
    streamer.start()

    print(f"StateStreamer started on port {port}")
    if record_replay:
        print(f"Recording to: {replay_path}")
    print()

    try:
        for episode in range(max_episodes):
            print(f"\n--- Episode {episode + 1}/{max_episodes} ---")

            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                # Use tactical CombatAI for the agent too (creates real dogfighting)
                agent_drone = env.drones[env.agent_drone_id]
                action, maneuver = env._get_tactical_action(agent_drone)
                agent_drone._current_maneuver = maneuver  # Store for visualization

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Convert combat events to protocol events
                for event in info.get("combat_events", []):
                    proto_event = _build_protocol_event(event, env)
                    if proto_event:
                        streamer.add_event(proto_event)

                # Stream state to visualizer
                streamer.publish_frame(env)

                # Print events
                for event in info.get("combat_events", []):
                    if event["type"] == "kill":
                        print(f"  💥 KILL: Drone {event['attacker']} -> Drone {event['target']}")
                    elif event["type"] == "hit":
                        print(f"  🎯 HIT: Drone {event['attacker']} -> Drone {event['target']} ({event['damage']:.0f} dmg)")

                # Debug: Show positions and maneuvers every 60 frames
                if episode_steps % 60 == 0:
                    for drone in env.drones.values():
                        maneuver = getattr(drone, '_current_maneuver', 'N/A')
                        dist = np.linalg.norm(drone.position[:2])  # Distance from center
                        print(f"    Drone {drone.drone_id}: pos=({drone.position[0]:.0f},{drone.position[1]:.0f},{drone.position[2]:.0f}) center_dist={dist:.0f}m HP={drone.health:.0f} [{maneuver}]")

                # 60 FPS timing
                time.sleep(1/60)

                if terminated or truncated:
                    break

            print(f"Episode ended: {episode_steps} steps, reward: {episode_reward:.1f}")
            print(f"Score: RED {env.red_kills} - {env.blue_kills} BLUE")

            # Brief pause between episodes
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        streamer.stop()
        env.close()

        if record_replay:
            print(f"\nReplay saved to: {replay_path}")
            print(f"Play it back with:")
            print(f"  python -m visualization.renderer.dogfight_viewer --replay {replay_path}")


def run_training_with_viz(
    port: int = 5555,
    total_timesteps: int = 50000,
):
    """
    Run actual training with visualization.

    Args:
        port: ZeroMQ streaming port
        total_timesteps: Total training timesteps
    """
    print("=" * 60)
    print("TRAINING WITH VISUALIZATION")
    print("=" * 60)
    print()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("stable-baselines3 not installed. Running demo instead.")
        run_demo_match(port=port)
        return

    # Custom callback to stream state
    class VisualizationCallback(BaseCallback):
        def __init__(self, env, streamer, verbose=0):
            super().__init__(verbose)
            self.env = env
            self.streamer = streamer
            self.last_stream_time = 0

        def _on_step(self) -> bool:
            # Stream at 60 FPS
            current_time = time.time()
            if current_time - self.last_stream_time >= 1/60:
                infos = self.locals.get("infos", [])
                if isinstance(infos, dict):
                    infos = [infos]
                for info in infos:
                    for event in info.get("combat_events", []):
                        proto_event = _build_protocol_event(event, self.env)
                        if proto_event:
                            self.streamer.add_event(proto_event)
                self.streamer.publish_frame(self.env)
                self.last_stream_time = current_time
            return True

    # Create environment
    env = create_2v2_dogfight()

    # Create streamer
    streamer = StateStreamer(port=port, enable_replay=True, replay_path="training_replay.dfrp")
    streamer.set_match_info(env)
    streamer.start()

    print(f"Streaming on port: {port}")
    print(f"Training for {total_timesteps} timesteps")
    print()
    print("Start the visualizer in another terminal:")
    print(f"  python -m visualization.renderer.dogfight_viewer --connect localhost:{port}")
    print()

    try:
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )

        # Create callback
        viz_callback = VisualizationCallback(env, streamer)

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=viz_callback,
            progress_bar=True,
        )

        # Save model
        model.save("dogfight_ppo")
        print("\nModel saved to: dogfight_ppo.zip")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted")

    finally:
        streamer.stop()
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Test Dogfight Visualizer')
    parser.add_argument('--mode', choices=['demo', 'train'], default='demo',
                       help='Mode: demo (random actions) or train (PPO)')
    parser.add_argument('--port', type=int, default=5555, help='ZeroMQ port')
    parser.add_argument('--red', type=int, default=2, help='Red team size')
    parser.add_argument('--blue', type=int, default=2, help='Blue team size')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--no-record', action='store_true', help='Disable replay recording')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo_match(
            num_red=args.red,
            num_blue=args.blue,
            port=args.port,
            record_replay=not args.no_record,
            max_episodes=args.episodes,
        )
    else:
        run_training_with_viz(
            port=args.port,
            total_timesteps=args.timesteps,
        )


if __name__ == '__main__':
    main()

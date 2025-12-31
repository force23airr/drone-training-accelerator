#!/usr/bin/env python3
"""
Example: Train an Urban Navigation Agent

Train a drone to navigate through indoor environments with obstacles.
"""

import argparse
from pathlib import Path

from simulation import UrbanNavigationEnv, get_platform_config
from training import ParallelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train urban navigation agent")
    parser.add_argument("--scenario", type=str, default="indoor",
                        choices=["outdoor_urban", "indoor", "parking_garage"])
    parser.add_argument("--waypoints", type=int, default=5)
    parser.add_argument("--obstacles", type=str, default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./models/urban_nav")
    args = parser.parse_args()

    print("=" * 60)
    print("Urban Navigation Agent Training")
    print("=" * 60)
    print(f"  Scenario: {args.scenario}")
    print(f"  Waypoints: {args.waypoints}")
    print(f"  Obstacle density: {args.obstacles}")

    # Create urban navigation environment
    env = UrbanNavigationEnv(
        platform_config=get_platform_config("inspection_drone"),
        scenario=args.scenario,
        num_waypoints=args.waypoints,
        obstacle_density=args.obstacles,
        render_mode=None
    )

    print(f"\n  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Train
    trainer = ParallelTrainer(
        env=env,
        mission=None,  # Urban env has built-in reward
        num_envs=args.num_envs,
        algorithm="ppo",
        output_dir=args.output_dir
    )

    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    trainer.train(total_timesteps=args.timesteps)

    # Evaluate
    print("\nEvaluating...")
    metrics = trainer.evaluate(n_eval_episodes=10)
    print(f"  Mean reward: {metrics['mean_reward']:.2f}")

    # Save
    model_path = Path(args.output_dir) / f"urban_nav_{args.scenario}_final"
    trainer.save(model_path)

    trainer.close()
    env.close()

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()

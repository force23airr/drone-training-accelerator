#!/usr/bin/env python3
"""
Quickstart Training Script

Simple entry point to start training a drone agent.
Run: python quickstart_train.py
"""

import argparse
from pathlib import Path

from simulation.environments import EnvironmentalSimulator
from simulation.platforms.platform_configs import get_platform_config, list_platforms
from simulation.control import ActionMode
from simulation.wrappers import ActionAdapterConfig, make_action_adapted_env
from training.suites.mission_suites import MissionSuite
from training.parallel.parallel_trainer import ParallelTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train autonomous drone agents"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="quadcopter_basic",
        help="Drone platform to train (use --list-platforms to see options)"
    )
    parser.add_argument(
        "--mission",
        type=str,
        default="hover_stability",
        help="Mission suite to train on"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available drone platforms"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization during training"
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default=None,
        help="Action interface: separated, motor_thrusts, attitude, attitude_rates, velocity"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_platforms:
        print("Available drone platforms:")
        for platform in list_platforms():
            print(f"  - {platform}")
        return

    print(f"=" * 60)
    print(f"Autonomous Flight Training Platform")
    print(f"=" * 60)
    print(f"Platform: {args.platform}")
    print(f"Mission:  {args.mission}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"=" * 60)

    # Load platform configuration
    platform_config = get_platform_config(args.platform)
    print(f"\nLoaded platform: {platform_config['name']}")

    # Initialize environment
    env = EnvironmentalSimulator(
        platform_config=platform_config,
        render_mode="human" if args.render else None
    )

    if args.action_mode:
        mode_str = args.action_mode.lower().strip()
        if mode_str == "separated":
            mode = ActionMode.VELOCITY
        else:
            mode = ActionMode(mode_str)
        env = make_action_adapted_env(env, ActionAdapterConfig(action_mode=mode))
        print(f"Action mode: {mode.value}")

    # Load mission suite
    mission = MissionSuite(args.mission)
    print(f"Loaded mission suite: {mission.name}")

    # Initialize trainer
    trainer = ParallelTrainer(
        env=env,
        mission=mission,
        num_envs=args.num_envs,
        output_dir=args.output_dir
    )

    # Start training
    print("\nStarting training...")
    trainer.train(total_timesteps=args.timesteps)

    # Save final model
    output_path = Path(args.output_dir) / f"{args.platform}_{args.mission}_final"
    trainer.save(output_path)
    print(f"\nTraining complete! Model saved to: {output_path}")


if __name__ == "__main__":
    main()

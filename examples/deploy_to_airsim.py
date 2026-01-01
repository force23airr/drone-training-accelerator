#!/usr/bin/env python3
"""
Example: Deploy Trained Policy to AirSim

Visualize your trained drone policy in photorealistic environments.

Requirements:
1. pip install airsim
2. Download and run AirSim: https://github.com/microsoft/AirSim/releases

Usage:
    python examples/deploy_to_airsim.py --model trained_models/quadcopter_basic_hover_stability_final
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Deploy trained drone policy to AirSim"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/quadcopter_basic_hover_stability_final",
        help="Path to trained model"
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="florida_street",
        choices=["florida_street", "mountain_recon", "urban_sar", "warehouse", "coastal_patrol"],
        help="Environment preset"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC", "TD3"],
        help="RL algorithm used for training"
    )
    parser.add_argument(
        "--record",
        type=str,
        help="Record video to this path (e.g., demo.mp4)"
    )
    parser.add_argument(
        "--list-environments",
        action="store_true",
        help="List available environments"
    )

    args = parser.parse_args()

    # Check if AirSim is available
    try:
        from simulation.physics.airsim_integration import AIRSIM_AVAILABLE
    except ImportError:
        print("Error: AirSim integration not found")
        print("Make sure you're in the project root directory")
        sys.exit(1)

    if not AIRSIM_AVAILABLE:
        print("=" * 60)
        print("AirSim not installed!")
        print("=" * 60)
        print()
        print("To use photorealistic environments, install AirSim:")
        print()
        print("  1. Install Python client:")
        print("     pip install airsim")
        print()
        print("  2. Download AirSim simulator:")
        print("     https://github.com/microsoft/AirSim/releases")
        print()
        print("  3. Run the simulator (e.g., Blocks.exe)")
        print()
        print("  4. Then run this script again")
        print()
        sys.exit(1)

    from simulation.physics.airsim_integration import (
        deploy_to_airsim,
        visualize_policy,
        record_video,
        list_environments,
        PRESET_ENVIRONMENTS,
    )

    if args.list_environments:
        print("Available environments:")
        print("-" * 40)
        for name, config in PRESET_ENVIRONMENTS.items():
            print(f"  {name}")
            print(f"    Type: {config.environment_type.value}")
            print(f"    Description: {config.description}")
            print()
        return

    print("=" * 60)
    print("AirSim Policy Deployment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Environment: {args.environment}")
    print(f"Algorithm: {args.algorithm}")
    print()

    if args.record:
        print(f"Recording video to: {args.record}")
        record_video(
            model_path=args.model,
            output_path=args.record,
            environment=args.environment,
            max_steps=args.max_steps,
            algorithm=args.algorithm,
        )
    elif args.episodes == 1:
        print("Running single visualization episode...")
        visualize_policy(
            model_path=args.model,
            environment=args.environment,
            max_steps=args.max_steps,
            algorithm=args.algorithm,
        )
    else:
        print(f"Running {args.episodes} evaluation episodes...")
        results = deploy_to_airsim(
            model_path=args.model,
            environment=args.environment,
            num_episodes=args.episodes,
            algorithm=args.algorithm,
        )


if __name__ == "__main__":
    main()

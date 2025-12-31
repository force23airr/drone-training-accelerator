#!/usr/bin/env python3
"""
Example: Train a Hover Stability Agent

This script demonstrates the basic training workflow:
1. Create environment with platform configuration
2. Set up mission suite
3. Train with parallel environments
4. Evaluate and save the model
"""

import argparse
from pathlib import Path

from simulation import BaseDroneEnv, get_platform_config, create_clear_day
from training import ParallelTrainer, MissionSuite


def main():
    parser = argparse.ArgumentParser(description="Train a hover stability agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--output-dir", type=str, default="./models/hover_agent")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("Hover Stability Agent Training")
    print("=" * 60)

    # 1. Create environment
    print("\n[1/4] Creating environment...")
    platform_config = get_platform_config("quadcopter_basic")
    conditions = create_clear_day()

    env = BaseDroneEnv(
        platform_config=platform_config,
        environmental_conditions=conditions,
        render_mode=None  # No rendering during training
    )

    print(f"  Platform: {platform_config['name']}")
    print(f"  Mass: {platform_config['mass']} kg")
    print(f"  Motors: {platform_config['num_motors']}")

    # 2. Set up mission
    print("\n[2/4] Setting up mission suite...")
    mission = MissionSuite("hover_stability")
    print(f"  Mission: {mission.name}")
    print(f"  Difficulty: {mission.difficulty}")
    print(f"  Objectives: {mission.config.objectives}")

    # 3. Train
    print("\n[3/4] Starting training...")
    trainer = ParallelTrainer(
        env=env,
        mission=mission,
        num_envs=args.num_envs,
        algorithm="ppo",
        output_dir=args.output_dir
    )

    trainer.train(total_timesteps=args.timesteps)

    # 4. Evaluate and save
    print("\n[4/4] Evaluating trained agent...")
    metrics = trainer.evaluate(n_eval_episodes=10, render=args.render)
    print(f"  Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")

    # Save final model
    model_path = Path(args.output_dir) / "hover_agent_final"
    trainer.save(model_path)

    # Save training config
    config = trainer.get_training_config()
    print(f"\nTraining config saved to: {args.output_dir}/training_config.json")

    trainer.close()
    env.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

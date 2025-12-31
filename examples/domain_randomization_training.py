#!/usr/bin/env python3
"""
Example: Training with Domain Randomization

Train a robust policy using domain randomization for sim-to-real transfer.
"""

import argparse
from pathlib import Path

from simulation import (
    BaseDroneEnv,
    get_platform_config,
    create_random_conditions,
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    WindModel,
)
from training import ParallelTrainer, MissionSuite


def main():
    parser = argparse.ArgumentParser(description="Train with domain randomization")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./models/robust_policy")
    args = parser.parse_args()

    print("=" * 60)
    print("Domain Randomization Training")
    print("=" * 60)
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Parallel envs: {args.num_envs}")

    # Show what will be randomized
    print("\nConditions will be randomized each episode:")
    print("  - Weather: Clear, Rain, Snow, Fog, etc.")
    print("  - Time: Dawn, Day, Dusk, Night")
    print("  - Wind: Speed, gusts, turbulence")
    print("  - Sensors: Noise levels based on conditions")

    # Create base environment with domain randomization enabled
    env = BaseDroneEnv(
        platform_config=get_platform_config("quadcopter_basic"),
        environmental_conditions=create_random_conditions(difficulty=args.difficulty),
        domain_randomization=True,  # Randomize mass, initial position, etc.
        render_mode=None
    )

    # Use hover stability mission but in varying conditions
    mission = MissionSuite("hover_stability")

    # Train
    trainer = ParallelTrainer(
        env=env,
        mission=mission,
        num_envs=args.num_envs,
        algorithm="ppo",
        output_dir=args.output_dir
    )

    print("\nStarting training...")
    trainer.train(total_timesteps=args.timesteps)

    # Evaluate in different conditions
    print("\nEvaluating in various conditions...")
    conditions_to_test = [
        ("Clear day", create_random_conditions("easy")),
        ("Windy", EnvironmentalConditions(
            wind=WindModel(base_speed=10.0, turbulence_intensity=1.0)
        )),
        ("Night", EnvironmentalConditions(time_of_day=TimeOfDay.NIGHT)),
        ("Rain", EnvironmentalConditions(weather=WeatherType.RAIN)),
    ]

    for name, conditions in conditions_to_test:
        env.set_conditions(conditions)
        metrics = trainer.evaluate(n_eval_episodes=5)
        print(f"  {name}: mean_reward = {metrics['mean_reward']:.2f}")

    # Save
    model_path = Path(args.output_dir) / "robust_policy_final"
    trainer.save(model_path)

    trainer.close()
    env.close()

    print(f"\nRobust policy saved to: {model_path}")


if __name__ == "__main__":
    main()

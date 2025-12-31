#!/usr/bin/env python3
"""
Example: Evaluate a Trained Agent

Load and visualize a trained drone policy.
"""

import argparse
import time

from stable_baselines3 import PPO
from simulation import BaseDroneEnv, get_platform_config, create_clear_day


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--platform", type=str, default="quadcopter_basic")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("Agent Evaluation")
    print("=" * 60)

    # Create environment
    env = BaseDroneEnv(
        platform_config=get_platform_config(args.platform),
        environmental_conditions=create_clear_day(),
        render_mode=None if args.no_render else "human"
    )

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = PPO.load(args.model)

    # Run evaluation episodes
    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("-" * 40)

    total_rewards = []
    total_steps = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        for step in range(args.max_steps):
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Small delay for visualization
            if not args.no_render:
                time.sleep(0.01)

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_steps.append(steps)
        print(f"Episode {episode + 1}: reward = {episode_reward:.2f}, steps = {steps}")

    # Summary
    print("-" * 40)
    print(f"\nSummary ({args.episodes} episodes):")
    print(f"  Mean reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"  Mean steps: {sum(total_steps) / len(total_steps):.1f}")
    print(f"  Best reward: {max(total_rewards):.2f}")
    print(f"  Worst reward: {min(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()

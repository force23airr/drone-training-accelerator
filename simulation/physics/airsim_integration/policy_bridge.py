"""
Policy Deployment Bridge

Deploys trained policies from PyBullet to AirSim for realistic visualization.

Workflow:
1. Train in PyBullet (fast, headless)
2. Save policy checkpoint
3. Load policy in this bridge
4. Visualize/test in AirSim (photorealistic)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable
import time

from simulation.physics.airsim_integration.airsim_environment import (
    AirSimDroneEnv,
    RealisticEnvironmentConfig,
    PRESET_ENVIRONMENTS,
)


class PolicyDeploymentBridge:
    """
    Bridge for deploying trained RL policies to AirSim.

    Handles observation/action space mapping between
    PyBullet training environment and AirSim visualization.
    """

    def __init__(
        self,
        model_path: str,
        environment_config: Optional[RealisticEnvironmentConfig] = None,
        algorithm: str = "PPO",
    ):
        """
        Initialize policy deployment bridge.

        Args:
            model_path: Path to trained model (.zip file from SB3)
            environment_config: AirSim environment configuration
            algorithm: RL algorithm used (PPO, SAC, TD3)
        """
        self.model_path = Path(model_path)
        self.env_config = environment_config or RealisticEnvironmentConfig.florida_street()
        self.algorithm = algorithm.upper()

        self.model = None
        self.env = None
        self._loaded = False

    def load(self) -> None:
        """Load the trained model and initialize AirSim environment."""
        print(f"Loading policy from: {self.model_path}")

        # Load Stable-Baselines3 model
        if self.algorithm == "PPO":
            from stable_baselines3 import PPO
            self.model = PPO.load(self.model_path)
        elif self.algorithm == "SAC":
            from stable_baselines3 import SAC
            self.model = SAC.load(self.model_path)
        elif self.algorithm == "TD3":
            from stable_baselines3 import TD3
            self.model = TD3.load(self.model_path)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        print(f"Model loaded successfully!")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Observation space: {self.model.observation_space}")
        print(f"  Action space: {self.model.action_space}")

        # Initialize AirSim environment
        print(f"\nInitializing AirSim environment: {self.env_config.name}")
        self.env = AirSimDroneEnv(
            environment_config=self.env_config,
            render_mode="human"
        )

        self._loaded = True
        print("Ready for deployment!")

    def run_episode(
        self,
        max_steps: int = 1000,
        deterministic: bool = True,
        delay: float = 0.02,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single episode with the trained policy.

        Args:
            max_steps: Maximum steps per episode
            deterministic: Use deterministic actions (vs stochastic)
            delay: Delay between steps for visualization
            callback: Optional callback(step, obs, reward) called each step

        Returns:
            Episode statistics
        """
        if not self._loaded:
            self.load()

        obs, info = self.env.reset()
        total_reward = 0.0
        steps = 0

        print(f"\nRunning episode in: {self.env_config.name}")
        print(f"  Max steps: {max_steps}")
        print(f"  Deterministic: {deterministic}")
        print("-" * 40)

        for step in range(max_steps):
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=deterministic)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1

            # Optional callback
            if callback:
                callback(step, obs, reward)

            # Print progress periodically
            if step % 100 == 0:
                pos = info.get('position', [0, 0, 0])
                print(f"  Step {step}: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], reward={reward:.2f}")

            # Visualization delay
            if delay > 0:
                time.sleep(delay)

            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                if info.get('collision'):
                    print("  Reason: Collision")
                break

        # Episode summary
        results = {
            "total_reward": total_reward,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            "collision": info.get('collision', False),
        }

        print("-" * 40)
        print(f"Episode complete!")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")

        return results

    def run_evaluation(
        self,
        num_episodes: int = 5,
        max_steps_per_episode: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run multiple evaluation episodes.

        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Max steps per episode

        Returns:
            Evaluation statistics
        """
        if not self._loaded:
            self.load()

        all_rewards = []
        all_steps = []
        collisions = 0

        print(f"\n{'='*60}")
        print(f"Running evaluation: {num_episodes} episodes")
        print(f"Environment: {self.env_config.name}")
        print(f"{'='*60}")

        for ep in range(num_episodes):
            print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
            results = self.run_episode(max_steps=max_steps_per_episode)

            all_rewards.append(results["total_reward"])
            all_steps.append(results["steps"])
            if results["collision"]:
                collisions += 1

        # Summary statistics
        summary = {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_steps": np.mean(all_steps),
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "collision_rate": collisions / num_episodes,
        }

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Mean reward: {summary['mean_reward']:.2f} +/- {summary['std_reward']:.2f}")
        print(f"Mean steps: {summary['mean_steps']:.1f}")
        print(f"Best reward: {summary['max_reward']:.2f}")
        print(f"Worst reward: {summary['min_reward']:.2f}")
        print(f"Collision rate: {summary['collision_rate']*100:.1f}%")

        return summary

    def close(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()
        self._loaded = False


def deploy_to_airsim(
    model_path: str,
    environment: str = "florida_street",
    num_episodes: int = 3,
    algorithm: str = "PPO",
) -> Dict[str, Any]:
    """
    Quick function to deploy a trained policy to AirSim.

    Args:
        model_path: Path to trained model
        environment: Environment preset name
        num_episodes: Number of episodes to run
        algorithm: RL algorithm used

    Returns:
        Evaluation results
    """
    env_config = PRESET_ENVIRONMENTS.get(environment)
    if env_config is None:
        raise ValueError(f"Unknown environment: {environment}")

    bridge = PolicyDeploymentBridge(
        model_path=model_path,
        environment_config=env_config,
        algorithm=algorithm,
    )

    try:
        results = bridge.run_evaluation(num_episodes=num_episodes)
        return results
    finally:
        bridge.close()


def visualize_policy(
    model_path: str,
    environment: str = "florida_street",
    max_steps: int = 500,
    algorithm: str = "PPO",
) -> None:
    """
    Quick function to visualize a single episode.

    Args:
        model_path: Path to trained model
        environment: Environment preset name
        max_steps: Maximum steps to run
        algorithm: RL algorithm used
    """
    env_config = PRESET_ENVIRONMENTS.get(environment)
    if env_config is None:
        raise ValueError(f"Unknown environment: {environment}")

    bridge = PolicyDeploymentBridge(
        model_path=model_path,
        environment_config=env_config,
        algorithm=algorithm,
    )

    try:
        bridge.load()
        bridge.run_episode(max_steps=max_steps, delay=0.05)
    finally:
        bridge.close()


def record_video(
    model_path: str,
    output_path: str,
    environment: str = "florida_street",
    max_steps: int = 500,
    fps: int = 30,
    algorithm: str = "PPO",
) -> str:
    """
    Record a video of the policy running in AirSim.

    Args:
        model_path: Path to trained model
        output_path: Output video file path
        environment: Environment preset name
        max_steps: Maximum steps to record
        fps: Video frame rate
        algorithm: RL algorithm used

    Returns:
        Path to saved video
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for video recording")

    env_config = PRESET_ENVIRONMENTS.get(environment)
    if env_config is None:
        raise ValueError(f"Unknown environment: {environment}")

    bridge = PolicyDeploymentBridge(
        model_path=model_path,
        environment_config=env_config,
        algorithm=algorithm,
    )

    frames = []

    def capture_frame(step, obs, reward):
        """Callback to capture frames."""
        frame = bridge.env.get_camera_view()
        frames.append(frame)

    try:
        bridge.load()
        bridge.run_episode(
            max_steps=max_steps,
            delay=1/fps,
            callback=capture_frame
        )
    finally:
        bridge.close()

    # Save video
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"Video saved to: {output_path}")
        return output_path

    return ""


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy trained policy to AirSim")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--environment", type=str, default="florida_street",
                       choices=list(PRESET_ENVIRONMENTS.keys()))
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC", "TD3"])
    parser.add_argument("--record", type=str, help="Record video to this path")

    args = parser.parse_args()

    if args.record:
        record_video(
            model_path=args.model,
            output_path=args.record,
            environment=args.environment,
            algorithm=args.algorithm,
        )
    else:
        deploy_to_airsim(
            model_path=args.model,
            environment=args.environment,
            num_episodes=args.episodes,
            algorithm=args.algorithm,
        )

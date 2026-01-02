"""
Generative Adversarial Imitation Learning (GAIL)

GAIL learns a reward function from demonstrations, then uses that
reward to train a policy via RL. This is powerful because:

1. It can learn complex, hard-to-specify rewards
2. The learned reward transfers to similar tasks
3. It doesn't require action labels (only state trajectories)

Architecture:
- Discriminator: Distinguishes expert vs. policy trajectories
- Generator (Policy): Tries to fool the discriminator
- The discriminator's output becomes the reward signal for RL

This enables learning from videos where we only have states, not actions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

from training.imitation.demonstration import (
    Demonstration,
    DemonstrationDataset,
)


class Discriminator(nn.Module):
    """
    GAIL Discriminator Network.

    Learns to distinguish between:
    - Expert (demonstration) state-action pairs
    - Policy (generated) state-action pairs

    The discriminator output is used as the reward signal:
    r(s, a) = -log(1 - D(s, a))

    This encourages the policy to produce trajectories that
    look like expert demonstrations.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256],
        use_spectral_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        input_dim = observation_dim + action_dim

        # Build network
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)

            # Spectral normalization for training stability
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Output layer (logits)
        output_layer = nn.Linear(prev_size, 1)
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        self.output = output_layer

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observations: (batch_size, observation_dim)
            actions: (batch_size, action_dim)

        Returns:
            logits: (batch_size, 1) - probability that input is from expert
        """
        x = torch.cat([observations, actions], dim=-1)
        features = self.features(x)
        logits = self.output(features)
        return logits

    def compute_reward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GAIL reward.

        r(s, a) = -log(1 - D(s, a))

        Higher reward when policy looks more like expert.
        """
        with torch.no_grad():
            logits = self.forward(observations, actions)
            prob = torch.sigmoid(logits)
            # Clamp for numerical stability
            prob = torch.clamp(prob, 0.01, 0.99)
            reward = -torch.log(1 - prob)
        return reward.squeeze(-1)

    def get_reward_numpy(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute reward from numpy arrays."""
        obs_tensor = torch.FloatTensor(observations)
        act_tensor = torch.FloatTensor(actions)

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            act_tensor = act_tensor.unsqueeze(0)

        reward = self.compute_reward(obs_tensor, act_tensor)
        return reward.cpu().numpy()


class StateOnlyDiscriminator(nn.Module):
    """
    State-only discriminator for learning from observations without actions.

    This is crucial for learning from videos where we only have
    pose estimates, not control inputs.

    Uses state transitions: D(s, s')
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: List[int] = [256, 256],
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        input_dim = observation_dim * 2  # (s, s')

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)

        output_layer = nn.Linear(prev_size, 1)
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        self.output = output_layer

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass on state transitions."""
        x = torch.cat([state, next_state], dim=-1)
        features = self.features(x)
        return self.output(features)

    def compute_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward from state transition."""
        with torch.no_grad():
            logits = self.forward(state, next_state)
            prob = torch.sigmoid(logits)
            prob = torch.clamp(prob, 0.01, 0.99)
            reward = -torch.log(1 - prob)
        return reward.squeeze(-1)


class GAILRewardWrapper:
    """
    Environment wrapper that replaces native rewards with GAIL discriminator rewards.

    This is the key to making GAIL work - the policy learns to maximize
    the discriminator's reward signal, not the environment's native reward.
    """

    def __init__(self, env, discriminator, state_only: bool = False):
        """
        Args:
            env: The environment to wrap
            discriminator: GAIL discriminator (Discriminator or StateOnlyDiscriminator)
            state_only: Whether discriminator uses state transitions only
        """
        self.env = env
        self.discriminator = discriminator
        self.state_only = state_only
        self._last_obs = None

        # Forward gym attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, 'metadata', {})
        self.spec = getattr(env, 'spec', None)

    def reset(self, **kwargs):
        """Reset environment and store initial observation."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        self._last_obs = obs
        return obs, info

    def step(self, action):
        """Step environment and replace reward with GAIL reward."""
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Compute GAIL reward from discriminator
        gail_reward = self._compute_gail_reward(self._last_obs, action, obs)

        # Store original reward for logging/debugging
        info['env_reward'] = env_reward
        info['gail_reward'] = gail_reward

        # Update last observation for next step
        self._last_obs = obs

        return obs, gail_reward, terminated, truncated, info

    def _compute_gail_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
    ) -> float:
        """Compute GAIL reward from discriminator."""
        self.discriminator.eval()

        # Get device from discriminator
        device = next(self.discriminator.parameters()).device

        # Prepare tensors
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)

        with torch.no_grad():
            if self.state_only:
                reward = self.discriminator.compute_reward(obs_t, next_obs_t)
            else:
                action_t = torch.FloatTensor(action).unsqueeze(0).to(device)
                reward = self.discriminator.compute_reward(obs_t, action_t)

        return float(reward.cpu().numpy()[0])

    def render(self, *args, **kwargs):
        """Forward render calls to wrapped environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Close the wrapped environment."""
        return self.env.close()

    def __getattr__(self, name):
        """Forward unknown attributes to wrapped environment."""
        return getattr(self.env, name)


class GAIL:
    """
    Generative Adversarial Imitation Learning trainer.

    Combines:
    1. Discriminator training (adversarial)
    2. Policy training (RL with learned reward)

    This learns both "what good behavior looks like" and
    "how to achieve it".
    """

    def __init__(
        self,
        env,
        expert_dataset: DemonstrationDataset,
        hidden_sizes: List[int] = [256, 256],
        discriminator_lr: float = 3e-4,
        policy_lr: float = 3e-4,
        n_discriminator_updates: int = 3,
        batch_size: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        device: str = "auto",
        state_only: bool = False,
    ):
        """
        Args:
            env: Gymnasium environment
            expert_dataset: Expert demonstrations
            hidden_sizes: Hidden layer sizes for discriminator
            discriminator_lr: Discriminator learning rate
            policy_lr: Policy learning rate (for internal policy)
            n_discriminator_updates: Discriminator updates per policy update
            batch_size: Batch size for training
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clip range
            entropy_coef: Entropy bonus coefficient
            device: Device to use
            state_only: Use state-only discriminator (for video learning)
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.env = env  # Original environment (for rollout collection)
        self.expert_dataset = expert_dataset
        self.batch_size = batch_size
        self.n_discriminator_updates = n_discriminator_updates
        self.gamma = gamma
        self.state_only = state_only

        # Get dimensions
        obs_dim = expert_dataset.observation_dim
        act_dim = expert_dataset.action_dim

        # Create discriminator
        if state_only:
            self.discriminator = StateOnlyDiscriminator(
                observation_dim=obs_dim,
                hidden_sizes=hidden_sizes,
            ).to(self.device)
        else:
            self.discriminator = Discriminator(
                observation_dim=obs_dim,
                action_dim=act_dim,
                hidden_sizes=hidden_sizes,
            ).to(self.device)

        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
        )

        # Create wrapped environment that uses GAIL rewards
        # This is the key fix - PPO will now train on discriminator rewards
        self.wrapped_env = GAILRewardWrapper(
            env=env,
            discriminator=self.discriminator,
            state_only=state_only,
        )

        # Create policy (using PPO from stable-baselines3)
        self._init_policy(policy_lr, clip_range, entropy_coef)

        # Prepare expert data
        self._prepare_expert_data()

        # Training statistics
        self.discriminator_losses: List[float] = []
        self.policy_rewards: List[float] = []
        self.expert_accuracy: List[float] = []
        self.policy_accuracy: List[float] = []

    def _init_policy(self, lr: float, clip_range: float, entropy_coef: float):
        """Initialize the policy using PPO with wrapped environment."""
        from stable_baselines3 import PPO

        # Use wrapped environment so PPO trains on GAIL rewards
        self.policy_model = PPO(
            "MlpPolicy",
            self.wrapped_env,  # Key change: use wrapped env with GAIL rewards
            learning_rate=lr,
            clip_range=clip_range,
            ent_coef=entropy_coef,
            gamma=self.gamma,
            verbose=0,
            device=self.device,
        )

    def _prepare_expert_data(self):
        """Prepare expert data for training."""
        if self.state_only:
            # Get transitions (s, s')
            obs, _, next_obs, _ = self.expert_dataset.get_all_transitions()
            self.expert_states = torch.FloatTensor(obs).to(self.device)
            self.expert_next_states = torch.FloatTensor(next_obs).to(self.device)
        else:
            # Get (s, a) pairs
            self.expert_obs = torch.FloatTensor(
                self.expert_dataset.get_all_observations()
            ).to(self.device)
            self.expert_acts = torch.FloatTensor(
                self.expert_dataset.get_all_actions()
            ).to(self.device)

    def sample_expert_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of expert data."""
        if self.state_only:
            indices = np.random.choice(len(self.expert_states), batch_size, replace=False)
            return self.expert_states[indices], self.expert_next_states[indices]
        else:
            indices = np.random.choice(len(self.expert_obs), batch_size, replace=False)
            return self.expert_obs[indices], self.expert_acts[indices]

    def collect_policy_rollout(
        self,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect rollout from current policy.

        Returns:
            observations, actions, next_observations
        """
        observations = []
        actions = []
        next_observations = []

        obs, _ = self.env.reset()

        for _ in range(n_steps):
            action, _ = self.policy_model.predict(obs, deterministic=False)

            observations.append(obs)
            actions.append(action)

            obs, _, terminated, truncated, _ = self.env.step(action)
            next_observations.append(obs)

            if terminated or truncated:
                obs, _ = self.env.reset()

        return (
            np.array(observations),
            np.array(actions),
            np.array(next_observations),
        )

    def update_discriminator(
        self,
        policy_obs: np.ndarray,
        policy_acts: np.ndarray,
        policy_next_obs: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update discriminator to distinguish expert from policy.

        Uses binary cross-entropy:
        - Expert samples labeled as 1
        - Policy samples labeled as 0
        """
        self.discriminator.train()

        # Sample expert batch
        batch_size = min(self.batch_size, len(policy_obs))

        if self.state_only:
            expert_s, expert_ns = self.sample_expert_batch(batch_size)
        else:
            expert_obs, expert_acts = self.sample_expert_batch(batch_size)

        # Convert policy data to tensors
        policy_obs_t = torch.FloatTensor(policy_obs[:batch_size]).to(self.device)
        policy_acts_t = torch.FloatTensor(policy_acts[:batch_size]).to(self.device)
        policy_next_obs_t = torch.FloatTensor(policy_next_obs[:batch_size]).to(self.device)

        # Discriminator loss
        loss_fn = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        expert_acc = 0.0
        policy_acc = 0.0

        for _ in range(self.n_discriminator_updates):
            self.discriminator_optimizer.zero_grad()

            # Expert forward pass
            if self.state_only:
                expert_logits = self.discriminator(expert_s, expert_ns)
                policy_logits = self.discriminator(policy_obs_t, policy_next_obs_t)
            else:
                expert_logits = self.discriminator(expert_obs, expert_acts)
                policy_logits = self.discriminator(policy_obs_t, policy_acts_t)

            # Labels: 1 for expert, 0 for policy
            expert_labels = torch.ones_like(expert_logits)
            policy_labels = torch.zeros_like(policy_logits)

            # Combined loss
            loss = loss_fn(expert_logits, expert_labels) + loss_fn(policy_logits, policy_labels)
            loss.backward()
            self.discriminator_optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                expert_preds = (torch.sigmoid(expert_logits) > 0.5).float()
                policy_preds = (torch.sigmoid(policy_logits) > 0.5).float()
                expert_acc += (expert_preds == expert_labels).float().mean().item()
                policy_acc += (policy_preds == policy_labels).float().mean().item()

        n_updates = self.n_discriminator_updates
        return {
            'discriminator_loss': total_loss / n_updates,
            'expert_accuracy': expert_acc / n_updates,
            'policy_accuracy': policy_acc / n_updates,
        }

    def compute_gail_rewards(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
    ) -> np.ndarray:
        """Compute GAIL rewards for a batch of transitions."""
        self.discriminator.eval()

        obs_t = torch.FloatTensor(observations).to(self.device)
        acts_t = torch.FloatTensor(actions).to(self.device)
        next_obs_t = torch.FloatTensor(next_observations).to(self.device)

        with torch.no_grad():
            if self.state_only:
                rewards = self.discriminator.compute_reward(obs_t, next_obs_t)
            else:
                rewards = self.discriminator.compute_reward(obs_t, acts_t)

        return rewards.cpu().numpy()

    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 2048,
        n_epochs: int = 10,
        verbose: bool = True,
        callback=None,
    ) -> Dict[str, Any]:
        """
        Train GAIL.

        Training loop alternates between:
        1. Collecting policy rollouts (for discriminator training)
        2. Updating discriminator to distinguish expert vs policy
        3. Updating policy via PPO using GAIL rewards from wrapped environment

        The wrapped environment (self.wrapped_env) automatically replaces
        environment rewards with discriminator-computed GAIL rewards during
        PPO's internal rollout collection.

        Args:
            total_timesteps: Total environment steps
            rollout_steps: Steps per rollout for discriminator training
            n_epochs: PPO epochs per update (not used - PPO uses its defaults)
            verbose: Print progress
            callback: Optional callback(iteration, disc_stats, mean_reward)

        Returns:
            Training history dict with timesteps, losses, rewards, accuracies
        """
        if verbose:
            print("=" * 60)
            print("GAIL TRAINING")
            print("=" * 60)
            print(f"  Expert demos: {len(self.expert_dataset)}")
            print(f"  Total timesteps: {total_timesteps}")
            print(f"  Device: {self.device}")
            print(f"  State-only mode: {self.state_only}")
            print(f"  Reward injection: Enabled (via GAILRewardWrapper)")

        history = {
            'timesteps': [],
            'discriminator_losses': [],
            'mean_rewards': [],
            'expert_accuracy': [],
            'policy_accuracy': [],
        }

        timesteps_done = 0
        iteration = 0

        while timesteps_done < total_timesteps:
            iteration += 1

            # Step 1: Collect rollout from current policy for discriminator training
            # This uses the original environment to get (s, a, s') tuples
            obs, acts, next_obs = self.collect_policy_rollout(rollout_steps)
            timesteps_done += rollout_steps

            # Step 2: Update discriminator to better distinguish expert vs policy
            disc_stats = self.update_discriminator(obs, acts, next_obs)

            # Compute GAIL rewards for logging/monitoring
            # (The actual reward injection happens in GAILRewardWrapper during PPO's rollout)
            gail_rewards = self.compute_gail_rewards(obs, acts, next_obs)
            mean_reward = np.mean(gail_rewards)

            # Store statistics
            history['timesteps'].append(timesteps_done)
            history['discriminator_losses'].append(disc_stats['discriminator_loss'])
            history['mean_rewards'].append(mean_reward)
            history['expert_accuracy'].append(disc_stats['expert_accuracy'])
            history['policy_accuracy'].append(disc_stats['policy_accuracy'])

            # Step 3: Update policy using GAIL rewards
            # PPO collects its own rollouts using self.wrapped_env, which
            # automatically computes GAIL rewards via the discriminator
            self.policy_model.learn(
                total_timesteps=rollout_steps,
                reset_num_timesteps=False,
            )

            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration}: steps={timesteps_done}, "
                      f"disc_loss={disc_stats['discriminator_loss']:.4f}, "
                      f"mean_gail_reward={mean_reward:.4f}, "
                      f"expert_acc={disc_stats['expert_accuracy']:.2f}, "
                      f"policy_acc={disc_stats['policy_accuracy']:.2f}")

            if callback:
                callback(iteration, disc_stats, mean_reward)

        if verbose:
            print("=" * 60)
            print("GAIL training complete!")
            print(f"  Final discriminator loss: {history['discriminator_losses'][-1]:.4f}")
            print(f"  Final mean GAIL reward: {history['mean_rewards'][-1]:.4f}")
            print("=" * 60)

        return history

    def get_policy(self):
        """Get the trained policy."""
        return self.policy_model

    def get_reward_function(self):
        """Get the learned reward function (discriminator)."""
        return self.discriminator

    def save(self, path: str):
        """Save GAIL model."""
        save_dict = {
            'discriminator_state_dict': self.discriminator.state_dict(),
            'state_only': self.state_only,
        }
        torch.save(save_dict, path)
        self.policy_model.save(path.replace('.pt', '_policy.zip'))
        print(f"Saved GAIL model to {path}")

    def load(self, path: str):
        """Load GAIL model."""
        from stable_baselines3 import PPO

        save_dict = torch.load(path, weights_only=False)
        self.discriminator.load_state_dict(save_dict['discriminator_state_dict'])
        self.policy_model = PPO.load(path.replace('.pt', '_policy.zip'))
        print(f"Loaded GAIL model from {path}")


def train_gail(
    env,
    expert_dataset: DemonstrationDataset,
    total_timesteps: int = 100000,
    hidden_sizes: List[int] = [256, 256],
    state_only: bool = False,
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[GAIL, Dict[str, Any]]:
    """
    Convenience function to train GAIL.

    Args:
        env: Gymnasium environment
        expert_dataset: Expert demonstrations
        total_timesteps: Training timesteps
        hidden_sizes: Hidden layer sizes
        state_only: Use state-only discriminator
        device: Device to use
        verbose: Print progress

    Returns:
        Trained GAIL instance and training history
    """
    gail = GAIL(
        env=env,
        expert_dataset=expert_dataset,
        hidden_sizes=hidden_sizes,
        state_only=state_only,
        device=device,
    )

    history = gail.train(
        total_timesteps=total_timesteps,
        verbose=verbose,
    )

    return gail, history


class AIRL:
    """
    Adversarial Inverse Reinforcement Learning.

    AIRL learns a disentangled reward function that:
    1. Separates reward into state-dependent and shaping terms
    2. Is more robust to changes in dynamics
    3. Transfers better to new environments

    The reward structure is:
    r(s, a, s') = f(s, a, s') - log(pi(a|s))

    Where f is decomposed as:
    f(s, a, s') = g(s) + gamma * h(s') - h(s)

    g(s) is the true reward, h(s) is shaping.
    """

    def __init__(
        self,
        env,
        expert_dataset: DemonstrationDataset,
        hidden_sizes: List[int] = [256, 256],
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.env = env
        self.expert_dataset = expert_dataset

        obs_dim = expert_dataset.observation_dim

        # Reward network g(s) - the "true" reward
        self.reward_net = self._build_network(obs_dim, 1, hidden_sizes).to(self.device)

        # Shaping network h(s)
        self.shaping_net = self._build_network(obs_dim, 1, hidden_sizes).to(self.device)

        self.optimizer = optim.Adam(
            list(self.reward_net.parameters()) + list(self.shaping_net.parameters()),
            lr=3e-4,
        )

    def _build_network(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
    ) -> nn.Module:
        """Build a simple MLP."""
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        return nn.Sequential(*layers)

    def compute_reward(self, state: np.ndarray) -> np.ndarray:
        """Compute learned reward for a state."""
        self.reward_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            reward = self.reward_net(state_t)
        return reward.cpu().numpy().squeeze()

    def get_transferable_reward(self) -> nn.Module:
        """
        Get the learned reward function.

        This is g(s) which should transfer to new dynamics.
        """
        return self.reward_net

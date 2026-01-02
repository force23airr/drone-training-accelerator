"""
Behavioral Metrics Collector

Measures behavioral similarity to expert demonstrations:
- Distance to expert state distribution
- Action similarity to expert actions
- Trajectory similarity
"""

import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from evaluation.metrics.base_metrics import MetricCollector, MetricResult

if TYPE_CHECKING:
    from training.imitation.demonstration import DemonstrationDataset


class BehavioralMetrics(MetricCollector):
    """
    Measures behavioral similarity to expert demonstrations.

    Uses KD-tree for efficient nearest-neighbor queries to measure
    how closely the policy's behavior matches expert demonstrations.

    Tracks:
    - Mean/max distance to nearest expert state
    - Mean/max distance to corresponding expert action
    - Trajectory divergence
    - Expert coverage (fraction of expert states visited)
    """

    def __init__(
        self,
        expert_observations: Optional[np.ndarray] = None,
        expert_actions: Optional[np.ndarray] = None,
        state_weight: float = 1.0,
        action_weight: float = 1.0,
        use_kdtree: bool = True,
    ):
        """
        Args:
            expert_observations: Expert observation data (N, obs_dim)
            expert_actions: Expert action data (N, action_dim)
            state_weight: Weight for state distance in combined score
            action_weight: Weight for action distance in combined score
            use_kdtree: Use KD-tree for efficient nearest neighbor search
        """
        super().__init__(name="BehavioralMetrics")

        self.expert_obs = expert_observations
        self.expert_acts = expert_actions
        self.state_weight = state_weight
        self.action_weight = action_weight
        self.use_kdtree = use_kdtree

        self._obs_tree = None
        self._state_distances: List[float] = []
        self._action_distances: List[float] = []
        self._nearest_indices: List[int] = []

        # Build KD-tree if data provided
        if expert_observations is not None and use_kdtree:
            self._build_kdtree()

    def _build_kdtree(self) -> None:
        """Build KD-tree for efficient nearest neighbor search."""
        try:
            from scipy.spatial import cKDTree
            self._obs_tree = cKDTree(self.expert_obs)
        except ImportError:
            print("Warning: scipy not available, using brute-force search")
            self._obs_tree = None

    @classmethod
    def from_dataset(
        cls,
        dataset: 'DemonstrationDataset',
        state_weight: float = 1.0,
        action_weight: float = 1.0,
    ) -> 'BehavioralMetrics':
        """
        Create BehavioralMetrics from a DemonstrationDataset.

        Args:
            dataset: Expert demonstration dataset
            state_weight: Weight for state distance
            action_weight: Weight for action distance

        Returns:
            BehavioralMetrics instance
        """
        return cls(
            expert_observations=dataset.get_all_observations(),
            expert_actions=dataset.get_all_actions(),
            state_weight=state_weight,
            action_weight=action_weight,
        )

    def reset(self) -> None:
        super().reset()
        self._state_distances = []
        self._action_distances = []
        self._nearest_indices = []

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        super().step(obs, action, reward, next_obs, terminated, truncated, info)

        if self.expert_obs is None:
            return

        # Find nearest expert state
        state_dist, nearest_idx = self._find_nearest(obs)
        self._state_distances.append(state_dist)
        self._nearest_indices.append(nearest_idx)

        # Compute action distance to nearest expert action
        if self.expert_acts is not None:
            expert_action = self.expert_acts[nearest_idx]
            action_dist = np.linalg.norm(action - expert_action)
            self._action_distances.append(action_dist)

    def _find_nearest(self, obs: np.ndarray) -> tuple:
        """Find nearest expert observation."""
        if self._obs_tree is not None:
            dist, idx = self._obs_tree.query(obs)
            return float(dist), int(idx)
        else:
            # Brute force search
            distances = np.linalg.norm(self.expert_obs - obs, axis=1)
            idx = np.argmin(distances)
            return float(distances[idx]), int(idx)

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        results = []

        if not self._state_distances:
            # No expert data available
            results.extend([
                MetricResult('mean_state_distance', 0.0, 'euclidean', higher_is_better=False),
                MetricResult('max_state_distance', 0.0, 'euclidean', higher_is_better=False),
                MetricResult('mean_action_distance', 0.0, 'euclidean', higher_is_better=False),
                MetricResult('expert_similarity_score', 1.0, 'normalized', higher_is_better=True),
            ])
            return results

        state_dists = np.array(self._state_distances)

        # State distance statistics
        results.extend([
            MetricResult('mean_state_distance', float(np.mean(state_dists)), 'euclidean', higher_is_better=False),
            MetricResult('max_state_distance', float(np.max(state_dists)), 'euclidean', higher_is_better=False),
            MetricResult('min_state_distance', float(np.min(state_dists)), 'euclidean', higher_is_better=False),
            MetricResult('state_distance_std', float(np.std(state_dists)), 'euclidean', higher_is_better=False),
            MetricResult('state_distance_percentile_95', float(np.percentile(state_dists, 95)), 'euclidean', higher_is_better=False),
        ])

        # Action distance statistics
        if self._action_distances:
            action_dists = np.array(self._action_distances)
            results.extend([
                MetricResult('mean_action_distance', float(np.mean(action_dists)), 'euclidean', higher_is_better=False),
                MetricResult('max_action_distance', float(np.max(action_dists)), 'euclidean', higher_is_better=False),
                MetricResult('action_distance_std', float(np.std(action_dists)), 'euclidean', higher_is_better=False),
            ])
        else:
            results.extend([
                MetricResult('mean_action_distance', 0.0, 'euclidean', higher_is_better=False),
                MetricResult('max_action_distance', 0.0, 'euclidean', higher_is_better=False),
                MetricResult('action_distance_std', 0.0, 'euclidean', higher_is_better=False),
            ])

        # Expert coverage (unique expert states visited)
        if self.expert_obs is not None:
            unique_visited = len(set(self._nearest_indices))
            coverage = unique_visited / len(self.expert_obs)
            results.append(
                MetricResult('expert_coverage', coverage, 'ratio', higher_is_better=True)
            )
        else:
            results.append(
                MetricResult('expert_coverage', 0.0, 'ratio', higher_is_better=True)
            )

        # Combined similarity score
        similarity_score = self._compute_similarity_score()
        results.append(
            MetricResult(
                'expert_similarity_score',
                similarity_score,
                'normalized',
                higher_is_better=True,
                metadata={
                    'state_weight': self.state_weight,
                    'action_weight': self.action_weight,
                }
            )
        )

        # Trajectory divergence (cumulative distance over time)
        cumulative_dist = np.cumsum(state_dists)
        final_divergence = cumulative_dist[-1] / len(state_dists) if len(state_dists) > 0 else 0.0
        results.append(
            MetricResult('trajectory_divergence', float(final_divergence), 'euclidean', higher_is_better=False)
        )

        return results

    def _compute_similarity_score(self) -> float:
        """
        Compute overall similarity score (0-1, higher is more similar).

        Uses inverse distance normalized by typical scales.
        """
        if not self._state_distances:
            return 1.0

        # State similarity
        mean_state_dist = np.mean(self._state_distances)
        state_sim = 1.0 / (1.0 + mean_state_dist)

        # Action similarity
        if self._action_distances:
            mean_action_dist = np.mean(self._action_distances)
            action_sim = 1.0 / (1.0 + mean_action_dist)
        else:
            action_sim = 1.0

        # Weighted combination
        total_weight = self.state_weight + self.action_weight
        score = (self.state_weight * state_sim + self.action_weight * action_sim) / total_weight

        return float(score)

    def get_running_metrics(self) -> Dict[str, float]:
        metrics = {}

        if self._state_distances:
            metrics['current_state_distance'] = self._state_distances[-1]
            metrics['mean_state_distance'] = float(np.mean(self._state_distances[-100:]))

        if self._action_distances:
            metrics['mean_action_distance'] = float(np.mean(self._action_distances[-100:]))

        return metrics


class DistributionMetrics(MetricCollector):
    """
    Measures distributional similarity between policy and expert.

    Uses statistical distances like KL divergence and Wasserstein distance
    to compare state/action distributions.
    """

    def __init__(
        self,
        expert_observations: Optional[np.ndarray] = None,
        expert_actions: Optional[np.ndarray] = None,
        n_bins: int = 50,
    ):
        super().__init__(name="DistributionMetrics")

        self.expert_obs = expert_observations
        self.expert_acts = expert_actions
        self.n_bins = n_bins

        self._policy_obs: List[np.ndarray] = []
        self._policy_acts: List[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self._policy_obs = []
        self._policy_acts = []

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        super().step(obs, action, reward, next_obs, terminated, truncated, info)
        self._policy_obs.append(obs.copy())
        self._policy_acts.append(action.copy())

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        results = []

        if not self._policy_obs or self.expert_obs is None:
            results.extend([
                MetricResult('state_wasserstein_distance', 0.0, 'distance', higher_is_better=False),
                MetricResult('action_wasserstein_distance', 0.0, 'distance', higher_is_better=False),
            ])
            return results

        policy_obs = np.array(self._policy_obs)
        policy_acts = np.array(self._policy_acts)

        # Compute Wasserstein distance for each dimension
        try:
            from scipy.stats import wasserstein_distance

            # State distribution distance (average over dimensions)
            state_dists = []
            for dim in range(min(policy_obs.shape[1], self.expert_obs.shape[1])):
                dist = wasserstein_distance(policy_obs[:, dim], self.expert_obs[:, dim])
                state_dists.append(dist)

            mean_state_dist = float(np.mean(state_dists))
            results.append(
                MetricResult('state_wasserstein_distance', mean_state_dist, 'distance', higher_is_better=False)
            )

            # Action distribution distance
            if self.expert_acts is not None:
                action_dists = []
                for dim in range(min(policy_acts.shape[1], self.expert_acts.shape[1])):
                    dist = wasserstein_distance(policy_acts[:, dim], self.expert_acts[:, dim])
                    action_dists.append(dist)

                mean_action_dist = float(np.mean(action_dists))
                results.append(
                    MetricResult('action_wasserstein_distance', mean_action_dist, 'distance', higher_is_better=False)
                )
            else:
                results.append(
                    MetricResult('action_wasserstein_distance', 0.0, 'distance', higher_is_better=False)
                )

        except ImportError:
            results.extend([
                MetricResult('state_wasserstein_distance', 0.0, 'distance', higher_is_better=False),
                MetricResult('action_wasserstein_distance', 0.0, 'distance', higher_is_better=False),
            ])

        return results

    def get_running_metrics(self) -> Dict[str, float]:
        return {'samples_collected': len(self._policy_obs)}

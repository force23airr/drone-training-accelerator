"""
Demonstration Quality Auto-Scorer

Automatically computes quality scores for demonstrations based on:
- Smoothness (inverse jerk)
- Success (task completion)
- Time efficiency
- Collision-free flight
- Control quality (saturation, smoothness)
- Signal quality (dropouts, timestamp consistency)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from training.imitation.demonstration import Demonstration, DemonstrationDataset


@dataclass
class QualityScoreBreakdown:
    """
    Detailed breakdown of quality score components.

    Each component is normalized to 0-1 where higher is better.
    """
    # Flight quality
    smoothness_score: float = 1.0       # Inverse jerk penalty
    control_smoothness: float = 1.0     # Action change smoothness
    saturation_score: float = 1.0       # Control saturation avoidance

    # Task performance
    success_score: float = 1.0          # Task completion
    efficiency_score: float = 1.0       # Time/path efficiency

    # Safety
    collision_score: float = 1.0        # Collision-free

    # Signal quality
    signal_continuity: float = 1.0      # No dropouts
    timestamp_consistency: float = 1.0  # Consistent timing

    # Overall
    overall_score: float = 1.0          # Weighted combination
    confidence: float = 1.0             # Confidence in the score

    # Metadata
    flags: List[str] = field(default_factory=list)  # Quality warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            'smoothness_score': self.smoothness_score,
            'control_smoothness': self.control_smoothness,
            'saturation_score': self.saturation_score,
            'success_score': self.success_score,
            'efficiency_score': self.efficiency_score,
            'collision_score': self.collision_score,
            'signal_continuity': self.signal_continuity,
            'timestamp_consistency': self.timestamp_consistency,
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'flags': self.flags,
        }


@dataclass
class ScorerConfig:
    """Configuration for quality scoring."""
    # Weights for overall score
    smoothness_weight: float = 0.15
    control_smoothness_weight: float = 0.10
    saturation_weight: float = 0.10
    success_weight: float = 0.25
    efficiency_weight: float = 0.10
    collision_weight: float = 0.20
    signal_continuity_weight: float = 0.05
    timestamp_weight: float = 0.05

    # Thresholds
    max_jerk_threshold: float = 50.0         # m/s^3, penalize above this
    max_action_rate_threshold: float = 5.0   # units/s, penalize rapid changes
    saturation_threshold: float = 0.95       # Action magnitude threshold
    max_missing_frame_ratio: float = 0.05    # 5% missing frames
    max_timestamp_cv: float = 0.5            # Coefficient of variation for dt

    # Expected ranges (for normalization)
    expected_min_duration: float = 1.0       # Minimum expected demo duration
    expected_max_jerk: float = 100.0         # For normalization


class DemonstrationQualityScorer:
    """
    Computes quality scores for demonstrations.

    Scores are used for:
    - Filtering low-quality demonstrations
    - Weighting demonstrations during training
    - Identifying data collection issues
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        """
        Args:
            config: Scoring configuration
        """
        self.config = config or ScorerConfig()

    def score_demonstration(
        self,
        demo: 'Demonstration',
    ) -> QualityScoreBreakdown:
        """
        Compute quality score for a single demonstration.

        Args:
            demo: Demonstration to score

        Returns:
            QualityScoreBreakdown with detailed scores
        """
        flags = []

        # Compute individual scores
        smoothness_score = self._compute_smoothness_score(demo, flags)
        control_smoothness = self._compute_control_smoothness(demo, flags)
        saturation_score = self._compute_saturation_score(demo, flags)
        success_score = self._compute_success_score(demo, flags)
        efficiency_score = self._compute_efficiency_score(demo, flags)
        collision_score = self._compute_collision_score(demo, flags)
        signal_continuity = self._compute_signal_continuity(demo, flags)
        timestamp_consistency = self._compute_timestamp_consistency(demo, flags)

        # Compute weighted overall score
        weights = [
            (smoothness_score, self.config.smoothness_weight),
            (control_smoothness, self.config.control_smoothness_weight),
            (saturation_score, self.config.saturation_weight),
            (success_score, self.config.success_weight),
            (efficiency_score, self.config.efficiency_weight),
            (collision_score, self.config.collision_weight),
            (signal_continuity, self.config.signal_continuity_weight),
            (timestamp_consistency, self.config.timestamp_weight),
        ]

        total_weight = sum(w for _, w in weights)
        overall_score = sum(s * w for s, w in weights) / total_weight

        # Compute confidence based on demo length and completeness
        confidence = self._compute_confidence(demo)

        return QualityScoreBreakdown(
            smoothness_score=smoothness_score,
            control_smoothness=control_smoothness,
            saturation_score=saturation_score,
            success_score=success_score,
            efficiency_score=efficiency_score,
            collision_score=collision_score,
            signal_continuity=signal_continuity,
            timestamp_consistency=timestamp_consistency,
            overall_score=overall_score,
            confidence=confidence,
            flags=flags,
        )

    def score_dataset(
        self,
        dataset: 'DemonstrationDataset',
        update_scores: bool = True,
    ) -> Dict[str, QualityScoreBreakdown]:
        """
        Score all demonstrations in a dataset.

        Args:
            dataset: Dataset to score
            update_scores: Whether to update demo.quality_score

        Returns:
            Dict mapping demo_id to score breakdown
        """
        scores = {}

        for demo in dataset.demonstrations:
            breakdown = self.score_demonstration(demo)
            scores[demo.demo_id] = breakdown

            if update_scores:
                demo.quality_score = breakdown.overall_score

        return scores

    def _compute_smoothness_score(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute smoothness score based on jerk."""
        if demo.num_steps < 3:
            flags.append("too_short_for_jerk")
            return 0.5  # Neutral score for insufficient data

        # Compute jerk from positions if available
        positions = demo.positions
        if positions is not None:
            dt = 1.0 / demo.sample_rate_hz
            velocities = np.diff(positions, axis=0) / dt
            accelerations = np.diff(velocities, axis=0) / dt
            jerks = np.diff(accelerations, axis=0) / dt

            jerk_magnitude = np.linalg.norm(jerks, axis=1)
            mean_jerk = np.mean(jerk_magnitude)
            max_jerk = np.max(jerk_magnitude)

            # Score based on mean jerk (lower is better)
            normalized_jerk = mean_jerk / self.config.expected_max_jerk
            score = max(0.0, 1.0 - normalized_jerk)

            if max_jerk > self.config.max_jerk_threshold:
                flags.append(f"high_jerk_{max_jerk:.1f}")

            return score

        # Fallback: compute from observations (velocity if available)
        obs = demo.observations
        if obs.shape[1] >= 6:  # Assume first 6 are position+velocity
            dt = 1.0 / demo.sample_rate_hz
            velocities = obs[:, 3:6]  # Assume velocity at indices 3-5
            accelerations = np.diff(velocities, axis=0) / dt
            jerks = np.diff(accelerations, axis=0) / dt

            jerk_magnitude = np.linalg.norm(jerks, axis=1)
            mean_jerk = np.mean(jerk_magnitude)

            normalized_jerk = mean_jerk / self.config.expected_max_jerk
            return max(0.0, 1.0 - normalized_jerk)

        # No position/velocity data - return neutral
        flags.append("no_position_data_for_smoothness")
        return 0.7

    def _compute_control_smoothness(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute control smoothness based on action rate of change."""
        if demo.num_steps < 2:
            return 0.5

        actions = demo.actions
        dt = 1.0 / demo.sample_rate_hz

        # Compute action rate of change
        action_diff = np.diff(actions, axis=0)
        action_rate = action_diff / dt

        # Compute RMS action rate
        rms_rate = np.sqrt(np.mean(action_rate ** 2))

        # Score (lower rate is better)
        normalized_rate = rms_rate / self.config.max_action_rate_threshold
        score = max(0.0, 1.0 - normalized_rate * 0.5)

        if rms_rate > self.config.max_action_rate_threshold:
            flags.append("high_action_rate")

        return score

    def _compute_saturation_score(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute score based on control saturation avoidance."""
        actions = demo.actions

        # Count saturated actions (near limits)
        action_magnitude = np.abs(actions)
        saturation_count = np.sum(action_magnitude > self.config.saturation_threshold)
        total_actions = actions.size

        saturation_ratio = saturation_count / total_actions if total_actions > 0 else 0

        # Score (lower saturation is better)
        score = 1.0 - saturation_ratio

        if saturation_ratio > 0.2:
            flags.append(f"high_saturation_{saturation_ratio:.1%}")

        return score

    def _compute_success_score(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute success score based on task completion."""
        if demo.success:
            return 1.0
        else:
            flags.append("task_failed")
            return 0.0

    def _compute_efficiency_score(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute efficiency score based on path/time efficiency."""
        # Check for time-to-goal in environment info
        if demo.steps and demo.steps[-1].info:
            info = demo.steps[-1].info
            if 'time_efficiency' in info:
                return float(info['time_efficiency'])

        # Fallback: score based on reasonable duration
        if demo.duration_seconds < self.config.expected_min_duration:
            flags.append("very_short_demo")
            return 0.5

        # Compute path efficiency if positions available
        positions = demo.positions
        if positions is not None and len(positions) >= 2:
            # Direct distance vs path length
            direct_distance = np.linalg.norm(positions[-1] - positions[0])
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

            if path_length > 0:
                efficiency = min(1.0, direct_distance / path_length)
                return efficiency

        # Default: assume good efficiency
        return 0.8

    def _compute_collision_score(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute collision score (1.0 if collision-free)."""
        # Check for collision info in steps
        collision_detected = False

        for step in demo.steps:
            if step.info:
                if step.info.get('collision', False):
                    collision_detected = True
                    break
                if step.info.get('crash', False):
                    collision_detected = True
                    break

        # Check environment info
        if demo.environment:
            if demo.environment.get('had_collision', False):
                collision_detected = True

        if collision_detected:
            flags.append("collision_detected")
            return 0.0

        return 1.0

    def _compute_signal_continuity(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute signal continuity score based on missing frames."""
        if demo.num_steps < 2:
            return 0.5

        expected_dt = 1.0 / demo.sample_rate_hz

        # Count gaps in timing
        gap_count = 0
        for i in range(len(demo.steps) - 1):
            actual_dt = demo.steps[i + 1].timestamp - demo.steps[i].timestamp
            if actual_dt > expected_dt * 2:  # Gap > 2x expected
                gap_count += 1

        gap_ratio = gap_count / (demo.num_steps - 1)

        if gap_ratio > self.config.max_missing_frame_ratio:
            flags.append(f"signal_gaps_{gap_ratio:.1%}")

        # Also check for NaN/inf values
        obs = demo.observations
        action = demo.actions
        nan_count = np.sum(~np.isfinite(obs)) + np.sum(~np.isfinite(action))

        if nan_count > 0:
            flags.append(f"nan_values_{nan_count}")
            return 0.0

        return max(0.0, 1.0 - gap_ratio / self.config.max_missing_frame_ratio)

    def _compute_timestamp_consistency(
        self,
        demo: 'Demonstration',
        flags: List[str],
    ) -> float:
        """Compute timestamp consistency score."""
        if demo.num_steps < 2:
            return 0.5

        # Compute dt values
        dts = []
        for i in range(len(demo.steps) - 1):
            dt = demo.steps[i + 1].timestamp - demo.steps[i].timestamp
            if dt > 0:
                dts.append(dt)

        if not dts:
            flags.append("no_valid_timestamps")
            return 0.0

        dts = np.array(dts)

        # Compute coefficient of variation
        mean_dt = np.mean(dts)
        std_dt = np.std(dts)

        if mean_dt > 0:
            cv = std_dt / mean_dt
        else:
            cv = 0

        if cv > self.config.max_timestamp_cv:
            flags.append(f"inconsistent_timestamps_cv_{cv:.2f}")

        # Score based on CV (lower is better)
        score = max(0.0, 1.0 - cv / self.config.max_timestamp_cv)

        return score

    def _compute_confidence(self, demo: 'Demonstration') -> float:
        """Compute confidence in the quality score."""
        confidence = 1.0

        # Lower confidence for short demos
        if demo.num_steps < 50:
            confidence *= 0.5 + (demo.num_steps / 100)

        # Lower confidence if position data missing
        if demo.positions is None:
            confidence *= 0.8

        # Lower confidence for unverified demos
        if not demo.verified:
            confidence *= 0.9

        return min(1.0, confidence)


def score_demonstrations(
    demos: 'DemonstrationDataset',
    config: Optional[ScorerConfig] = None,
) -> Tuple['DemonstrationDataset', Dict[str, QualityScoreBreakdown]]:
    """
    Convenience function to score all demonstrations in a dataset.

    Args:
        demos: Dataset to score
        config: Optional scorer configuration

    Returns:
        Tuple of (updated dataset, score breakdowns)
    """
    scorer = DemonstrationQualityScorer(config)
    breakdowns = scorer.score_dataset(demos, update_scores=True)
    return demos, breakdowns


def filter_by_quality(
    demos: 'DemonstrationDataset',
    min_score: float = 0.5,
    required_flags_absent: Optional[List[str]] = None,
    config: Optional[ScorerConfig] = None,
) -> 'DemonstrationDataset':
    """
    Filter demonstrations by quality score.

    Args:
        demos: Dataset to filter
        min_score: Minimum quality score
        required_flags_absent: Flags that must not be present
        config: Optional scorer configuration

    Returns:
        Filtered dataset
    """
    from training.imitation.demonstration import DemonstrationDataset

    scorer = DemonstrationQualityScorer(config)
    required_flags_absent = required_flags_absent or []

    filtered = []
    for demo in demos.demonstrations:
        breakdown = scorer.score_demonstration(demo)

        # Check score
        if breakdown.overall_score < min_score:
            continue

        # Check flags
        if any(flag in breakdown.flags for flag in required_flags_absent):
            continue

        demo.quality_score = breakdown.overall_score
        filtered.append(demo)

    return DemonstrationDataset(filtered)

"""
Quality Filters for Demonstrations

Rules-based filtering to remove low-quality demonstrations:
- Pose jitter detection
- Missing frame detection
- Timestamp consistency
- Value range validation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from training.imitation.demonstration import Demonstration, DemonstrationDataset


@dataclass
class FilterResult:
    """Result of applying a filter to a demonstration."""
    passed: bool
    filter_name: str
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.filter_name}: {self.reason}"


@dataclass
class FilterConfig:
    """Configuration for quality filters."""
    # Pose jitter thresholds
    max_pose_jitter_m: float = 0.1          # Max position jitter in meters
    max_velocity_jitter_m_s: float = 1.0    # Max velocity jitter in m/s

    # Missing frame thresholds
    max_missing_frame_ratio: float = 0.05   # Max 5% missing frames
    max_gap_duration_s: float = 0.5         # Max gap between frames

    # Timestamp thresholds
    max_timestamp_variance_pct: float = 50  # Max variance as % of mean dt

    # Value range thresholds
    max_position_magnitude: float = 1000.0  # Max position magnitude
    max_velocity_magnitude: float = 50.0    # Max velocity magnitude
    max_action_magnitude: float = 2.0       # Max action magnitude

    # Length thresholds
    min_steps: int = 10                     # Minimum steps
    min_duration_s: float = 0.5             # Minimum duration

    # Quality score threshold
    min_quality_score: float = 0.5          # Minimum quality score


class QualityFilter:
    """
    Applies quality filters to demonstrations.

    Uses rules-based filtering to identify and remove low-quality data.
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        self._filters: List[Callable] = [
            self._check_length,
            self._check_pose_jitter,
            self._check_missing_frames,
            self._check_timestamp_consistency,
            self._check_value_ranges,
            self._check_nan_values,
            self._check_quality_score,
        ]

    def add_filter(self, filter_fn: Callable[['Demonstration', FilterConfig], FilterResult]):
        """Add a custom filter function."""
        self._filters.append(filter_fn)

    def check_demonstration(
        self,
        demo: 'Demonstration',
    ) -> List[FilterResult]:
        """
        Check a demonstration against all filters.

        Args:
            demo: Demonstration to check

        Returns:
            List of FilterResults (one per filter)
        """
        results = []
        for filter_fn in self._filters:
            result = filter_fn(demo, self.config)
            results.append(result)
        return results

    def passes_all_filters(
        self,
        demo: 'Demonstration',
    ) -> bool:
        """Check if demonstration passes all filters."""
        results = self.check_demonstration(demo)
        return all(r.passed for r in results)

    def filter_dataset(
        self,
        dataset: 'DemonstrationDataset',
        verbose: bool = False,
    ) -> 'DemonstrationDataset':
        """
        Filter a dataset, keeping only passing demonstrations.

        Args:
            dataset: Dataset to filter
            verbose: Print filter statistics

        Returns:
            Filtered dataset
        """
        from training.imitation.demonstration import DemonstrationDataset

        passed = []
        failed_counts: Dict[str, int] = {}

        for demo in dataset.demonstrations:
            results = self.check_demonstration(demo)

            if all(r.passed for r in results):
                passed.append(demo)
            else:
                for r in results:
                    if not r.passed:
                        failed_counts[r.filter_name] = failed_counts.get(r.filter_name, 0) + 1

        if verbose:
            print(f"Filter results: {len(passed)}/{len(dataset.demonstrations)} passed")
            for filter_name, count in sorted(failed_counts.items(), key=lambda x: -x[1]):
                print(f"  {filter_name}: {count} failed")

        return DemonstrationDataset(passed)

    def _check_length(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check minimum length requirements."""
        if demo.num_steps < config.min_steps:
            return FilterResult(
                passed=False,
                filter_name="length",
                reason=f"Too few steps: {demo.num_steps} < {config.min_steps}",
                details={'num_steps': demo.num_steps},
            )

        if demo.duration_seconds < config.min_duration_s:
            return FilterResult(
                passed=False,
                filter_name="length",
                reason=f"Too short: {demo.duration_seconds:.2f}s < {config.min_duration_s}s",
                details={'duration': demo.duration_seconds},
            )

        return FilterResult(
            passed=True,
            filter_name="length",
            reason=f"OK: {demo.num_steps} steps, {demo.duration_seconds:.2f}s",
        )

    def _check_pose_jitter(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check for excessive pose jitter."""
        positions = demo.positions
        if positions is None:
            return FilterResult(
                passed=True,
                filter_name="pose_jitter",
                reason="No position data to check",
            )

        if len(positions) < 3:
            return FilterResult(
                passed=True,
                filter_name="pose_jitter",
                reason="Insufficient data for jitter check",
            )

        # Compute jitter as difference from smoothed trajectory
        dt = 1.0 / demo.sample_rate_hz

        # Simple smoothing: compare position to average of neighbors
        jitters = []
        for i in range(1, len(positions) - 1):
            smoothed = (positions[i-1] + positions[i+1]) / 2
            jitter = np.linalg.norm(positions[i] - smoothed)
            jitters.append(jitter)

        max_jitter = np.max(jitters)
        mean_jitter = np.mean(jitters)

        if max_jitter > config.max_pose_jitter_m:
            return FilterResult(
                passed=False,
                filter_name="pose_jitter",
                reason=f"High jitter: {max_jitter:.3f}m > {config.max_pose_jitter_m}m",
                details={'max_jitter': max_jitter, 'mean_jitter': mean_jitter},
            )

        return FilterResult(
            passed=True,
            filter_name="pose_jitter",
            reason=f"OK: max jitter {max_jitter:.3f}m",
            details={'max_jitter': max_jitter, 'mean_jitter': mean_jitter},
        )

    def _check_missing_frames(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check for missing frames (gaps in timing)."""
        if demo.num_steps < 2:
            return FilterResult(
                passed=True,
                filter_name="missing_frames",
                reason="Too short to check",
            )

        expected_dt = 1.0 / demo.sample_rate_hz

        gap_count = 0
        max_gap = 0.0
        total_gaps = 0

        for i in range(len(demo.steps) - 1):
            actual_dt = demo.steps[i + 1].timestamp - demo.steps[i].timestamp

            if actual_dt > expected_dt * 2:  # Gap > 2x expected
                gap_count += 1
                gap_duration = actual_dt - expected_dt
                max_gap = max(max_gap, gap_duration)
                total_gaps += int(round(actual_dt / expected_dt)) - 1

        gap_ratio = total_gaps / demo.num_steps if demo.num_steps > 0 else 0

        if gap_ratio > config.max_missing_frame_ratio:
            return FilterResult(
                passed=False,
                filter_name="missing_frames",
                reason=f"Too many gaps: {gap_ratio:.1%} > {config.max_missing_frame_ratio:.1%}",
                details={'gap_ratio': gap_ratio, 'gap_count': gap_count, 'max_gap': max_gap},
            )

        if max_gap > config.max_gap_duration_s:
            return FilterResult(
                passed=False,
                filter_name="missing_frames",
                reason=f"Large gap: {max_gap:.2f}s > {config.max_gap_duration_s}s",
                details={'gap_ratio': gap_ratio, 'gap_count': gap_count, 'max_gap': max_gap},
            )

        return FilterResult(
            passed=True,
            filter_name="missing_frames",
            reason=f"OK: {gap_count} gaps, max {max_gap:.2f}s",
            details={'gap_ratio': gap_ratio, 'gap_count': gap_count, 'max_gap': max_gap},
        )

    def _check_timestamp_consistency(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check for inconsistent timestamps."""
        if demo.num_steps < 2:
            return FilterResult(
                passed=True,
                filter_name="timestamp_consistency",
                reason="Too short to check",
            )

        dts = []
        for i in range(len(demo.steps) - 1):
            dt = demo.steps[i + 1].timestamp - demo.steps[i].timestamp
            if dt > 0:
                dts.append(dt)

        if not dts:
            return FilterResult(
                passed=False,
                filter_name="timestamp_consistency",
                reason="No valid time deltas",
            )

        dts = np.array(dts)
        mean_dt = np.mean(dts)
        std_dt = np.std(dts)

        if mean_dt > 0:
            variance_pct = (std_dt / mean_dt) * 100
        else:
            variance_pct = 0

        if variance_pct > config.max_timestamp_variance_pct:
            return FilterResult(
                passed=False,
                filter_name="timestamp_consistency",
                reason=f"High variance: {variance_pct:.1f}% > {config.max_timestamp_variance_pct}%",
                details={'mean_dt': mean_dt, 'std_dt': std_dt, 'variance_pct': variance_pct},
            )

        return FilterResult(
            passed=True,
            filter_name="timestamp_consistency",
            reason=f"OK: variance {variance_pct:.1f}%",
            details={'mean_dt': mean_dt, 'std_dt': std_dt, 'variance_pct': variance_pct},
        )

    def _check_value_ranges(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check for out-of-range values."""
        issues = []

        # Check actions
        actions = demo.actions
        max_action = np.max(np.abs(actions))
        if max_action > config.max_action_magnitude:
            issues.append(f"action magnitude {max_action:.2f} > {config.max_action_magnitude}")

        # Check positions if available
        positions = demo.positions
        if positions is not None:
            max_pos = np.max(np.abs(positions))
            if max_pos > config.max_position_magnitude:
                issues.append(f"position magnitude {max_pos:.1f} > {config.max_position_magnitude}")

        # Check velocities if available
        for step in demo.steps:
            if step.velocity is not None:
                vel_mag = np.linalg.norm(step.velocity)
                if vel_mag > config.max_velocity_magnitude:
                    issues.append(f"velocity magnitude {vel_mag:.1f} > {config.max_velocity_magnitude}")
                    break

        if issues:
            return FilterResult(
                passed=False,
                filter_name="value_ranges",
                reason="; ".join(issues),
            )

        return FilterResult(
            passed=True,
            filter_name="value_ranges",
            reason="OK: all values in range",
        )

    def _check_nan_values(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check for NaN or infinite values."""
        obs = demo.observations
        actions = demo.actions

        obs_nan = np.sum(~np.isfinite(obs))
        action_nan = np.sum(~np.isfinite(actions))

        if obs_nan > 0 or action_nan > 0:
            return FilterResult(
                passed=False,
                filter_name="nan_values",
                reason=f"Found NaN/inf: {obs_nan} in obs, {action_nan} in actions",
                details={'obs_nan': obs_nan, 'action_nan': action_nan},
            )

        return FilterResult(
            passed=True,
            filter_name="nan_values",
            reason="OK: no NaN/inf values",
        )

    def _check_quality_score(
        self,
        demo: 'Demonstration',
        config: FilterConfig,
    ) -> FilterResult:
        """Check quality score threshold."""
        if demo.quality_score < config.min_quality_score:
            return FilterResult(
                passed=False,
                filter_name="quality_score",
                reason=f"Low quality: {demo.quality_score:.2f} < {config.min_quality_score}",
                details={'quality_score': demo.quality_score},
            )

        return FilterResult(
            passed=True,
            filter_name="quality_score",
            reason=f"OK: quality {demo.quality_score:.2f}",
            details={'quality_score': demo.quality_score},
        )


def create_strict_filter() -> QualityFilter:
    """Create a strict quality filter for production use."""
    config = FilterConfig(
        max_pose_jitter_m=0.05,
        max_missing_frame_ratio=0.02,
        max_timestamp_variance_pct=30,
        min_quality_score=0.7,
    )
    return QualityFilter(config)


def create_lenient_filter() -> QualityFilter:
    """Create a lenient quality filter for exploratory use."""
    config = FilterConfig(
        max_pose_jitter_m=0.2,
        max_missing_frame_ratio=0.10,
        max_timestamp_variance_pct=100,
        min_quality_score=0.3,
    )
    return QualityFilter(config)

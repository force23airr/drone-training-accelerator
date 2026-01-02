"""
Promotion Gates

Gates that policies must pass before promotion to production.

Implements:
- PerformanceGate: Must beat baseline by X%
- SafetyGate: Zero crashes in M episodes (strict)
- RegressionGate: No regression on key metrics
- CompositeGate: Combine multiple gates
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from evaluation.harness.evaluation_harness import EpisodeResult


@dataclass
class GateCheckResult:
    """
    Result of checking a promotion gate.

    Attributes:
        passed: Whether the gate was passed
        gate_name: Name of the gate
        gate_type: Type of gate (performance, safety, regression)
        required_value: Value required to pass
        actual_value: Actual observed value
        message: Human-readable result message
        details: Additional details about the check
    """
    passed: bool
    gate_name: str
    gate_type: str
    required_value: Any
    actual_value: Any
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"[{status}] {self.gate_name}: {self.message}"


class PromotionGate(ABC):
    """
    Abstract base class for promotion gates.

    A gate defines a condition that must be met before a policy
    can be promoted to the next stage (e.g., staging, production).
    """

    def __init__(self, name: str, gate_type: str = "generic"):
        self.name = name
        self.gate_type = gate_type

    @abstractmethod
    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        """
        Check if the gate is passed.

        Args:
            episodes: List of episode results from evaluation
            aggregated_metrics: Aggregated metrics across episodes
                Format: {metric_name: {mean, std, min, max, count}}

        Returns:
            GateCheckResult with pass/fail status
        """
        pass


class PerformanceGate(PromotionGate):
    """
    Gate requiring policy to beat baseline by X%.

    Compares a specific metric against a baseline value and requires
    improvement by a specified percentage.
    """

    def __init__(
        self,
        metric_name: str,
        baseline_value: float,
        improvement_pct: float = 5.0,
        comparison: str = 'higher',
        use_stat: str = 'mean',
    ):
        """
        Args:
            metric_name: Name of metric to compare
            baseline_value: Baseline value (e.g., from PID controller)
            improvement_pct: Required improvement percentage
            comparison: 'higher' or 'lower' (which direction is better)
            use_stat: Which statistic to use ('mean', 'min', 'max')
        """
        super().__init__(f"performance_{metric_name}", "performance")
        self.metric_name = metric_name
        self.baseline_value = baseline_value
        self.improvement_pct = improvement_pct
        self.comparison = comparison
        self.use_stat = use_stat

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        # Check if metric exists
        if self.metric_name not in aggregated_metrics:
            return GateCheckResult(
                passed=False,
                gate_name=self.name,
                gate_type=self.gate_type,
                required_value=None,
                actual_value=None,
                message=f"Metric '{self.metric_name}' not found in results",
            )

        # Get actual value
        stats = aggregated_metrics[self.metric_name]
        actual = stats.get(self.use_stat, stats.get('mean', 0))

        # Compute required value
        if self.comparison == 'higher':
            required = self.baseline_value * (1 + self.improvement_pct / 100)
            passed = actual >= required
            direction = "above"
        else:  # lower is better
            required = self.baseline_value * (1 - self.improvement_pct / 100)
            passed = actual <= required
            direction = "below"

        # Compute actual improvement
        if self.baseline_value != 0:
            improvement = (actual - self.baseline_value) / abs(self.baseline_value) * 100
        else:
            improvement = 0

        message = (
            f"{'PASSED' if passed else 'FAILED'}: {self.metric_name} = {actual:.4f} "
            f"(required: {direction} {required:.4f}, baseline: {self.baseline_value:.4f}, "
            f"improvement: {improvement:+.1f}%)"
        )

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value=required,
            actual_value=actual,
            message=message,
            details={
                'baseline': self.baseline_value,
                'improvement_pct_required': self.improvement_pct,
                'improvement_pct_actual': improvement,
                'comparison': self.comparison,
            }
        )


class SafetyGate(PromotionGate):
    """
    Gate requiring zero crashes over M episodes.

    This is a strict gate: any crash results in failure.
    """

    def __init__(
        self,
        min_episodes: int = 100,
        max_crashes: int = 0,
        max_critical_violations: int = 0,
    ):
        """
        Args:
            min_episodes: Minimum episodes required for valid evaluation
            max_crashes: Maximum allowed crashes (default: 0 for strict)
            max_critical_violations: Maximum allowed critical constraint violations
        """
        super().__init__(f"safety_{min_episodes}ep_{max_crashes}crash", "safety")
        self.min_episodes = min_episodes
        self.max_crashes = max_crashes
        self.max_critical_violations = max_critical_violations

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        # Check minimum episodes
        if len(episodes) < self.min_episodes:
            return GateCheckResult(
                passed=False,
                gate_name=self.name,
                gate_type=self.gate_type,
                required_value=self.min_episodes,
                actual_value=len(episodes),
                message=f"Insufficient episodes: {len(episodes)} < {self.min_episodes} required",
            )

        # Count crashes from episodes
        crash_count = 0
        critical_violations = 0

        for ep in episodes:
            # Check crash count metric
            if 'crash_count' in ep.metrics:
                crash_count += int(ep.metrics['crash_count'])
            elif ep.metrics.get('terminated', 0) > 0 and not ep.success:
                # Infer crash from termination without success
                crash_count += 1

            # Check critical violations
            if 'critical_violations' in ep.metrics:
                critical_violations += int(ep.metrics['critical_violations'])

        # Also check aggregated metrics
        if 'crash_count' in aggregated_metrics:
            total_crashes = int(aggregated_metrics['crash_count'].get('sum',
                               aggregated_metrics['crash_count'].get('mean', 0) * len(episodes)))
            crash_count = max(crash_count, total_crashes)

        # Check gates
        crash_passed = crash_count <= self.max_crashes
        violation_passed = critical_violations <= self.max_critical_violations
        passed = crash_passed and violation_passed

        if not crash_passed:
            message = f"FAILED: {crash_count} crashes (max allowed: {self.max_crashes})"
        elif not violation_passed:
            message = f"FAILED: {critical_violations} critical violations (max: {self.max_critical_violations})"
        else:
            message = f"PASSED: {crash_count} crashes, {critical_violations} critical violations in {len(episodes)} episodes"

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value={'max_crashes': self.max_crashes, 'min_episodes': self.min_episodes},
            actual_value={'crashes': crash_count, 'critical_violations': critical_violations},
            message=message,
            details={
                'crash_count': crash_count,
                'critical_violations': critical_violations,
                'episodes_evaluated': len(episodes),
            }
        )


class RegressionGate(PromotionGate):
    """
    Gate ensuring no regression on key metrics.

    Compares current performance against a baseline (e.g., previous version)
    and fails if any metric regresses beyond tolerance.

    IMPORTANT: Fails CLOSED on missing metrics - if a metric is missing from
    either baseline or current eval, the gate fails. This prevents silent
    regressions due to metric name mismatches.
    """

    def __init__(
        self,
        baseline_metrics: Optional[Dict[str, float]] = None,
        baseline_path: Optional[str] = None,
        metrics_to_check: Optional[List[str]] = None,
        tolerance_pct: float = 5.0,
        critical_metrics: Optional[List[str]] = None,
        fail_on_missing: bool = True,  # FAIL CLOSED by default
    ):
        """
        Args:
            baseline_metrics: Dict of metric_name -> baseline_value
            baseline_path: Path to baseline evaluation_result.json (alternative to baseline_metrics)
            metrics_to_check: List of metrics to check (if using baseline_path)
            tolerance_pct: Allowed regression percentage
            critical_metrics: Metrics that must not regress at all
            fail_on_missing: If True, fail when metrics are missing (RECOMMENDED)
        """
        super().__init__("regression_check", "regression")

        # Load baseline from file if path provided
        if baseline_path and baseline_metrics is None:
            self.baseline_metrics = self._load_baseline(baseline_path, metrics_to_check or [])
            self.baseline_path = baseline_path
        else:
            self.baseline_metrics = baseline_metrics or {}
            self.baseline_path = None

        self.metrics_to_check = metrics_to_check or list(self.baseline_metrics.keys())
        self.tolerance_pct = tolerance_pct
        self.critical_metrics = critical_metrics or []
        self.fail_on_missing = fail_on_missing

    def _load_baseline(self, path: str, metrics: List[str]) -> Dict[str, float]:
        """Load baseline metrics from evaluation result JSON."""
        import json
        from pathlib import Path

        baseline = {}
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Try to extract metrics from various locations
            metrics_data = data.get('metrics', {})
            eval_summary = data.get('evaluation_summary', {})

            for metric in metrics:
                # Check aggregated metrics
                if metric in metrics_data:
                    if isinstance(metrics_data[metric], dict):
                        baseline[metric] = metrics_data[metric].get('mean', 0)
                    else:
                        baseline[metric] = float(metrics_data[metric])
                # Check evaluation summary
                elif metric in eval_summary:
                    baseline[metric] = float(eval_summary[metric])
                # Check top-level
                elif metric in data:
                    if isinstance(data[metric], dict):
                        baseline[metric] = data[metric].get('mean', 0)
                    else:
                        baseline[metric] = float(data[metric])

        except Exception as e:
            # Baseline file issues should fail loudly
            raise ValueError(f"Failed to load baseline from {path}: {e}")

        return baseline

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        regressions = []
        critical_regressions = []
        missing_metrics = []
        comparisons = {}

        for metric_name in self.metrics_to_check:
            # Check if metric exists in baseline
            if metric_name not in self.baseline_metrics:
                missing_metrics.append(f"{metric_name} (missing from baseline)")
                continue

            baseline_value = self.baseline_metrics[metric_name]

            # Check if metric exists in current eval - FAIL CLOSED if missing
            if metric_name not in aggregated_metrics:
                missing_metrics.append(f"{metric_name} (missing from current eval)")
                continue

            actual = aggregated_metrics[metric_name].get('mean', 0)

            # Compute regression (assuming higher is better)
            # Negative change means regression
            if baseline_value != 0:
                change_pct = (actual - baseline_value) / abs(baseline_value) * 100
            else:
                change_pct = 0

            comparisons[metric_name] = {
                'baseline': baseline_value,
                'actual': actual,
                'change_pct': change_pct,
            }

            # Check for regression
            min_allowed = baseline_value * (1 - self.tolerance_pct / 100)

            if actual < min_allowed:
                if metric_name in self.critical_metrics:
                    critical_regressions.append(
                        f"{metric_name}: {actual:.4f} < {min_allowed:.4f} ({change_pct:+.1f}%)"
                    )
                else:
                    regressions.append(
                        f"{metric_name}: {actual:.4f} < {min_allowed:.4f} ({change_pct:+.1f}%)"
                    )

        # Determine pass/fail - FAIL CLOSED on missing metrics
        missing_failure = self.fail_on_missing and len(missing_metrics) > 0
        passed = (
            len(critical_regressions) == 0
            and len(regressions) == 0
            and not missing_failure
        )

        if missing_failure:
            message = f"FAILED: Missing metrics (fail-closed): {', '.join(missing_metrics)}"
        elif critical_regressions:
            message = f"FAILED: Critical regressions: {', '.join(critical_regressions)}"
        elif regressions:
            message = f"FAILED: Regressions: {', '.join(regressions)}"
        else:
            message = f"PASSED: No regressions detected ({len(comparisons)} metrics checked)"

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value=f"No regression beyond {self.tolerance_pct}%",
            actual_value={'regressions': regressions, 'critical': critical_regressions},
            message=message,
            details={
                'comparisons': comparisons,
                'tolerance_pct': self.tolerance_pct,
            }
        )


class SuccessRateGate(PromotionGate):
    """
    Gate requiring minimum success rate.
    """

    def __init__(
        self,
        min_success_rate: float = 0.9,
        min_episodes: int = 50,
    ):
        """
        Args:
            min_success_rate: Minimum required success rate (0-1)
            min_episodes: Minimum episodes for valid evaluation
        """
        super().__init__(f"success_rate_{int(min_success_rate*100)}pct", "performance")
        self.min_success_rate = min_success_rate
        self.min_episodes = min_episodes

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        if len(episodes) < self.min_episodes:
            return GateCheckResult(
                passed=False,
                gate_name=self.name,
                gate_type=self.gate_type,
                required_value=self.min_episodes,
                actual_value=len(episodes),
                message=f"Insufficient episodes: {len(episodes)} < {self.min_episodes}",
            )

        # Compute success rate
        successes = sum(1 for ep in episodes if ep.success)
        success_rate = successes / len(episodes)

        passed = success_rate >= self.min_success_rate

        message = (
            f"{'PASSED' if passed else 'FAILED'}: "
            f"Success rate {success_rate*100:.1f}% ({successes}/{len(episodes)}) "
            f"{'≥' if passed else '<'} {self.min_success_rate*100:.1f}% required"
        )

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value=self.min_success_rate,
            actual_value=success_rate,
            message=message,
            details={
                'successes': successes,
                'total_episodes': len(episodes),
            }
        )


class InterventionRateGate(PromotionGate):
    """
    Gate checking safety shield intervention rates.

    This gate evaluates how often the safety shield had to intervene
    during evaluation. High intervention rates indicate the policy
    relies on the shield to stay safe rather than being safe itself.

    Use this to determine if a policy is safe enough to run WITHOUT
    the shield, or at least with minimal shield assistance.

    Requirements:
    - Emergency interventions must be zero (policy should never get that bad)
    - Override (guardian takeover) rate must be below threshold
    - Clamp rate below a higher threshold (minor clamping is acceptable)
    """

    def __init__(
        self,
        max_emergency_interventions: int = 0,
        max_override_rate: float = 0.01,      # 1% of timesteps
        max_clamp_rate: float = 0.05,         # 5% of timesteps
        min_episodes: int = 50,
    ):
        """
        Args:
            max_emergency_interventions: Maximum allowed emergency stops (default: 0)
            max_override_rate: Maximum override (guardian) rate as fraction of total steps
            max_clamp_rate: Maximum clamp rate as fraction of total steps
            min_episodes: Minimum episodes for valid evaluation
        """
        super().__init__("intervention_rate", "safety")
        self.max_emergency_interventions = max_emergency_interventions
        self.max_override_rate = max_override_rate
        self.max_clamp_rate = max_clamp_rate
        self.min_episodes = min_episodes

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        if len(episodes) < self.min_episodes:
            return GateCheckResult(
                passed=False,
                gate_name=self.name,
                gate_type=self.gate_type,
                required_value=self.min_episodes,
                actual_value=len(episodes),
                message=f"Insufficient episodes: {len(episodes)} < {self.min_episodes}",
            )

        # Accumulate intervention counts across episodes
        total_steps = 0
        total_clamps = 0
        total_overrides = 0
        total_emergencies = 0

        for ep in episodes:
            # Get step count
            ep_steps = ep.metrics.get('steps', ep.metrics.get('episode_length', 0))
            total_steps += int(ep_steps)

            # Check for shield stats in episode metrics
            # These come from shield's get_intervention_stats() if recorded
            shield_stats = ep.metrics.get('shield_stats', {})

            if isinstance(shield_stats, dict):
                by_type = shield_stats.get('by_type', {})
                total_clamps += by_type.get('clamp', 0)
                total_overrides += by_type.get('guardian', 0)
                total_emergencies += by_type.get('emergency', 0)
            else:
                # Fallback: check individual metrics
                total_clamps += int(ep.metrics.get('shield_clamp_count', 0))
                total_overrides += int(ep.metrics.get('shield_override_count', 0))
                total_emergencies += int(ep.metrics.get('shield_emergency_count', 0))

        # Compute rates
        if total_steps > 0:
            clamp_rate = total_clamps / total_steps
            override_rate = total_overrides / total_steps
        else:
            clamp_rate = 0
            override_rate = 0

        # Check each criterion
        emergency_passed = total_emergencies <= self.max_emergency_interventions
        override_passed = override_rate <= self.max_override_rate
        clamp_passed = clamp_rate <= self.max_clamp_rate

        passed = emergency_passed and override_passed and clamp_passed

        # Build message
        failures = []
        if not emergency_passed:
            failures.append(f"emergencies={total_emergencies} (max={self.max_emergency_interventions})")
        if not override_passed:
            failures.append(f"override_rate={override_rate*100:.2f}% (max={self.max_override_rate*100:.1f}%)")
        if not clamp_passed:
            failures.append(f"clamp_rate={clamp_rate*100:.2f}% (max={self.max_clamp_rate*100:.1f}%)")

        if passed:
            message = (
                f"PASSED: Intervention rates acceptable - "
                f"emergency={total_emergencies}, "
                f"override={override_rate*100:.2f}%, "
                f"clamp={clamp_rate*100:.2f}% "
                f"({total_steps} total steps)"
            )
        else:
            message = f"FAILED: {'; '.join(failures)}"

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value={
                'max_emergency': self.max_emergency_interventions,
                'max_override_rate': self.max_override_rate,
                'max_clamp_rate': self.max_clamp_rate,
            },
            actual_value={
                'emergencies': total_emergencies,
                'override_rate': override_rate,
                'clamp_rate': clamp_rate,
            },
            message=message,
            details={
                'total_steps': total_steps,
                'total_clamps': total_clamps,
                'total_overrides': total_overrides,
                'total_emergencies': total_emergencies,
                'episodes_evaluated': len(episodes),
            }
        )


class ViolationEpisodeGate(PromotionGate):
    """
    Gate checking episode-level violation counts.

    For conservative tasks (hover, landing, waypoint), violations should be
    rare or zero. This gate counts EPISODES with violations, not total events.

    An episode with tilt_violations > 0 counts as one tilt_violation_episode,
    regardless of how many violations occurred within it.

    Use this to ensure promoted policies don't routinely push limits and
    rely on recovery.
    """

    def __init__(
        self,
        max_tilt_violation_episodes: int = 0,
        max_altitude_violation_episodes: int = 0,
        max_geofence_violation_episodes: int = 0,
        max_speed_violation_episodes: int = 0,
        min_episodes: int = 50,
        task_type: str = "hover",  # hover, landing, waypoint, obstacle, racing
    ):
        """
        Args:
            max_tilt_violation_episodes: Max episodes with tilt violations
            max_altitude_violation_episodes: Max episodes with altitude violations
            max_geofence_violation_episodes: Max episodes with geofence violations
            max_speed_violation_episodes: Max episodes with speed violations
            min_episodes: Minimum episodes for valid evaluation
            task_type: Task type for context in messages
        """
        super().__init__("violation_episodes", "safety")
        self.max_tilt = max_tilt_violation_episodes
        self.max_altitude = max_altitude_violation_episodes
        self.max_geofence = max_geofence_violation_episodes
        self.max_speed = max_speed_violation_episodes
        self.min_episodes = min_episodes
        self.task_type = task_type

    @classmethod
    def for_hover(cls, min_episodes: int = 50) -> 'ViolationEpisodeGate':
        """Strict defaults for hover task - zero violations allowed."""
        return cls(
            max_tilt_violation_episodes=0,
            max_altitude_violation_episodes=0,
            max_geofence_violation_episodes=0,
            max_speed_violation_episodes=0,
            min_episodes=min_episodes,
            task_type="hover",
        )

    @classmethod
    def for_landing(cls, min_episodes: int = 50) -> 'ViolationEpisodeGate':
        """Strict defaults for landing task - zero violations allowed."""
        return cls(
            max_tilt_violation_episodes=0,
            max_altitude_violation_episodes=0,
            max_geofence_violation_episodes=0,
            max_speed_violation_episodes=0,
            min_episodes=min_episodes,
            task_type="landing",
        )

    @classmethod
    def for_waypoint(cls, min_episodes: int = 50) -> 'ViolationEpisodeGate':
        """Moderate defaults for waypoint - occasional violations OK."""
        return cls(
            max_tilt_violation_episodes=2,
            max_altitude_violation_episodes=1,
            max_geofence_violation_episodes=0,
            max_speed_violation_episodes=2,
            min_episodes=min_episodes,
            task_type="waypoint",
        )

    @classmethod
    def for_aggressive(cls, min_episodes: int = 50) -> 'ViolationEpisodeGate':
        """Loose defaults for aggressive tasks (obstacle, racing)."""
        return cls(
            max_tilt_violation_episodes=10,
            max_altitude_violation_episodes=5,
            max_geofence_violation_episodes=0,  # Still zero for geofence
            max_speed_violation_episodes=10,
            min_episodes=min_episodes,
            task_type="aggressive",
        )

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        if len(episodes) < self.min_episodes:
            return GateCheckResult(
                passed=False,
                gate_name=self.name,
                gate_type=self.gate_type,
                required_value=self.min_episodes,
                actual_value=len(episodes),
                message=f"Insufficient episodes: {len(episodes)} < {self.min_episodes}",
            )

        # Count episodes WITH violations (not total violations)
        tilt_episodes = 0
        altitude_episodes = 0
        geofence_episodes = 0
        speed_episodes = 0

        for ep in episodes:
            if ep.metrics.get('tilt_violations', 0) > 0:
                tilt_episodes += 1
            if ep.metrics.get('altitude_violations', 0) > 0:
                altitude_episodes += 1
            if ep.metrics.get('geofence_violations', 0) > 0:
                geofence_episodes += 1
            if ep.metrics.get('speed_violations', 0) > 0:
                speed_episodes += 1

        # Check each limit
        failures = []
        if tilt_episodes > self.max_tilt:
            failures.append(f"tilt={tilt_episodes} (max={self.max_tilt})")
        if altitude_episodes > self.max_altitude:
            failures.append(f"altitude={altitude_episodes} (max={self.max_altitude})")
        if geofence_episodes > self.max_geofence:
            failures.append(f"geofence={geofence_episodes} (max={self.max_geofence})")
        if speed_episodes > self.max_speed:
            failures.append(f"speed={speed_episodes} (max={self.max_speed})")

        passed = len(failures) == 0

        if passed:
            message = (
                f"PASSED ({self.task_type}): Violation episodes within limits - "
                f"tilt={tilt_episodes}, altitude={altitude_episodes}, "
                f"geofence={geofence_episodes}, speed={speed_episodes}"
            )
        else:
            message = f"FAILED ({self.task_type}): {'; '.join(failures)}"

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value={
                'max_tilt': self.max_tilt,
                'max_altitude': self.max_altitude,
                'max_geofence': self.max_geofence,
                'max_speed': self.max_speed,
            },
            actual_value={
                'tilt_episodes': tilt_episodes,
                'altitude_episodes': altitude_episodes,
                'geofence_episodes': geofence_episodes,
                'speed_episodes': speed_episodes,
            },
            message=message,
            details={
                'task_type': self.task_type,
                'episodes_evaluated': len(episodes),
            }
        )


class CompositeGate(PromotionGate):
    """
    Combines multiple gates with AND logic.

    All gates must pass for the composite to pass.
    """

    def __init__(
        self,
        gates: List[PromotionGate],
        name: str = "composite_gate",
        require_all: bool = True,
    ):
        """
        Args:
            gates: List of gates to check
            name: Name for this composite gate
            require_all: If True, all gates must pass. If False, any gate passing is sufficient.
        """
        super().__init__(name, "composite")
        self.gates = gates
        self.require_all = require_all

    def check(
        self,
        episodes: List['EpisodeResult'],
        aggregated_metrics: Dict[str, Dict[str, float]],
    ) -> GateCheckResult:
        results = []

        for gate in self.gates:
            result = gate.check(episodes, aggregated_metrics)
            results.append(result)

        # Determine overall pass/fail
        if self.require_all:
            passed = all(r.passed for r in results)
            logic = "AND"
        else:
            passed = any(r.passed for r in results)
            logic = "OR"

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        if passed:
            message = f"PASSED: {passed_count}/{total} gates passed ({logic} logic)"
        else:
            failed_gates = [r.gate_name for r in results if not r.passed]
            message = f"FAILED: {passed_count}/{total} gates passed. Failed: {', '.join(failed_gates)}"

        return GateCheckResult(
            passed=passed,
            gate_name=self.name,
            gate_type=self.gate_type,
            required_value=f"All gates pass" if self.require_all else "Any gate passes",
            actual_value=results,
            message=message,
            details={
                'gate_results': [
                    {'gate': r.gate_name, 'passed': r.passed, 'message': r.message}
                    for r in results
                ],
            }
        )


def create_default_gates(
    baseline_reward: float = 0.0,
    baseline_success_rate: float = 0.5,
    improvement_required: float = 10.0,
    min_episodes: int = 100,
    include_intervention_gate: bool = False,
    max_override_rate: float = 0.01,
    max_clamp_rate: float = 0.05,
) -> CompositeGate:
    """
    Create default promotion gates.

    Args:
        baseline_reward: Baseline mean reward (e.g., from PID controller)
        baseline_success_rate: Baseline success rate
        improvement_required: Required improvement percentage
        min_episodes: Minimum episodes for evaluation
        include_intervention_gate: Whether to include shield intervention rate gate
        max_override_rate: Maximum guardian override rate (if intervention gate enabled)
        max_clamp_rate: Maximum clamp rate (if intervention gate enabled)

    Returns:
        CompositeGate with standard gates:
        - Performance: Beat baseline reward by X%
        - Safety: Zero crashes in M episodes
        - Success rate: At least 90%
        - (Optional) Intervention rate: Shield usage limits
    """
    gates = [
        # Performance gate: beat baseline reward
        PerformanceGate(
            metric_name='total_reward',
            baseline_value=baseline_reward,
            improvement_pct=improvement_required,
            comparison='higher',
        ),

        # Safety gate: zero crashes (strict)
        SafetyGate(
            min_episodes=min_episodes,
            max_crashes=0,
            max_critical_violations=0,
        ),

        # Success rate gate
        SuccessRateGate(
            min_success_rate=0.9,
            min_episodes=min_episodes // 2,
        ),
    ]

    # Optionally add intervention rate gate for shielded evaluation
    if include_intervention_gate:
        gates.append(
            InterventionRateGate(
                max_emergency_interventions=0,
                max_override_rate=max_override_rate,
                max_clamp_rate=max_clamp_rate,
                min_episodes=min_episodes // 2,
            )
        )

    return CompositeGate(gates, name="default_promotion_gates")

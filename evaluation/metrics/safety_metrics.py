"""
Safety Metrics Collector

Tracks safety-critical metrics for UAV evaluation:
- Crash detection
- Near-miss events
- Constraint violations (tilt, altitude, speed, no-fly zones)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from evaluation.metrics.base_metrics import (
    MetricCollector,
    MetricResult,
    ConstraintViolation,
)


@dataclass
class NoFlyZone:
    """Definition of a no-fly zone."""
    center: np.ndarray  # [x, y, z] center position
    radius: float       # Radius in meters
    height: Optional[float] = None  # Height limit (None = infinite)
    name: str = "no_fly_zone"


class SafetyMetrics(MetricCollector):
    """
    Collects safety-related metrics.

    Tracks:
    - Crashes (ground collision, obstacle collision)
    - Near-misses (close approaches to obstacles)
    - Constraint violations:
      - Tilt angle exceeds limit
      - Altitude below minimum
      - Speed exceeds maximum
      - Entry into no-fly zones
    """

    def __init__(
        self,
        max_tilt_deg: float = 60.0,
        min_altitude_m: float = 0.5,
        max_speed_m_s: float = 20.0,
        max_acceleration_m_s2: float = 30.0,
        no_fly_zones: Optional[List[NoFlyZone]] = None,
        near_miss_threshold_m: float = 1.0,
        critical_distance_m: float = 0.5,
    ):
        """
        Args:
            max_tilt_deg: Maximum allowed tilt angle (roll/pitch)
            min_altitude_m: Minimum allowed altitude above ground
            max_speed_m_s: Maximum allowed speed
            max_acceleration_m_s2: Maximum allowed acceleration
            no_fly_zones: List of no-fly zone definitions
            near_miss_threshold_m: Distance threshold for near-miss detection
            critical_distance_m: Distance threshold for critical near-miss
        """
        super().__init__(name="SafetyMetrics")

        self.max_tilt_deg = max_tilt_deg
        self.min_altitude_m = min_altitude_m
        self.max_speed_m_s = max_speed_m_s
        self.max_acceleration_m_s2 = max_acceleration_m_s2
        self.no_fly_zones = no_fly_zones or []
        self.near_miss_threshold = near_miss_threshold_m
        self.critical_distance = critical_distance_m

        self._violations: List[ConstraintViolation] = []
        self._crashes: int = 0
        self._near_misses: int = 0
        self._critical_near_misses: int = 0
        self._min_obstacle_distance: float = float('inf')
        self._min_ground_clearance: float = float('inf')
        self._prev_velocity: Optional[np.ndarray] = None
        self._max_tilt_observed: float = 0.0
        self._max_speed_observed: float = 0.0
        self._time_in_violation: float = 0.0

    def reset(self) -> None:
        super().reset()
        self._violations = []
        self._crashes = 0
        self._near_misses = 0
        self._critical_near_misses = 0
        self._min_obstacle_distance = float('inf')
        self._min_ground_clearance = float('inf')
        self._prev_velocity = None
        self._max_tilt_observed = 0.0
        self._max_speed_observed = 0.0
        self._time_in_violation = 0.0

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

        # Extract state from observation
        # Standard observation: [pos(3), vel(3), quat(4), angular_vel(3)]
        pos = next_obs[:3]
        vel = next_obs[3:6]
        quat = next_obs[6:10] if len(next_obs) >= 10 else np.array([1, 0, 0, 0])

        step_num = self._step_count
        timestamp = info.get('time', step_num * 0.01)

        # Check altitude
        altitude = pos[2]
        self._min_ground_clearance = min(self._min_ground_clearance, altitude)

        if altitude < self.min_altitude_m:
            self._violations.append(ConstraintViolation(
                timestamp=timestamp,
                step=step_num,
                constraint_type='min_altitude',
                value=altitude,
                limit=self.min_altitude_m,
                position=pos.copy(),
                severity='critical' if altitude < 0.1 else 'warning',
            ))
            self._time_in_violation += 0.01

        # Check tilt angle
        tilt_deg = self._compute_tilt_from_quat(quat)
        self._max_tilt_observed = max(self._max_tilt_observed, tilt_deg)

        if tilt_deg > self.max_tilt_deg:
            self._violations.append(ConstraintViolation(
                timestamp=timestamp,
                step=step_num,
                constraint_type='tilt_limit',
                value=tilt_deg,
                limit=self.max_tilt_deg,
                position=pos.copy(),
                severity='critical' if tilt_deg > 80 else 'warning',
            ))
            self._time_in_violation += 0.01

        # Check speed
        speed = np.linalg.norm(vel)
        self._max_speed_observed = max(self._max_speed_observed, speed)

        if speed > self.max_speed_m_s:
            self._violations.append(ConstraintViolation(
                timestamp=timestamp,
                step=step_num,
                constraint_type='max_speed',
                value=speed,
                limit=self.max_speed_m_s,
                position=pos.copy(),
                severity='warning',
            ))

        # Check acceleration
        if self._prev_velocity is not None:
            dt = info.get('dt', 0.01)
            accel = np.linalg.norm(vel - self._prev_velocity) / max(dt, 0.001)
            if accel > self.max_acceleration_m_s2:
                self._violations.append(ConstraintViolation(
                    timestamp=timestamp,
                    step=step_num,
                    constraint_type='max_acceleration',
                    value=accel,
                    limit=self.max_acceleration_m_s2,
                    position=pos.copy(),
                    severity='warning',
                ))
        self._prev_velocity = vel.copy()

        # Check no-fly zones
        for zone in self.no_fly_zones:
            dist_xy = np.linalg.norm(pos[:2] - zone.center[:2])
            in_zone = dist_xy < zone.radius
            if zone.height is not None:
                in_zone = in_zone and (pos[2] < zone.height)

            if in_zone:
                self._violations.append(ConstraintViolation(
                    timestamp=timestamp,
                    step=step_num,
                    constraint_type='no_fly_zone',
                    value=dist_xy,
                    limit=zone.radius,
                    position=pos.copy(),
                    severity='critical',
                ))

        # Check obstacle distances (from info)
        if 'obstacle_distances' in info:
            min_dist = min(info['obstacle_distances'])
            self._min_obstacle_distance = min(self._min_obstacle_distance, min_dist)

            if min_dist < self.near_miss_threshold:
                self._near_misses += 1
                if min_dist < self.critical_distance:
                    self._critical_near_misses += 1

        # Check for collision in info
        if info.get('collision', False) or info.get('collisions', 0) > 0:
            self._crashes = 1  # Binary for episode

    def _compute_tilt_from_quat(self, quat: np.ndarray) -> float:
        """
        Compute tilt angle (combined roll/pitch) from quaternion.

        Args:
            quat: Quaternion [w, x, y, z] or [x, y, z, w]

        Returns:
            Tilt angle in degrees
        """
        # Handle both quaternion conventions
        if len(quat) >= 4:
            # Assume [w, x, y, z] convention
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]

            # Compute roll and pitch
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            sinp = np.clip(sinp, -1, 1)
            pitch = np.arcsin(sinp)

            # Combined tilt
            tilt_rad = np.sqrt(roll**2 + pitch**2)
            return np.degrees(tilt_rad)

        return 0.0

    def episode_end(self, info: Dict[str, Any]) -> List[MetricResult]:
        # Check final termination reason
        if info.get('terminated', False):
            if info.get('collision', False) or 'crash' in str(info.get('termination_reason', '')).lower():
                self._crashes = 1

        # Count violations by type
        tilt_violations = sum(1 for v in self._violations if v.constraint_type == 'tilt_limit')
        altitude_violations = sum(1 for v in self._violations if v.constraint_type == 'min_altitude')
        speed_violations = sum(1 for v in self._violations if v.constraint_type == 'max_speed')
        nfz_violations = sum(1 for v in self._violations if v.constraint_type == 'no_fly_zone')
        accel_violations = sum(1 for v in self._violations if v.constraint_type == 'max_acceleration')
        critical_violations = sum(1 for v in self._violations if v.severity == 'critical')

        return [
            MetricResult(
                'crash_count', self._crashes, 'count',
                higher_is_better=False,
                metadata={'is_crash': self._crashes > 0}
            ),
            MetricResult(
                'near_miss_count', self._near_misses, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'critical_near_miss_count', self._critical_near_misses, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'total_constraint_violations', len(self._violations), 'count',
                higher_is_better=False
            ),
            MetricResult(
                'critical_violations', critical_violations, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'tilt_violations', tilt_violations, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'altitude_violations', altitude_violations, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'speed_violations', speed_violations, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'no_fly_zone_violations', nfz_violations, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'acceleration_violations', accel_violations, 'count',
                higher_is_better=False
            ),
            MetricResult(
                'min_obstacle_distance', self._min_obstacle_distance, 'meters',
                higher_is_better=True
            ),
            MetricResult(
                'min_ground_clearance', self._min_ground_clearance, 'meters',
                higher_is_better=True
            ),
            MetricResult(
                'max_tilt_observed', self._max_tilt_observed, 'degrees',
                higher_is_better=False
            ),
            MetricResult(
                'max_speed_observed', self._max_speed_observed, 'm/s',
                higher_is_better=False
            ),
            MetricResult(
                'time_in_violation', self._time_in_violation, 'seconds',
                higher_is_better=False
            ),
            MetricResult(
                'safety_score',
                self._compute_safety_score(),
                'normalized',
                higher_is_better=True,
                metadata={'crash': self._crashes, 'violations': len(self._violations)}
            ),
        ]

    def _compute_safety_score(self) -> float:
        """
        Compute overall safety score (0-1, higher is safer).

        Factors:
        - No crashes: 50% weight
        - No critical violations: 30% weight
        - Low violation count: 20% weight
        """
        crash_score = 1.0 if self._crashes == 0 else 0.0

        critical_count = sum(1 for v in self._violations if v.severity == 'critical')
        critical_score = 1.0 / (1.0 + critical_count)

        violation_score = 1.0 / (1.0 + len(self._violations) / 10.0)

        return 0.5 * crash_score + 0.3 * critical_score + 0.2 * violation_score

    def get_running_metrics(self) -> Dict[str, float]:
        return {
            'violations': len(self._violations),
            'near_misses': self._near_misses,
            'max_tilt': self._max_tilt_observed,
        }

    def get_violations(self) -> List[ConstraintViolation]:
        """Get all recorded violations for analysis."""
        return self._violations.copy()

"""
Loitering Munition Environment.

Training environment for loitering munitions (Switchblade, Harop) with:
- Loiter pattern establishment
- Target acquisition and tracking
- Man-in-the-loop abort capability
- Terminal attack dive
- Anti-radiation seeking (Harop)

Key differences from strike environment:
- Expendable platform (mission ends with impact)
- Extended loiter capability
- Terminal dive physics
- Abort/wave-off mechanics
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum, auto
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base_fixed_wing_env import BaseFixedWingEnv, WeaponState, FixedWingObservation


class LoiterPhase(Enum):
    """Loitering munition mission phases."""
    TRANSIT = auto()       # Flying to loiter area
    LOITER = auto()        # Searching for targets
    ACQUIRE = auto()       # Target acquired, tracking
    TERMINAL = auto()      # Terminal attack dive
    ABORT = auto()         # Aborted attack, returning to loiter
    IMPACT = auto()        # Impact/detonation


class TargetMotion(Enum):
    """Target motion patterns."""
    STATIC = "static"
    LINEAR = "linear"
    EVASIVE = "evasive"


@dataclass
class LoiterTarget:
    """Target for loitering munition attack."""
    position: np.ndarray        # Current position
    velocity: np.ndarray        # Current velocity
    target_type: str            # "vehicle", "radar", "infantry"
    motion: TargetMotion
    detected: bool = False
    tracking_time: float = 0.0  # Time target has been tracked
    signature: float = 1.0      # Detection signature (radar/IR)

    def update(self, dt: float):
        """Update target position based on motion pattern."""
        if self.motion == TargetMotion.STATIC:
            pass  # No movement
        elif self.motion == TargetMotion.LINEAR:
            self.position += self.velocity * dt
        elif self.motion == TargetMotion.EVASIVE:
            # Random evasive maneuvers
            self.velocity += np.random.normal(0, 2, 3) * dt
            self.velocity[2] = 0  # Stay on ground
            self.position += self.velocity * dt


@dataclass
class RadarEmitter:
    """Radar emitter for anti-radiation munitions (Harop)."""
    position: np.ndarray
    frequency_band: str         # "S", "X", "L", etc.
    power: float                # Effective radiated power
    active: bool = True
    emission_detected: bool = False

    def get_signal_strength(self, receiver_pos: np.ndarray) -> float:
        """Get signal strength at receiver position."""
        if not self.active:
            return 0.0

        dist = np.linalg.norm(receiver_pos - self.position)
        # Signal drops with distance squared
        return self.power / (dist ** 2 + 1.0)


class LoiteringMunitionEnv(BaseFixedWingEnv):
    """
    Loitering munition training environment.

    Simulates Switchblade 600 and IAI Harop style munitions with:
    - Extended loiter patterns (racetrack, orbit, figure-8)
    - Target acquisition using onboard sensors
    - Man-in-the-loop abort capability
    - Terminal attack dive mechanics
    - Anti-radiation homing (for Harop)

    The episode ends with:
    - Successful impact on target
    - Abort and RTB
    - Fuel/battery exhaustion
    - Loss of control

    Observation (28 dims):
    - Base flight state (21)
    - Target info (4)
    - Mission state (3)

    Action space:
    - [aileron, elevator, rudder, throttle, terminal_commit, abort]
    """

    def __init__(
        self,
        platform_config: Dict[str, Any],
        munition_type: str = "switchblade",  # or "harop"
        num_targets: int = 3,
        target_motion: str = "static",
        loiter_area_radius: float = 2000.0,
        **kwargs
    ):
        """
        Initialize loitering munition environment.

        Args:
            platform_config: Platform configuration
            munition_type: "switchblade" or "harop"
            num_targets: Number of potential targets
            target_motion: "static", "linear", "evasive"
            loiter_area_radius: Radius of loiter area [m]
        """
        # Override max episode steps for loiter endurance
        if 'max_episode_steps' not in kwargs:
            kwargs['max_episode_steps'] = 8000  # ~40 minutes at 48 Hz

        super().__init__(platform_config, **kwargs)

        self.munition_type = munition_type
        self.num_targets = num_targets
        self.target_motion_type = TargetMotion(target_motion)
        self.loiter_area_radius = loiter_area_radius

        # Mission state
        self._phase = LoiterPhase.TRANSIT
        self._targets: List[LoiterTarget] = []
        self._radar_emitters: List[RadarEmitter] = []
        self._acquired_target: Optional[LoiterTarget] = None
        self._tracking_time = 0.0
        self._terminal_dive_started = False
        self._abort_count = 0

        # Loiter area
        self._loiter_center = np.array([5000.0, 0.0, 1000.0])

        # Terminal dive parameters
        self._terminal_dive_angle = np.radians(60)  # 60 degree dive
        self._terminal_min_altitude = 50.0
        self._impact_radius = 5.0  # CEP

        # Sensor parameters
        physics = platform_config.get('physics_params', {})
        self._sensor_range = 3000.0
        self._sensor_fov = np.radians(60)
        self._is_anti_radiation = (munition_type == "harop" or
                                   physics.get('seeker_type') == 'anti_radiation')

        # Extended observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )

        # Action space with terminal commit and abort
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new mission."""
        obs_array, info = super().reset(seed=seed, options=options)

        # Reset mission state
        self._phase = LoiterPhase.TRANSIT
        self._acquired_target = None
        self._tracking_time = 0.0
        self._terminal_dive_started = False
        self._abort_count = 0

        # Generate targets
        self._generate_targets()

        # Generate radar emitters (for anti-radiation)
        if self._is_anti_radiation:
            self._generate_radar_emitters()

        # Initial position (launch point)
        self._position = np.array([0.0, 0.0, 500.0])
        self._velocity = np.array([30.0, 0.0, 5.0])  # Climbing launch

        # Get observation
        obs = self._get_loiter_observation()

        info.update({
            'phase': self._phase.name,
            'targets_available': len([t for t in self._targets if not t.detected]),
            'abort_count': self._abort_count,
        })

        return obs, info

    def _generate_targets(self):
        """Generate ground targets in loiter area."""
        self._targets = []

        for i in range(self.num_targets):
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(0, self.loiter_area_radius)

            pos = self._loiter_center.copy()
            pos[0] += radius * np.cos(angle)
            pos[1] += radius * np.sin(angle)
            pos[2] = 0  # Ground level

            # Random velocity for moving targets
            if self.target_motion_type != TargetMotion.STATIC:
                vel = np.array([
                    self.np_random.uniform(-5, 5),
                    self.np_random.uniform(-5, 5),
                    0.0
                ])
            else:
                vel = np.zeros(3)

            target_type = self.np_random.choice(["vehicle", "infantry", "radar"])

            self._targets.append(LoiterTarget(
                position=pos,
                velocity=vel,
                target_type=target_type,
                motion=self.target_motion_type,
                signature=self.np_random.uniform(0.5, 1.5),
            ))

    def _generate_radar_emitters(self):
        """Generate radar emitters for anti-radiation mission."""
        self._radar_emitters = []

        for target in self._targets:
            if target.target_type == "radar":
                self._radar_emitters.append(RadarEmitter(
                    position=target.position.copy(),
                    frequency_band=self.np_random.choice(["S", "X", "L"]),
                    power=self.np_random.uniform(1000, 5000),
                ))

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with loitering munition logic."""
        # Parse special actions
        terminal_commit = action[4] > 0.5 if len(action) > 4 else False
        abort = action[5] > 0.5 if len(action) > 5 else False

        # Handle abort
        if abort and self._phase in [LoiterPhase.ACQUIRE, LoiterPhase.TERMINAL]:
            self._phase = LoiterPhase.ABORT
            self._abort_count += 1
            self._acquired_target = None
            terminal_commit = False

        # Execute base physics
        self._current_step += 1
        dt = 1.0 / self.control_hz

        # Parse control actions
        aileron_cmd = float(action[0])
        elevator_cmd = float(action[1])
        rudder_cmd = float(action[2])
        throttle = float(action[3])

        # In terminal phase, override controls for dive
        if self._phase == LoiterPhase.TERMINAL and self._acquired_target:
            aileron_cmd, elevator_cmd, rudder_cmd, throttle = self._compute_terminal_guidance()

        # Mix control surfaces
        surfaces = self._mixer.mix(
            roll_cmd=aileron_cmd,
            pitch_cmd=elevator_cmd,
            yaw_cmd=rudder_cmd,
            throttle=throttle,
            dt=dt
        )

        self._current_surfaces = np.array([
            surfaces.aileron, surfaces.elevator,
            surfaces.rudder, surfaces.flaps
        ])
        self._throttle = throttle

        # Physics simulation
        for _ in range(self.physics_steps_per_control):
            self._physics_step(surfaces, throttle)

        # Update targets
        for target in self._targets:
            target.update(dt)

        # Update mission phase
        self._update_loiter_phase(terminal_commit)

        # Sensor updates
        self._update_sensor_tracking(dt)

        # Get observation
        obs = self._get_loiter_observation()

        # Compute reward
        reward = self._compute_loiter_reward(action)

        # Check termination
        terminated = self._check_loiter_terminated()
        truncated = self._current_step >= self._max_episode_steps

        info = {
            'phase': self._phase.name,
            'acquired_target': self._acquired_target is not None,
            'tracking_time': self._tracking_time,
            'altitude': self._position[2],
            'abort_count': self._abort_count,
        }

        return obs, reward, terminated, truncated, info

    def _update_loiter_phase(self, terminal_commit: bool):
        """Update mission phase based on state."""
        dist_to_loiter = np.linalg.norm(
            self._position[:2] - self._loiter_center[:2]
        )

        if self._phase == LoiterPhase.TRANSIT:
            if dist_to_loiter < self.loiter_area_radius:
                self._phase = LoiterPhase.LOITER

        elif self._phase == LoiterPhase.LOITER:
            # Check for target acquisition
            if self._acquired_target is not None:
                self._phase = LoiterPhase.ACQUIRE

        elif self._phase == LoiterPhase.ACQUIRE:
            # Require tracking time before terminal commit
            if terminal_commit and self._tracking_time > 2.0:
                self._phase = LoiterPhase.TERMINAL
                self._terminal_dive_started = True

            # Lost tracking
            if self._acquired_target is None:
                self._phase = LoiterPhase.LOITER
                self._tracking_time = 0.0

        elif self._phase == LoiterPhase.TERMINAL:
            # Check for impact
            if self._acquired_target:
                dist_to_target = np.linalg.norm(
                    self._position - self._acquired_target.position
                )
                if dist_to_target < self._impact_radius or self._position[2] < 5:
                    self._phase = LoiterPhase.IMPACT

        elif self._phase == LoiterPhase.ABORT:
            # Return to loiter
            if self._position[2] > 500:
                self._phase = LoiterPhase.LOITER

    def _update_sensor_tracking(self, dt: float):
        """Update target tracking with onboard sensor."""
        if self._phase in [LoiterPhase.IMPACT, LoiterPhase.TERMINAL]:
            return

        # Check sensor footprint
        best_target = None
        best_score = 0.0

        for target in self._targets:
            dist = np.linalg.norm(self._position - target.position)

            # Range check
            if dist > self._sensor_range:
                continue

            # FOV check (simplified - look down)
            if self._position[2] < target.position[2]:
                continue

            # Compute detection score
            score = target.signature / (dist / 1000.0 + 1.0)

            # Anti-radiation bonus
            if self._is_anti_radiation:
                for emitter in self._radar_emitters:
                    if np.linalg.norm(target.position - emitter.position) < 50:
                        signal = emitter.get_signal_strength(self._position)
                        score += signal * 0.01

            if score > best_score:
                best_score = score
                best_target = target

        # Update tracking
        if best_target is not None:
            if self._acquired_target == best_target:
                self._tracking_time += dt
            else:
                self._acquired_target = best_target
                self._tracking_time = 0.0
                best_target.detected = True
        else:
            self._acquired_target = None
            self._tracking_time = 0.0

    def _compute_terminal_guidance(self) -> Tuple[float, float, float, float]:
        """Compute control inputs for terminal dive."""
        if not self._acquired_target:
            return 0.0, 0.0, 0.0, 0.5

        # Vector to target
        to_target = self._acquired_target.position - self._position
        dist = np.linalg.norm(to_target)

        if dist < 1.0:
            return 0.0, 0.0, 0.0, 1.0

        # Desired direction
        desired_dir = to_target / dist

        # Current direction
        speed = np.linalg.norm(self._velocity)
        if speed > 1.0:
            current_dir = self._velocity / speed
        else:
            current_dir = np.array([1.0, 0.0, 0.0])

        # Proportional navigation
        # Cross product for required turn
        cross = np.cross(current_dir, desired_dir)

        # Pitch down for dive
        pitch_cmd = np.clip(-desired_dir[2] - 0.5, -1.0, 1.0)

        # Roll/yaw for lateral correction
        roll_cmd = np.clip(cross[2] * 2.0, -1.0, 1.0)
        yaw_cmd = np.clip(cross[1] * 2.0, -1.0, 1.0)

        # Full throttle for terminal
        throttle = 1.0

        return roll_cmd, pitch_cmd, yaw_cmd, throttle

    def _get_loiter_observation(self) -> np.ndarray:
        """Get observation with loiter-specific info."""
        base_obs = self._get_observation()
        base_array = base_obs.to_array()  # 21 dims

        # Target info (4 dims)
        if self._acquired_target:
            rel_pos = (self._acquired_target.position - self._position) / 1000.0
            target_info = np.array([
                rel_pos[0], rel_pos[1], rel_pos[2],
                self._tracking_time / 10.0
            ])
        else:
            target_info = np.zeros(4)

        # Mission state (3 dims)
        mission_state = np.array([
            float(self._phase.value) / 6.0,
            float(self._acquired_target is not None),
            float(self._terminal_dive_started),
        ])

        return np.concatenate([
            base_array,      # 21
            target_info,     # 4
            mission_state,   # 3
        ]).astype(np.float32)  # Total: 28

    def _compute_loiter_reward(self, action: np.ndarray) -> float:
        """Compute reward for loitering mission."""
        reward = 0.0

        # Survival bonus (less important for expendable munition)
        reward += 0.5

        # Phase-specific rewards
        if self._phase == LoiterPhase.TRANSIT:
            dist_to_loiter = np.linalg.norm(
                self._position[:2] - self._loiter_center[:2]
            )
            reward += 0.001 * (10000 - dist_to_loiter) / 10000

        elif self._phase == LoiterPhase.LOITER:
            # Stay in loiter area
            dist_to_loiter = np.linalg.norm(
                self._position[:2] - self._loiter_center[:2]
            )
            if dist_to_loiter < self.loiter_area_radius:
                reward += 0.5
            else:
                reward -= 0.5

            # Altitude for good sensor coverage
            if 800 < self._position[2] < 1500:
                reward += 0.3

        elif self._phase == LoiterPhase.ACQUIRE:
            # Reward tracking
            reward += 2.0 * min(self._tracking_time / 5.0, 1.0)

            # Maintain track
            if self._acquired_target:
                dist_to_target = np.linalg.norm(
                    self._position[:2] - self._acquired_target.position[:2]
                )
                if dist_to_target < self._sensor_range * 0.8:
                    reward += 0.5

        elif self._phase == LoiterPhase.TERMINAL:
            # Reward closing on target
            if self._acquired_target:
                dist = np.linalg.norm(
                    self._position - self._acquired_target.position
                )
                reward += 5.0 * (1.0 - dist / 1000.0)

        elif self._phase == LoiterPhase.IMPACT:
            # Big reward for successful impact
            reward += 100.0

            # Bonus for target type
            if self._acquired_target:
                if self._acquired_target.target_type == "radar":
                    reward += 50.0  # High value target
                elif self._acquired_target.target_type == "vehicle":
                    reward += 30.0

        elif self._phase == LoiterPhase.ABORT:
            # Small penalty for abort, but not too much (safety matters)
            reward -= 5.0

        # Penalties
        reward -= 0.01 * np.sum(action[:3] ** 2)  # Smooth control

        # Penalty for excessive aborts
        reward -= 2.0 * self._abort_count

        return reward

    def _check_loiter_terminated(self) -> bool:
        """Check if loitering mission should terminate."""
        # Impact
        if self._phase == LoiterPhase.IMPACT:
            return True

        # Crash
        if self._position[2] < 0:
            return True

        # Base termination
        if super()._check_terminated():
            return True

        # Fuel/battery exhaustion
        if self._fuel_remaining <= 0:
            return True

        return False

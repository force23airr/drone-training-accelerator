"""
Air-to-Ground Strike Environment.

Training environment for UCAV strike missions with:
- Mission phases (ingress, attack, egress)
- Threat modeling (SAM, AAA, radar)
- Target acquisition and engagement
- Weapon delivery mechanics
- Terrain masking opportunities

Designed for X-47B and similar UCAVs.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum, auto
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base_fixed_wing_env import BaseFixedWingEnv, WeaponState, FixedWingObservation


class MissionPhase(Enum):
    """Current phase of strike mission."""
    INGRESS = auto()      # Flying to target area
    ATTACK = auto()       # In target area, engaging
    EGRESS = auto()       # Exiting target area
    RTB = auto()          # Return to base
    COMPLETE = auto()     # Mission complete


class ThreatType(Enum):
    """Types of air defense threats."""
    SAM_LONG = "sam_long"      # Long-range SAM (S-300 class)
    SAM_MEDIUM = "sam_medium"  # Medium-range SAM (Buk class)
    SAM_SHORT = "sam_short"    # Short-range SAM (Tor class)
    AAA = "aaa"                # Anti-aircraft artillery
    MANPADS = "manpads"        # Man-portable AD


@dataclass
class Threat:
    """Air defense threat definition."""
    position: np.ndarray        # [x, y, z] position
    threat_type: ThreatType
    detection_range: float      # Radar detection range [m]
    engagement_range: float     # Weapon engagement range [m]
    min_altitude: float         # Minimum engagement altitude [m]
    max_altitude: float         # Maximum engagement altitude [m]
    reaction_time: float        # Time to engage after detection [s]
    pk_base: float              # Base probability of kill
    active: bool = True         # Is threat operational
    tracking: bool = False      # Currently tracking aircraft
    time_tracking: float = 0.0  # Time spent tracking

    def can_detect(self, aircraft_pos: np.ndarray, rcs: float = 1.0) -> bool:
        """Check if threat can detect aircraft."""
        if not self.active:
            return False

        dist = np.linalg.norm(aircraft_pos[:2] - self.position[:2])
        alt = aircraft_pos[2]

        # Altitude check
        if alt < self.min_altitude or alt > self.max_altitude:
            return False

        # Range check (modified by RCS)
        effective_range = self.detection_range * np.sqrt(rcs)
        return dist < effective_range

    def can_engage(self, aircraft_pos: np.ndarray) -> bool:
        """Check if threat can engage aircraft."""
        if not self.active or not self.tracking:
            return False

        dist = np.linalg.norm(aircraft_pos - self.position)
        alt = aircraft_pos[2]

        return (dist < self.engagement_range and
                self.min_altitude <= alt <= self.max_altitude and
                self.time_tracking >= self.reaction_time)

    def compute_pk(self, aircraft_pos: np.ndarray, aircraft_vel: np.ndarray) -> float:
        """Compute probability of kill if engaged."""
        if not self.can_engage(aircraft_pos):
            return 0.0

        dist = np.linalg.norm(aircraft_pos - self.position)
        speed = np.linalg.norm(aircraft_vel)

        # PK decreases with range
        range_factor = 1.0 - (dist / self.engagement_range) ** 2

        # PK decreases with speed (harder to hit fast targets)
        speed_factor = 1.0 / (1.0 + speed / 300.0)

        # Maneuvering penalty (based on velocity changes) - simplified
        maneuver_factor = 0.8

        return self.pk_base * range_factor * speed_factor * maneuver_factor


@dataclass
class GroundTarget:
    """Ground target for strike mission."""
    position: np.ndarray        # [x, y, z] position (z=0 for ground)
    target_type: str            # "vehicle", "structure", "radar", etc.
    value: float                # Target value for scoring
    hardness: float             # Damage resistance (0-1)
    destroyed: bool = False
    damage: float = 0.0         # Current damage level (0-1)

    def apply_damage(self, weapon_damage: float) -> bool:
        """Apply damage and check if destroyed."""
        effective_damage = weapon_damage * (1.0 - self.hardness * 0.5)
        self.damage = min(1.0, self.damage + effective_damage)

        if self.damage >= 1.0:
            self.destroyed = True

        return self.destroyed


@dataclass
class Weapon:
    """Weapon configuration."""
    name: str
    quantity: int
    damage: float               # Base damage (0-1)
    cep: float                  # Circular error probable [m]
    min_release_alt: float      # Minimum release altitude [m]
    max_release_range: float    # Maximum release range [m]
    guidance: str               # "gps", "laser", "eo_ir", "unguided"


class AirToGroundEnv(BaseFixedWingEnv):
    """
    Air-to-ground strike mission environment.

    The agent must:
    1. Navigate to target area while avoiding threats
    2. Identify and engage ground targets
    3. Egress safely after weapon delivery
    4. Return to base with mission success

    Observation includes:
    - Base fixed-wing state (21 dims)
    - Nearest threat info (4 dims)
    - Target info (4 dims)
    - Mission state (3 dims)
    Total: 32 dimensions

    Action space: Same as BaseFixedWingEnv + weapon release
    """

    def __init__(
        self,
        platform_config: Dict[str, Any],
        num_targets: int = 3,
        threat_density: str = "medium",
        terrain_type: str = "flat",
        **kwargs
    ):
        """
        Initialize air-to-ground environment.

        Args:
            platform_config: Platform configuration
            num_targets: Number of ground targets
            threat_density: "low", "medium", "high"
            terrain_type: "flat", "hilly", "mountainous"
        """
        super().__init__(platform_config, **kwargs)

        self.num_targets = num_targets
        self.threat_density = threat_density
        self.terrain_type = terrain_type

        # Mission state
        self._mission_phase = MissionPhase.INGRESS
        self._targets: List[GroundTarget] = []
        self._threats: List[Threat] = []
        self._targets_destroyed = 0
        self._time_detected = 0.0
        self._engaged_count = 0

        # Weapon loadout
        self._weapons = self._create_weapon_loadout()
        self._weapon_state = WeaponState.SAFE

        # Base position for RTB
        self._base_position = np.array([0.0, 0.0, 0.0])

        # Target area
        self._target_area_center = np.array([15000.0, 0.0, 0.0])
        self._target_area_radius = 3000.0

        # RCS from platform config (for stealth)
        physics = platform_config.get('physics_params', {})
        self._rcs = physics.get('rcs', 1.0)

        # Extended observation space (32 dims)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32
        )

        # Extended action space: add weapon release
        # [aileron, elevator, rudder, throttle, flaps, weapon_release]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def _create_weapon_loadout(self) -> List[Weapon]:
        """Create weapon loadout based on platform."""
        # Default loadout for UCAV
        return [
            Weapon(
                name="JDAM",
                quantity=2,
                damage=0.9,
                cep=10.0,
                min_release_alt=500.0,
                max_release_range=15000.0,
                guidance="gps"
            ),
            Weapon(
                name="SDB",
                quantity=4,
                damage=0.6,
                cep=5.0,
                min_release_alt=300.0,
                max_release_range=70000.0,
                guidance="gps"
            ),
        ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with mission setup."""
        # Reset base environment
        obs_array, info = super().reset(seed=seed, options=options)

        # Reset mission state
        self._mission_phase = MissionPhase.INGRESS
        self._targets_destroyed = 0
        self._time_detected = 0.0
        self._engaged_count = 0
        self._weapon_state = WeaponState.SAFE

        # Reset weapons
        self._weapons = self._create_weapon_loadout()

        # Generate targets
        self._generate_targets()

        # Generate threats
        self._generate_threats()

        # Initial position (further from target area)
        self._position = np.array([-5000.0, 0.0, 3000.0])
        self._velocity = np.array([150.0, 0.0, 0.0])

        # Get extended observation
        obs = self._get_extended_observation()

        info.update({
            'mission_phase': self._mission_phase.name,
            'targets_remaining': self.num_targets - self._targets_destroyed,
            'threats_active': sum(1 for t in self._threats if t.active),
        })

        return obs, info

    def _generate_targets(self):
        """Generate ground targets in target area."""
        self._targets = []

        for i in range(self.num_targets):
            # Random position within target area
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(0, self._target_area_radius * 0.8)

            pos = self._target_area_center.copy()
            pos[0] += radius * np.cos(angle)
            pos[1] += radius * np.sin(angle)
            pos[2] = 0  # Ground level

            target_type = self.np_random.choice(
                ["vehicle", "structure", "radar", "command"],
                p=[0.4, 0.3, 0.2, 0.1]
            )

            # Value and hardness based on type
            values = {"vehicle": 50, "structure": 100, "radar": 150, "command": 200}
            hardness = {"vehicle": 0.2, "structure": 0.5, "radar": 0.3, "command": 0.4}

            self._targets.append(GroundTarget(
                position=pos,
                target_type=target_type,
                value=values[target_type],
                hardness=hardness[target_type],
            ))

    def _generate_threats(self):
        """Generate air defense threats."""
        self._threats = []

        # Number of threats based on density
        threat_counts = {
            "low": {"SAM_MEDIUM": 1, "SAM_SHORT": 1, "AAA": 2},
            "medium": {"SAM_MEDIUM": 2, "SAM_SHORT": 2, "AAA": 4},
            "high": {"SAM_LONG": 1, "SAM_MEDIUM": 3, "SAM_SHORT": 3, "AAA": 6},
        }

        counts = threat_counts.get(self.threat_density, threat_counts["medium"])

        # Threat parameters
        threat_params = {
            "SAM_LONG": {
                "detection_range": 150000, "engagement_range": 100000,
                "min_altitude": 100, "max_altitude": 25000,
                "reaction_time": 15.0, "pk_base": 0.7
            },
            "SAM_MEDIUM": {
                "detection_range": 80000, "engagement_range": 40000,
                "min_altitude": 50, "max_altitude": 20000,
                "reaction_time": 10.0, "pk_base": 0.6
            },
            "SAM_SHORT": {
                "detection_range": 30000, "engagement_range": 12000,
                "min_altitude": 20, "max_altitude": 10000,
                "reaction_time": 5.0, "pk_base": 0.5
            },
            "AAA": {
                "detection_range": 8000, "engagement_range": 4000,
                "min_altitude": 0, "max_altitude": 3000,
                "reaction_time": 2.0, "pk_base": 0.3
            },
        }

        for threat_name, count in counts.items():
            params = threat_params.get(threat_name, threat_params["AAA"])
            threat_type = ThreatType[threat_name] if threat_name in ThreatType.__members__ else ThreatType.AAA

            for _ in range(count):
                # Position around target area
                angle = self.np_random.uniform(0, 2 * np.pi)
                radius = self.np_random.uniform(
                    self._target_area_radius * 0.5,
                    self._target_area_radius * 2.0
                )

                pos = self._target_area_center.copy()
                pos[0] += radius * np.cos(angle)
                pos[1] += radius * np.sin(angle)
                pos[2] = 0

                self._threats.append(Threat(
                    position=pos,
                    threat_type=threat_type,
                    **params
                ))

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with strike mission logic."""
        # Extract weapon release from action
        weapon_release = action[5] > 0.5 if len(action) > 5 else False

        # Execute base physics step (first 5 actions)
        base_action = action[:5]

        # Run physics
        self._current_step += 1

        # Parse action
        aileron_cmd = float(base_action[0])
        elevator_cmd = float(base_action[1])
        rudder_cmd = float(base_action[2])
        throttle = float(base_action[3])
        flaps_cmd = float(base_action[4])

        # Mix control surfaces
        surfaces = self._mixer.mix(
            roll_cmd=aileron_cmd,
            pitch_cmd=elevator_cmd,
            yaw_cmd=rudder_cmd,
            throttle=throttle,
            flap_cmd=flaps_cmd,
            dt=1.0 / self.control_hz
        )

        # Store current surfaces
        self._current_surfaces = np.array([
            surfaces.aileron, surfaces.elevator,
            surfaces.rudder, surfaces.flaps
        ])
        self._throttle = throttle

        # Run physics simulation
        for _ in range(self.physics_steps_per_control):
            self._physics_step(surfaces, throttle)

        # Update mission state
        self._update_mission_phase()

        # Update threat tracking
        self._update_threats()

        # Handle weapon release
        if weapon_release and self._weapon_state == WeaponState.ARMED:
            self._attempt_weapon_release()

        # Get extended observation
        obs = self._get_extended_observation()

        # Compute reward
        reward = self._compute_strike_reward(action)

        # Check termination
        terminated = self._check_strike_terminated()
        truncated = self._current_step >= self._max_episode_steps

        # Mission info
        info = {
            'mission_phase': self._mission_phase.name,
            'targets_destroyed': self._targets_destroyed,
            'targets_remaining': self.num_targets - self._targets_destroyed,
            'threats_active': sum(1 for t in self._threats if t.active),
            'time_detected': self._time_detected,
            'weapons_remaining': sum(w.quantity for w in self._weapons),
            'fuel_fraction': self._fuel_remaining / self._fuel_capacity,
        }

        return obs, reward, terminated, truncated, info

    def _update_mission_phase(self):
        """Update mission phase based on position and status."""
        dist_to_target_area = np.linalg.norm(
            self._position[:2] - self._target_area_center[:2]
        )
        dist_to_base = np.linalg.norm(self._position[:2] - self._base_position[:2])

        if self._mission_phase == MissionPhase.INGRESS:
            if dist_to_target_area < self._target_area_radius:
                self._mission_phase = MissionPhase.ATTACK
                self._weapon_state = WeaponState.ARMED

        elif self._mission_phase == MissionPhase.ATTACK:
            # Transition to egress when targets destroyed or out of weapons
            all_targets_done = self._targets_destroyed >= self.num_targets
            out_of_weapons = sum(w.quantity for w in self._weapons) == 0

            if all_targets_done or out_of_weapons:
                self._mission_phase = MissionPhase.EGRESS
                self._weapon_state = WeaponState.SAFE

            # Also egress if leaving target area
            if dist_to_target_area > self._target_area_radius * 1.5:
                self._mission_phase = MissionPhase.EGRESS

        elif self._mission_phase == MissionPhase.EGRESS:
            # RTB when clear of target area
            if dist_to_target_area > self._target_area_radius * 3:
                self._mission_phase = MissionPhase.RTB

        elif self._mission_phase == MissionPhase.RTB:
            if dist_to_base < 1000:
                self._mission_phase = MissionPhase.COMPLETE

    def _update_threats(self):
        """Update threat detection and tracking status."""
        dt = 1.0 / self.control_hz
        detected_this_step = False

        for threat in self._threats:
            if not threat.active:
                continue

            can_detect = threat.can_detect(self._position, self._rcs)

            if can_detect:
                detected_this_step = True
                if not threat.tracking:
                    threat.tracking = True
                    threat.time_tracking = 0.0
                else:
                    threat.time_tracking += dt

                # Check for engagement
                if threat.can_engage(self._position):
                    pk = threat.compute_pk(self._position, self._velocity)
                    # Probabilistic engagement check
                    if self.np_random.random() < pk * dt * 0.1:
                        self._engaged_count += 1
            else:
                threat.tracking = False
                threat.time_tracking = 0.0

        if detected_this_step:
            self._time_detected += dt

    def _attempt_weapon_release(self):
        """Attempt to release weapon on nearest valid target."""
        if self._weapon_state != WeaponState.ARMED:
            return

        # Find best weapon and target
        best_target = None
        best_weapon = None
        min_dist = float('inf')

        for target in self._targets:
            if target.destroyed:
                continue

            dist = np.linalg.norm(self._position - target.position)

            for weapon in self._weapons:
                if weapon.quantity <= 0:
                    continue

                # Check release conditions
                if (self._position[2] >= weapon.min_release_alt and
                    dist <= weapon.max_release_range):

                    if dist < min_dist:
                        min_dist = dist
                        best_target = target
                        best_weapon = weapon

        if best_target and best_weapon:
            # Release weapon
            best_weapon.quantity -= 1

            # Compute hit probability based on CEP
            miss_dist = self.np_random.normal(0, best_weapon.cep)
            hit = abs(miss_dist) < best_weapon.cep * 2

            if hit:
                destroyed = best_target.apply_damage(best_weapon.damage)
                if destroyed:
                    self._targets_destroyed += 1

            self._weapon_state = WeaponState.RELEASED

            # Re-arm if more weapons available
            if sum(w.quantity for w in self._weapons) > 0:
                self._weapon_state = WeaponState.ARMED

    def _get_extended_observation(self) -> np.ndarray:
        """Get observation with mission-specific info."""
        # Base observation (21 dims)
        base_obs = self._get_observation()
        base_array = base_obs.to_array()

        # Nearest threat info (4 dims)
        threat_info = self._get_nearest_threat_info()

        # Nearest target info (4 dims)
        target_info = self._get_nearest_target_info()

        # Mission state (3 dims)
        mission_state = np.array([
            float(self._mission_phase.value) / 5.0,  # Normalized phase
            float(self._targets_destroyed) / max(self.num_targets, 1),
            float(sum(w.quantity for w in self._weapons)) / 6.0,  # Normalized weapons
        ])

        return np.concatenate([
            base_array,      # 21
            threat_info,     # 4
            target_info,     # 4
            mission_state,   # 3
        ]).astype(np.float32)  # Total: 32

    def _get_nearest_threat_info(self) -> np.ndarray:
        """Get info about nearest active threat."""
        min_dist = float('inf')
        nearest = None

        for threat in self._threats:
            if not threat.active:
                continue
            dist = np.linalg.norm(self._position - threat.position)
            if dist < min_dist:
                min_dist = dist
                nearest = threat

        if nearest is None:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Relative position (normalized)
        rel_pos = (nearest.position - self._position) / 10000.0

        # Threat level (0-1)
        in_range = float(min_dist < nearest.engagement_range)

        return np.array([rel_pos[0], rel_pos[1], min_dist / 50000.0, in_range])

    def _get_nearest_target_info(self) -> np.ndarray:
        """Get info about nearest undestroyed target."""
        min_dist = float('inf')
        nearest = None

        for target in self._targets:
            if target.destroyed:
                continue
            dist = np.linalg.norm(self._position - target.position)
            if dist < min_dist:
                min_dist = dist
                nearest = target

        if nearest is None:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Relative position (normalized)
        rel_pos = (nearest.position - self._position) / 10000.0

        # Can engage (0-1)
        can_engage = float(any(
            w.quantity > 0 and
            self._position[2] >= w.min_release_alt and
            min_dist <= w.max_release_range
            for w in self._weapons
        ))

        return np.array([rel_pos[0], rel_pos[1], min_dist / 20000.0, can_engage])

    def _compute_strike_reward(self, action: np.ndarray) -> float:
        """Compute reward for strike mission."""
        reward = 0.0

        # Survival bonus
        reward += 1.0

        # Phase-specific rewards
        if self._mission_phase == MissionPhase.INGRESS:
            # Reward progress toward target
            dist_to_target = np.linalg.norm(
                self._position[:2] - self._target_area_center[:2]
            )
            reward += 0.001 * (20000 - dist_to_target) / 20000

            # Penalty for detection
            reward -= 0.5 * self._time_detected / 60.0

        elif self._mission_phase == MissionPhase.ATTACK:
            # Big reward for target destruction
            reward += 50.0 * self._targets_destroyed

            # Penalty for time in threat envelopes
            for threat in self._threats:
                if threat.tracking:
                    reward -= 0.2

        elif self._mission_phase == MissionPhase.EGRESS:
            # Reward distance from target area
            dist_from_target = np.linalg.norm(
                self._position[:2] - self._target_area_center[:2]
            )
            reward += 0.001 * dist_from_target / 10000

        elif self._mission_phase == MissionPhase.RTB:
            # Reward progress to base
            dist_to_base = np.linalg.norm(self._position[:2] - self._base_position[:2])
            reward += 0.001 * (20000 - dist_to_base) / 20000

        elif self._mission_phase == MissionPhase.COMPLETE:
            # Mission complete bonus
            reward += 100.0
            reward += 20.0 * self._targets_destroyed

        # Penalties
        # Engaged by threats
        reward -= 10.0 * self._engaged_count

        # Low altitude (AAA risk)
        if self._position[2] < 500:
            reward -= 0.5

        # Action smoothness
        reward -= 0.01 * np.sum(action[:3] ** 2)

        return reward

    def _check_strike_terminated(self) -> bool:
        """Check if strike mission should terminate."""
        # Base termination conditions
        if super()._check_terminated():
            return True

        # Shot down (too many engagements)
        if self._engaged_count >= 3:
            return True

        # Mission complete
        if self._mission_phase == MissionPhase.COMPLETE:
            return True

        return False

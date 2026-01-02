"""
Combat Maneuvers for Fixed-Wing UAV Dogfighting

Defines specific maneuver sequences that can be executed and detected.
Each maneuver has:
- Required control inputs
- Duration
- Detection criteria based on orientation/velocity changes
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Callable


class ManeuverType(Enum):
    """Available combat maneuvers."""
    NONE = "none"

    # Basic maneuvers
    LEVEL_TURN = "level_turn"
    CLIMB = "climb"
    DIVE = "dive"

    # Offensive maneuvers
    HIGH_YO_YO = "high_yo_yo"        # Climb then dive onto target
    LOW_YO_YO = "low_yo_yo"          # Dive then climb to reduce closure
    LAG_PURSUIT = "lag_pursuit"      # Follow behind target's turn
    LEAD_PURSUIT = "lead_pursuit"    # Cut inside target's turn

    # Defensive maneuvers
    BREAK_TURN = "break_turn"        # Hard turn to evade
    BARREL_ROLL = "barrel_roll"      # Roll while maintaining heading
    SPLIT_S = "split_s"              # Roll inverted then dive
    IMMELMANN = "immelmann"          # Half loop up, roll to level
    SCISSORS = "scissors"            # Series of reversing turns

    # Energy maneuvers
    ZOOM_CLIMB = "zoom_climb"        # Trade speed for altitude
    POWER_DIVE = "power_dive"        # Trade altitude for speed
    CHANDELLE = "chandelle"          # Climbing turn 180 degrees

    # Attack runs
    GUN_RUN = "gun_run"              # Steady attack approach
    MISSILE_LOCK = "missile_lock"    # Holding lock for missile
    STRAFING_PASS = "strafing_pass"  # High-speed attack pass


@dataclass
class ManeuverState:
    """Current maneuver execution state."""
    maneuver: ManeuverType = ManeuverType.NONE
    start_time: float = 0.0
    duration: float = 0.0
    phase: int = 0  # For multi-phase maneuvers
    initial_orientation: Optional[np.ndarray] = None
    initial_velocity: Optional[np.ndarray] = None
    initial_position: Optional[np.ndarray] = None
    target_id: Optional[int] = None


@dataclass
class ManeuverParams:
    """Parameters for executing a maneuver."""
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    throttle: float = 0.7
    duration: float = 2.0
    phases: List[dict] = field(default_factory=list)


class ManeuverController:
    """
    Controls maneuver execution and detection for a single drone.

    This system:
    1. Executes maneuvers by providing control inputs over time
    2. Detects when a maneuver is being performed based on flight state
    3. Tracks maneuver history for scoring/display
    """

    def __init__(self):
        self.current_state = ManeuverState()
        self.history: List[Tuple[float, ManeuverType]] = []

        # Orientation history for detection (last N samples)
        self.orientation_history: List[np.ndarray] = []
        self.velocity_history: List[np.ndarray] = []
        self.history_size = 30  # ~0.5 seconds at 60Hz

    def start_maneuver(
        self,
        maneuver: ManeuverType,
        current_time: float,
        orientation: np.ndarray,
        velocity: np.ndarray,
        position: np.ndarray,
        target_id: Optional[int] = None,
    ):
        """Start executing a specific maneuver."""
        self.current_state = ManeuverState(
            maneuver=maneuver,
            start_time=current_time,
            duration=self._get_maneuver_duration(maneuver),
            phase=0,
            initial_orientation=orientation.copy(),
            initial_velocity=velocity.copy(),
            initial_position=position.copy(),
            target_id=target_id,
        )
        self.history.append((current_time, maneuver))

    def _get_maneuver_duration(self, maneuver: ManeuverType) -> float:
        """Get expected duration for a maneuver."""
        durations = {
            ManeuverType.LEVEL_TURN: 3.0,
            ManeuverType.CLIMB: 2.0,
            ManeuverType.DIVE: 2.0,
            ManeuverType.HIGH_YO_YO: 4.0,
            ManeuverType.LOW_YO_YO: 4.0,
            ManeuverType.BREAK_TURN: 2.5,
            ManeuverType.BARREL_ROLL: 3.0,
            ManeuverType.SPLIT_S: 3.5,
            ManeuverType.IMMELMANN: 4.0,
            ManeuverType.SCISSORS: 5.0,
            ManeuverType.ZOOM_CLIMB: 3.0,
            ManeuverType.POWER_DIVE: 2.5,
            ManeuverType.CHANDELLE: 4.0,
            ManeuverType.GUN_RUN: 2.0,
            ManeuverType.MISSILE_LOCK: 3.0,
            ManeuverType.STRAFING_PASS: 2.0,
        }
        return durations.get(maneuver, 2.0)

    def get_maneuver_controls(
        self,
        current_time: float,
        orientation: np.ndarray,
        velocity: np.ndarray,
        position: np.ndarray,
        target_pos: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        Get control inputs for the current maneuver.

        Returns:
            Tuple of (roll_cmd, pitch_cmd, throttle) in [-1, 1] range
        """
        if self.current_state.maneuver == ManeuverType.NONE:
            return 0.0, 0.0, 0.7

        elapsed = current_time - self.current_state.start_time

        # Check if maneuver complete
        if elapsed >= self.current_state.duration:
            self.current_state.maneuver = ManeuverType.NONE
            return 0.0, 0.0, 0.7

        # Progress through maneuver (0 to 1)
        progress = elapsed / self.current_state.duration

        return self._execute_maneuver(
            self.current_state.maneuver,
            progress,
            orientation,
            velocity,
            position,
            target_pos,
        )

    def _execute_maneuver(
        self,
        maneuver: ManeuverType,
        progress: float,
        orientation: np.ndarray,
        velocity: np.ndarray,
        position: np.ndarray,
        target_pos: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """Execute a specific maneuver phase."""

        if maneuver == ManeuverType.BARREL_ROLL:
            # Full barrel roll - continuous roll with slight pitch
            roll = 1.0  # Full roll rate
            pitch = 0.2 * np.sin(progress * 2 * np.pi)  # Oscillating pitch
            throttle = 0.8
            return roll, pitch, throttle

        elif maneuver == ManeuverType.SPLIT_S:
            # Phase 1: Roll inverted (0-30%)
            # Phase 2: Pull through (30-100%)
            if progress < 0.3:
                return 1.0, 0.0, 0.6  # Roll to inverted
            else:
                return 0.0, -1.0, 0.5  # Pull through dive

        elif maneuver == ManeuverType.IMMELMANN:
            # Phase 1: Pull up into half loop (0-60%)
            # Phase 2: Roll to level (60-100%)
            if progress < 0.6:
                return 0.0, 1.0, 1.0  # Full pull up
            else:
                return 1.0, 0.0, 0.7  # Roll to level

        elif maneuver == ManeuverType.BREAK_TURN:
            # Hard turn - maximum bank and pull
            return 1.0, 0.8, 1.0  # Full deflection turn

        elif maneuver == ManeuverType.HIGH_YO_YO:
            # Phase 1: Climb (0-40%)
            # Phase 2: Roll and dive onto target (40-100%)
            if progress < 0.4:
                return 0.3, 0.8, 1.0  # Slight bank, strong pull
            else:
                return 0.6, -0.4, 0.6  # Roll over and dive

        elif maneuver == ManeuverType.LOW_YO_YO:
            # Phase 1: Dive (0-40%)
            # Phase 2: Pull up (40-100%)
            if progress < 0.4:
                return 0.3, -0.5, 0.5  # Dive
            else:
                return 0.2, 0.6, 1.0  # Pull up

        elif maneuver == ManeuverType.SCISSORS:
            # Reversing turns
            phase = int(progress * 4) % 2
            direction = 1.0 if phase == 0 else -1.0
            return direction * 0.9, 0.3, 0.7

        elif maneuver == ManeuverType.ZOOM_CLIMB:
            return 0.0, 0.9, 1.0  # Steep climb

        elif maneuver == ManeuverType.POWER_DIVE:
            return 0.0, -0.7, 0.3  # Steep dive, low throttle

        elif maneuver == ManeuverType.CHANDELLE:
            # Climbing turn 180 degrees
            if progress < 0.7:
                return 0.6, 0.5, 1.0  # Bank and climb
            else:
                return 0.2, 0.2, 0.7  # Level off

        elif maneuver == ManeuverType.GUN_RUN:
            # Steady approach to target
            if target_pos is not None:
                return self._track_target(orientation, velocity, position, target_pos)
            return 0.0, 0.0, 0.8

        elif maneuver == ManeuverType.MISSILE_LOCK:
            # Maintain steady pursuit for lock
            if target_pos is not None:
                roll, pitch, _ = self._track_target(orientation, velocity, position, target_pos)
                return roll * 0.5, pitch * 0.5, 0.6  # Gentler tracking
            return 0.0, 0.0, 0.7

        # Default
        return 0.0, 0.0, 0.7

    def _track_target(
        self,
        orientation: np.ndarray,
        velocity: np.ndarray,
        position: np.ndarray,
        target_pos: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Generate controls to track a target."""
        to_target = target_pos - position
        distance = np.linalg.norm(to_target)
        if distance < 1.0:
            return 0.0, 0.0, 0.7

        to_target_norm = to_target / distance

        yaw = orientation[2]
        pitch = orientation[1]
        heading = np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch)
        ])

        # Cross product gives turn direction
        cross = np.cross(heading, to_target_norm)

        # Pitch to target
        alt_diff = target_pos[2] - position[2]
        pitch_cmd = np.clip(alt_diff / 200, -0.5, 0.5)

        # Roll to turn toward target
        roll_cmd = np.clip(cross[2] * 2, -1, 1)

        # Throttle based on distance
        throttle = 0.6 if distance < 300 else 0.9

        return roll_cmd, pitch_cmd, throttle

    def update_history(self, orientation: np.ndarray, velocity: np.ndarray):
        """Update flight state history for maneuver detection."""
        self.orientation_history.append(orientation.copy())
        self.velocity_history.append(velocity.copy())

        if len(self.orientation_history) > self.history_size:
            self.orientation_history.pop(0)
            self.velocity_history.pop(0)

    def detect_maneuver(self) -> ManeuverType:
        """
        Detect what maneuver is being performed based on flight state history.

        Returns:
            Detected maneuver type
        """
        if len(self.orientation_history) < 10:
            return ManeuverType.NONE

        # Calculate changes over recent history
        recent_orient = np.array(self.orientation_history[-10:])
        recent_vel = np.array(self.velocity_history[-10:])

        # Roll rate (change in roll over time)
        roll_changes = np.diff(recent_orient[:, 0])
        avg_roll_rate = np.mean(np.abs(roll_changes)) * 60  # Per second at 60Hz

        # Pitch rate
        pitch_changes = np.diff(recent_orient[:, 1])
        avg_pitch_rate = np.mean(pitch_changes) * 60

        # Altitude change
        alt_changes = np.diff([v[2] for v in self.velocity_history[-10:]])
        avg_alt_rate = np.mean(recent_vel[:, 2])

        # Speed changes
        speeds = np.linalg.norm(recent_vel, axis=1)
        speed_change = speeds[-1] - speeds[0]

        # Current orientation
        current_roll = recent_orient[-1, 0]
        current_pitch = recent_orient[-1, 1]

        # Detection logic

        # Barrel Roll: High roll rate, low net orientation change
        if avg_roll_rate > 2.5:  # Fast rolling
            total_roll = np.sum(roll_changes)
            if abs(total_roll) > 5.0:  # Accumulated significant roll
                return ManeuverType.BARREL_ROLL

        # Split-S: Inverted (roll ~180) then diving
        if abs(current_roll) > 2.5 and avg_pitch_rate < -1.0:
            return ManeuverType.SPLIT_S

        # Immelmann: Strong pitch up with roll at end
        if avg_pitch_rate > 1.5 and current_pitch > 0.8:
            if avg_roll_rate > 1.0:
                return ManeuverType.IMMELMANN
            return ManeuverType.ZOOM_CLIMB

        # Break Turn: High bank angle with pull
        if abs(current_roll) > 1.0 and avg_pitch_rate > 0.5:
            return ManeuverType.BREAK_TURN

        # High/Low Yo-Yo: Significant altitude change with pursuit
        if abs(avg_alt_rate) > 30:  # Rapid altitude change
            if avg_alt_rate > 30:
                return ManeuverType.HIGH_YO_YO
            else:
                return ManeuverType.LOW_YO_YO

        # Power Dive: Strong negative pitch
        if avg_pitch_rate < -1.0:
            return ManeuverType.POWER_DIVE

        # Zoom Climb: Strong positive pitch
        if avg_pitch_rate > 1.0:
            return ManeuverType.ZOOM_CLIMB

        # Level Turn: Bank without much pitch
        if abs(current_roll) > 0.5 and abs(avg_pitch_rate) < 0.3:
            return ManeuverType.LEVEL_TURN

        # Climb/Dive based on velocity
        if avg_alt_rate > 20:
            return ManeuverType.CLIMB
        elif avg_alt_rate < -20:
            return ManeuverType.DIVE

        return ManeuverType.NONE

    @property
    def current_maneuver_name(self) -> str:
        """Get display name of current maneuver."""
        if self.current_state.maneuver != ManeuverType.NONE:
            return self.current_state.maneuver.value.replace("_", " ").upper()
        detected = self.detect_maneuver()
        if detected != ManeuverType.NONE:
            return detected.value.replace("_", " ").upper()
        return ""


class CombatStyle(Enum):
    """Different combat fighting styles."""
    AGGRESSIVE = "aggressive"    # Close combat, guns, break turns
    DEFENSIVE = "defensive"      # Evasive, yo-yos, missiles from range
    ACROBATIC = "acrobatic"      # Barrel rolls, Immelmanns, flashy moves
    TACTICAL = "tactical"        # Balanced, energy management


# Pre-defined style configs
STYLE_CONFIGS = {
    CombatStyle.AGGRESSIVE: {
        'preferred_distance': 250,
        'preferred_weapon': 0,  # Gun
        'favorite_maneuvers': [ManeuverType.GUN_RUN, ManeuverType.BREAK_TURN, ManeuverType.LEAD_PURSUIT],
        'aggression': 0.9,
        'decision_interval': 0.3,  # Fast decisions
    },
    CombatStyle.DEFENSIVE: {
        'preferred_distance': 700,
        'preferred_weapon': 1,  # Missile
        'favorite_maneuvers': [ManeuverType.SPLIT_S, ManeuverType.HIGH_YO_YO, ManeuverType.MISSILE_LOCK],
        'aggression': 0.4,
        'decision_interval': 0.6,
    },
    CombatStyle.ACROBATIC: {
        'preferred_distance': 400,
        'preferred_weapon': 0,
        'favorite_maneuvers': [ManeuverType.BARREL_ROLL, ManeuverType.IMMELMANN, ManeuverType.CHANDELLE],
        'aggression': 0.7,
        'decision_interval': 0.4,
    },
    CombatStyle.TACTICAL: {
        'preferred_distance': 500,
        'preferred_weapon': 0,
        'favorite_maneuvers': [ManeuverType.LAG_PURSUIT, ManeuverType.ZOOM_CLIMB, ManeuverType.LEVEL_TURN],
        'aggression': 0.6,
        'decision_interval': 0.5,
    },
}


class CombatAI:
    """
    Combat AI that selects and executes maneuvers based on tactical situation.

    Each drone can have a unique combat style that influences its behavior.
    """

    def __init__(self, drone_id: int, team: int, style: Optional[CombatStyle] = None):
        self.drone_id = drone_id
        self.team = team
        self.maneuver_controller = ManeuverController()
        self.last_decision_time = 0.0
        self.current_target_id: Optional[int] = None

        # Assign style based on team - both teams get same style for fair fight
        # Red team (0) = AGGRESSIVE, Blue team (1) = ACROBATIC
        if style is None:
            if team == 0:
                style = CombatStyle.AGGRESSIVE  # Red: close combat, guns, break turns
            else:
                style = CombatStyle.ACROBATIC   # Blue: barrel rolls, Immelmanns

        self.style = style
        self.config = STYLE_CONFIGS[style]
        self.aggression = self.config['aggression']
        self.decision_interval = self.config['decision_interval']
        self.preferred_distance = self.config['preferred_distance']
        self.preferred_weapon = self.config['preferred_weapon']
        self.favorite_maneuvers = self.config['favorite_maneuvers']

    def get_action(
        self,
        current_time: float,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_orient: np.ndarray,
        drone_health: float,
        enemies: List[dict],  # List of {id, pos, vel, health}
        allies: List[dict],
    ) -> Tuple[np.ndarray, str]:
        """
        Get combat action - prioritizes pursuit and firing over fancy maneuvers.

        Returns:
            Tuple of (action array, maneuver name)
        """
        # Update maneuver history for detection
        self.maneuver_controller.update_history(drone_orient, drone_vel)

        # No enemies - simple patrol
        if not enemies:
            t = current_time + self.drone_id * 3.14
            roll = 0.3 * np.sin(t * 0.5)
            pitch = 0.1 * np.sin(t * 0.3)
            return np.array([roll, pitch, 0.7, 0, 0, 0]), "PATROL"

        # Find closest enemy
        closest_enemy = min(enemies, key=lambda e: np.linalg.norm(e['pos'] - drone_pos))
        self.current_target_id = closest_enemy['id']
        target_pos = closest_enemy['pos']
        target_vel = closest_enemy['vel']

        # Calculate pursuit geometry
        to_target = target_pos - drone_pos
        distance = np.linalg.norm(to_target)
        to_target_norm = to_target / (distance + 1e-6)

        # Current heading
        yaw = drone_orient[2]
        pitch_angle = drone_orient[1]
        heading = np.array([
            np.cos(yaw) * np.cos(pitch_angle),
            np.sin(yaw) * np.cos(pitch_angle),
            np.sin(pitch_angle)
        ])

        # Alignment with target (1 = pointing at, -1 = pointing away)
        alignment = np.dot(heading, to_target_norm)

        # Cross product for turn direction
        cross = np.cross(heading, to_target_norm)

        # COMBAT-FOCUSED CONTROL
        # Primary goal: Turn toward enemy and close distance

        # Roll to turn toward target
        roll_cmd = np.clip(cross[2] * 3.0, -1, 1)  # Strong roll to face target

        # Pitch to match altitude and pursue
        alt_diff = target_pos[2] - drone_pos[2]
        pitch_cmd = np.clip(alt_diff / 150 + alignment * 0.2, -0.8, 0.8)

        # Throttle based on distance and alignment
        if distance > 500:
            throttle = 1.0  # Full speed to close
        elif distance < 150:
            throttle = 0.5  # Slow down when close
        else:
            throttle = 0.8

        # Add style-specific flair
        maneuver_name = "PURSUIT"
        if self.style == CombatStyle.ACROBATIC and alignment > 0.8:
            # Add some roll for style when on target
            roll_cmd += 0.3 * np.sin(current_time * 5)
            maneuver_name = "ATTACK RUN"
        elif self.style == CombatStyle.AGGRESSIVE:
            # More aggressive pursuit
            roll_cmd *= 1.2
            throttle = min(1.0, throttle + 0.2)
            maneuver_name = "GUN RUN"

        # FIRE DECISION - be more aggressive about firing
        fire = 0
        weapon = 0
        target_idx = 0

        # Find target index
        for i, e in enumerate(enemies):
            if e['id'] == self.current_target_id:
                target_idx = i
                break

        # Fire guns when somewhat aligned and in range
        if alignment > 0.7 and distance < 350:
            fire = 1
            weapon = 0
            maneuver_name = "GUNS GUNS GUNS"
        # Fire missiles at longer range with good alignment
        elif alignment > 0.8 and distance > 300 and distance < 1000:
            fire = 1
            weapon = 1
            maneuver_name = "FOX TWO"

        # SAFETY OVERRIDES
        altitude = drone_pos[2]
        speed = np.linalg.norm(drone_vel)

        # Ground avoidance
        if altitude < 200:
            pitch_cmd = max(pitch_cmd, 0.7)
            maneuver_name = "PULL UP"
        elif altitude > 1400:
            pitch_cmd = min(pitch_cmd, -0.4)

        # Stall prevention
        if speed < 70:
            pitch_cmd = min(pitch_cmd, -0.2)
            throttle = 1.0
            maneuver_name = "RECOVER"

        # BOUNDARY AVOIDANCE - hard turn back if near edge
        arena_limit = 900  # Stay within 900m of center
        dist_from_center = np.sqrt(drone_pos[0]**2 + drone_pos[1]**2)
        if dist_from_center > arena_limit:
            # Turn toward center
            center_dir = -drone_pos[:2] / (dist_from_center + 1e-6)
            heading_2d = heading[:2] / (np.linalg.norm(heading[:2]) + 1e-6)
            cross_center = np.cross(np.append(heading_2d, 0), np.append(center_dir, 0))
            roll_cmd = np.clip(cross_center[2] * 4.0, -1, 1)  # Hard turn to center
            maneuver_name = "RTB"

        roll_cmd = np.clip(roll_cmd, -1, 1)
        pitch_cmd = np.clip(pitch_cmd, -1, 1)

        action = np.array([roll_cmd, pitch_cmd, throttle, weapon, fire, target_idx])
        return action, maneuver_name

    def _make_tactical_decision(
        self,
        current_time: float,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_orient: np.ndarray,
        drone_health: float,
        enemies: List[dict],
        allies: List[dict],
    ):
        """Make a tactical decision using style-specific maneuvers."""

        if not enemies:
            # No enemies - do patrol pattern based on style
            self.current_target_id = None
            self._select_patrol_maneuver(current_time, drone_orient, drone_vel, drone_pos)
            return

        # Select target (closest enemy)
        closest_enemy = min(
            enemies,
            key=lambda e: np.linalg.norm(e['pos'] - drone_pos)
        )
        self.current_target_id = closest_enemy['id']

        target_pos = closest_enemy['pos']
        to_target = target_pos - drone_pos
        distance = np.linalg.norm(to_target)

        # Calculate angles
        yaw = drone_orient[2]
        pitch_angle = drone_orient[1]
        heading = np.array([
            np.cos(yaw) * np.cos(pitch_angle),
            np.sin(yaw) * np.cos(pitch_angle),
            np.sin(pitch_angle)
        ])

        alignment = np.dot(heading, to_target / (distance + 1e-6))
        enemy_behind = alignment < 0
        alt_diff = target_pos[2] - drone_pos[2]

        # Emergency evasion if low health (all styles)
        if drone_health < 30 and not enemy_behind:
            self.maneuver_controller.start_maneuver(
                ManeuverType.SPLIT_S, current_time,
                drone_orient, drone_vel, drone_pos, closest_enemy['id']
            )
            return

        # Style-specific combat behavior
        if self.style == CombatStyle.AGGRESSIVE:
            self._aggressive_tactics(
                current_time, drone_pos, drone_vel, drone_orient,
                closest_enemy, distance, alignment, enemy_behind, alt_diff
            )
        elif self.style == CombatStyle.ACROBATIC:
            self._acrobatic_tactics(
                current_time, drone_pos, drone_vel, drone_orient,
                closest_enemy, distance, alignment, enemy_behind, alt_diff
            )
        elif self.style == CombatStyle.DEFENSIVE:
            self._defensive_tactics(
                current_time, drone_pos, drone_vel, drone_orient,
                closest_enemy, distance, alignment, enemy_behind, alt_diff
            )
        else:  # TACTICAL
            self._tactical_tactics(
                current_time, drone_pos, drone_vel, drone_orient,
                closest_enemy, distance, alignment, enemy_behind, alt_diff
            )

    def _select_patrol_maneuver(self, current_time, orient, vel, pos):
        """Select a patrol maneuver based on style."""
        if self.style == CombatStyle.ACROBATIC:
            self.maneuver_controller.start_maneuver(
                ManeuverType.BARREL_ROLL, current_time, orient, vel, pos, None
            )
        else:
            self.maneuver_controller.start_maneuver(
                ManeuverType.LEVEL_TURN, current_time, orient, vel, pos, None
            )

    def _aggressive_tactics(self, current_time, drone_pos, drone_vel, drone_orient,
                           enemy, distance, alignment, enemy_behind, alt_diff):
        """AGGRESSIVE: Close combat, guns, break turns."""
        target_id = enemy['id']

        if enemy_behind:
            # Hard break turn to get guns on target
            self.maneuver_controller.start_maneuver(
                ManeuverType.BREAK_TURN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alignment > 0.7 and distance < 300:
            # In gun range - commit to gun run
            self.maneuver_controller.start_maneuver(
                ManeuverType.GUN_RUN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif distance > 400:
            # Too far - lead pursuit to close distance fast
            self.maneuver_controller.start_maneuver(
                ManeuverType.LEAD_PURSUIT, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        else:
            # Break turn to get firing solution
            self.maneuver_controller.start_maneuver(
                ManeuverType.BREAK_TURN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )

    def _acrobatic_tactics(self, current_time, drone_pos, drone_vel, drone_orient,
                          enemy, distance, alignment, enemy_behind, alt_diff):
        """ACROBATIC: Barrel rolls, Immelmanns, flashy maneuvers."""
        target_id = enemy['id']

        if enemy_behind:
            # Immelmann to reverse and come back down
            self.maneuver_controller.start_maneuver(
                ManeuverType.IMMELMANN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alignment > 0.8 and distance < 400:
            # Strafing pass with barrel roll entry
            self.maneuver_controller.start_maneuver(
                ManeuverType.BARREL_ROLL, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alt_diff > 100:
            # Enemy above - chandelle to gain altitude with style
            self.maneuver_controller.start_maneuver(
                ManeuverType.CHANDELLE, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alt_diff < -100:
            # Enemy below - split-S to dive on them
            self.maneuver_controller.start_maneuver(
                ManeuverType.SPLIT_S, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        else:
            # Same altitude - Immelmann for repositioning
            self.maneuver_controller.start_maneuver(
                ManeuverType.IMMELMANN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )

    def _defensive_tactics(self, current_time, drone_pos, drone_vel, drone_orient,
                          enemy, distance, alignment, enemy_behind, alt_diff):
        """DEFENSIVE: Evasive, yo-yos, missiles from range."""
        target_id = enemy['id']

        if enemy_behind:
            # Split-S to escape and reset
            self.maneuver_controller.start_maneuver(
                ManeuverType.SPLIT_S, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif distance < 400:
            # Too close - high yo-yo to extend
            self.maneuver_controller.start_maneuver(
                ManeuverType.HIGH_YO_YO, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alignment > 0.7 and distance < 900:
            # Good missile range - lock on
            self.maneuver_controller.start_maneuver(
                ManeuverType.MISSILE_LOCK, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        else:
            # Maintain distance with lag pursuit
            self.maneuver_controller.start_maneuver(
                ManeuverType.LAG_PURSUIT, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )

    def _tactical_tactics(self, current_time, drone_pos, drone_vel, drone_orient,
                         enemy, distance, alignment, enemy_behind, alt_diff):
        """TACTICAL: Balanced, energy management."""
        target_id = enemy['id']

        if enemy_behind:
            if distance < 400:
                self.maneuver_controller.start_maneuver(
                    ManeuverType.BREAK_TURN, current_time,
                    drone_orient, drone_vel, drone_pos, target_id
                )
            else:
                self.maneuver_controller.start_maneuver(
                    ManeuverType.ZOOM_CLIMB, current_time,
                    drone_orient, drone_vel, drone_pos, target_id
                )
        elif alignment > 0.8 and distance < 350:
            self.maneuver_controller.start_maneuver(
                ManeuverType.GUN_RUN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alt_diff > 100:
            self.maneuver_controller.start_maneuver(
                ManeuverType.LOW_YO_YO, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        elif alt_diff < -100:
            self.maneuver_controller.start_maneuver(
                ManeuverType.HIGH_YO_YO, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )
        else:
            self.maneuver_controller.start_maneuver(
                ManeuverType.LEVEL_TURN, current_time,
                drone_orient, drone_vel, drone_pos, target_id
            )

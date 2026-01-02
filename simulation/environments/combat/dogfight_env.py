"""
Fixed-Wing Dogfight Environment

Multi-agent air combat simulation for training autonomous combat drones.
Supports continuous self-play training to discover optimal fighting strategies.

Features:
- Realistic fixed-wing aerodynamics
- Weapon systems (missiles, guns, radar locks)
- Multi-agent combat (1v1, 2v2, swarm vs swarm)
- Self-play with opponent pool
- Combat analytics and strategy extraction
- Continuous training for days/weeks
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Callable
from gymnasium import spaces
import gymnasium as gym

from simulation.environments.combat.base_fixed_wing_env import BaseFixedWingEnv
from simulation.environments.combat.maneuvers import CombatAI, ManeuverType


class WeaponType(Enum):
    """Available weapon types."""
    GUN = "gun"                    # Close range, high fire rate
    MISSILE_IR = "missile_ir"      # Heat-seeking missile
    MISSILE_RADAR = "missile_radar"  # Radar-guided missile
    LASER = "laser"                # Instant hit, limited range


class CombatResult(Enum):
    """Possible combat outcomes."""
    ONGOING = "ongoing"
    KILL = "kill"
    KILLED = "killed"
    MUTUAL_KILL = "mutual_kill"
    TIMEOUT = "timeout"
    OUT_OF_BOUNDS = "out_of_bounds"
    CRASHED = "crashed"


@dataclass
class Weapon:
    """Weapon state and properties."""
    weapon_type: WeaponType
    ammo: int
    max_ammo: int
    cooldown: float = 0.0
    cooldown_time: float = 0.5
    range: float = 500.0
    damage: float = 50.0
    lock_required: bool = False
    lock_time: float = 2.0
    current_lock: float = 0.0

    def can_fire(self) -> bool:
        return self.ammo > 0 and self.cooldown <= 0

    def fire(self) -> bool:
        if self.can_fire():
            self.ammo -= 1
            self.cooldown = self.cooldown_time
            return True
        return False

    def update(self, dt: float):
        self.cooldown = max(0, self.cooldown - dt)


@dataclass
class CombatDrone:
    """State of a drone in combat."""
    drone_id: int
    team: int  # 0 = red, 1 = blue
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray  # roll, pitch, yaw
    angular_velocity: np.ndarray
    health: float = 100.0
    max_health: float = 100.0
    is_alive: bool = True
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    weapons: List[Weapon] = field(default_factory=list)
    locked_target: Optional[int] = None
    locked_by: List[int] = field(default_factory=list)

    def take_damage(self, amount: float) -> bool:
        """Take damage, return True if killed."""
        self.damage_taken += amount
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            self.deaths += 1
            return True
        return False

    def respawn(self, position: np.ndarray, velocity: np.ndarray):
        """Respawn the drone."""
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.health = self.max_health
        self.is_alive = True
        self.locked_target = None
        self.locked_by = []
        for weapon in self.weapons:
            weapon.ammo = weapon.max_ammo
            weapon.cooldown = 0
            weapon.current_lock = 0


@dataclass
class DogfightConfig:
    """Configuration for dogfight scenarios."""
    # Arena
    arena_size: float = 2000.0  # meters
    arena_height_min: float = 100.0
    arena_height_max: float = 2000.0

    # Teams
    num_red: int = 1
    num_blue: int = 1

    # Combat rules
    respawn_enabled: bool = True
    respawn_delay: float = 3.0
    friendly_fire: bool = False
    max_match_time: float = 300.0  # 5 minutes
    win_condition: str = "kills"  # "kills", "last_alive", "score"
    kills_to_win: int = 10

    # Weapons
    weapons_config: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"type": "gun", "ammo": 500, "range": 300, "damage": 10, "cooldown": 0.1},
        {"type": "missile_ir", "ammo": 4, "range": 1000, "damage": 100, "cooldown": 2.0, "lock_time": 1.5},
    ])

    # Physics
    min_speed: float = 50.0  # m/s - stall speed
    max_speed: float = 300.0  # m/s
    max_g_force: float = 9.0

    # Rewards
    reward_kill: float = 100.0
    reward_hit: float = 10.0
    reward_death: float = -50.0
    reward_out_of_bounds: float = -20.0
    reward_per_second: float = -0.1  # Encourages aggression
    reward_pursuit: float = 1.0  # For chasing enemy
    reward_evasion: float = 0.5  # For evading missiles


class DogfightEnv(gym.Env):
    """
    Fixed-wing drone dogfight environment.

    Supports:
    - 1v1 duels to large swarm battles
    - Realistic fixed-wing flight dynamics
    - Multiple weapon systems
    - Self-play training
    - Continuous multi-day training

    Observation Space:
    - Own state: position, velocity, orientation, health, ammo
    - Relative enemy states: position, velocity, threat level
    - Weapon states: cooldowns, locks
    - Tactical info: distance to boundaries, altitude

    Action Space:
    - Roll rate command
    - Pitch rate command
    - Throttle
    - Weapon select
    - Fire trigger
    - Target selection
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: Optional[DogfightConfig] = None,
        agent_team: int = 0,
        agent_id: int = 0,
        opponent_policy: Optional[Callable] = None,
        render_mode: Optional[str] = None,
        fixed_wing_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dogfight environment.

        Args:
            config: Dogfight configuration
            agent_team: Which team the learning agent is on (0=red, 1=blue)
            agent_id: Agent's ID within the team
            opponent_policy: Policy function for opponents (for self-play)
            render_mode: Rendering mode
            fixed_wing_config: Configuration for fixed-wing dynamics
        """
        super().__init__()

        self.config = config or DogfightConfig()
        self.agent_team = agent_team
        self.agent_id = agent_id
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode

        # Fixed-wing configuration
        self.fw_config = fixed_wing_config or {
            "mass": 500.0,  # kg
            "wing_area": 15.0,  # m^2
            "thrust_max": 50000.0,  # N
            "cl_alpha": 5.7,  # lift curve slope
            "cd0": 0.02,  # parasitic drag
            "oswald": 0.8,  # Oswald efficiency
            "aspect_ratio": 8.0,
        }

        # Initialize drones
        self.drones: Dict[int, CombatDrone] = {}
        self.combat_ais: Dict[int, CombatAI] = {}  # Tactical AI for each drone
        self._init_drones()

        # Get the learning agent
        self.agent_drone_id = self._get_agent_drone_id()

        # Combat state
        self.match_time = 0.0
        self.red_kills = 0
        self.blue_kills = 0
        self.combat_log: List[Dict[str, Any]] = []

        # Physics
        self.dt = 1.0 / 60.0  # 60 Hz
        self.gravity = 9.81

        # Define spaces
        self._define_spaces()

        # Statistics tracking
        self.episode_stats = {
            "kills": 0,
            "deaths": 0,
            "damage_dealt": 0,
            "damage_taken": 0,
            "shots_fired": 0,
            "shots_hit": 0,
            "missiles_fired": 0,
            "missiles_hit": 0,
            "time_alive": 0,
            "distance_flown": 0,
        }

    def _init_drones(self):
        """Initialize all drones for combat with tactical AI."""
        drone_id = 0

        # Red team
        for i in range(self.config.num_red):
            spawn_pos = self._get_spawn_position(0, i)
            spawn_vel = np.array([self.config.min_speed * 1.5, 0, 0])

            drone = CombatDrone(
                drone_id=drone_id,
                team=0,
                position=spawn_pos,
                velocity=spawn_vel,
                orientation=np.zeros(3),
                angular_velocity=np.zeros(3),
                weapons=self._create_weapons(),
            )
            self.drones[drone_id] = drone
            # Create tactical AI for this drone
            self.combat_ais[drone_id] = CombatAI(drone_id, team=0)
            drone_id += 1

        # Blue team
        for i in range(self.config.num_blue):
            spawn_pos = self._get_spawn_position(1, i)
            spawn_vel = np.array([-self.config.min_speed * 1.5, 0, 0])

            drone = CombatDrone(
                drone_id=drone_id,
                team=1,
                position=spawn_pos,
                velocity=spawn_vel,
                orientation=np.array([0, 0, np.pi]),  # Facing opposite
                angular_velocity=np.zeros(3),
                weapons=self._create_weapons(),
            )
            self.drones[drone_id] = drone
            # Create tactical AI for this drone
            self.combat_ais[drone_id] = CombatAI(drone_id, team=1)
            drone_id += 1

    def _get_spawn_position(self, team: int, index: int) -> np.ndarray:
        """Get spawn position for a drone."""
        x = -self.config.arena_size / 3 if team == 0 else self.config.arena_size / 3
        y = (index - (self.config.num_red if team == 0 else self.config.num_blue) / 2) * 50
        z = (self.config.arena_height_min + self.config.arena_height_max) / 2
        return np.array([x, y, z])

    def _create_weapons(self) -> List[Weapon]:
        """Create weapon loadout for a drone."""
        weapons = []
        for wc in self.config.weapons_config:
            weapon = Weapon(
                weapon_type=WeaponType(wc["type"]),
                ammo=wc["ammo"],
                max_ammo=wc["ammo"],
                range=wc["range"],
                damage=wc["damage"],
                cooldown_time=wc["cooldown"],
                lock_time=wc.get("lock_time", 0),
                lock_required=wc.get("lock_time", 0) > 0,
            )
            weapons.append(weapon)
        return weapons

    def _get_agent_drone_id(self) -> int:
        """Get the drone ID for the learning agent."""
        team_drones = [d for d in self.drones.values() if d.team == self.agent_team]
        if self.agent_id < len(team_drones):
            return team_drones[self.agent_id].drone_id
        return team_drones[0].drone_id if team_drones else 0

    def _define_spaces(self):
        """Define observation and action spaces."""
        num_enemies = self.config.num_blue if self.agent_team == 0 else self.config.num_red
        num_allies = (self.config.num_red if self.agent_team == 0 else self.config.num_blue) - 1

        # Observation: own state + relative enemy states + weapon states
        # Own state: pos(3) + vel(3) + orient(3) + ang_vel(3) + health(1) + speed(1) = 14
        # Per enemy: rel_pos(3) + rel_vel(3) + threat(1) = 7
        # Per weapon: ammo(1) + cooldown(1) + lock(1) = 3
        # Tactical: altitude(1) + boundary_dist(1) + g_force(1) = 3

        own_state_dim = 14
        enemy_state_dim = 7 * num_enemies
        weapon_state_dim = 3 * len(self.config.weapons_config)
        tactical_dim = 3

        obs_dim = own_state_dim + enemy_state_dim + weapon_state_dim + tactical_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: roll_rate, pitch_rate, throttle, weapon_select, fire, target_select
        # Continuous: roll(-1,1), pitch(-1,1), throttle(0,1)
        # Discrete embedded: weapon(0-n), fire(0,1), target(0-m)

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, len(self.config.weapons_config), 1, num_enemies]),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new match."""
        super().reset(seed=seed)

        # Reset match state
        self.match_time = 0.0
        self.red_kills = 0
        self.blue_kills = 0
        self.combat_log = []

        # Reset all drones
        for drone_id, drone in self.drones.items():
            spawn_pos = self._get_spawn_position(drone.team, drone_id % max(self.config.num_red, self.config.num_blue))
            spawn_vel = np.array([
                self.config.min_speed * 1.5 * (1 if drone.team == 0 else -1),
                0, 0
            ])
            drone.respawn(spawn_pos, spawn_vel)
            drone.orientation = np.array([0, 0, 0 if drone.team == 0 else np.pi])
            drone.kills = 0
            drone.deaths = 0
            drone.damage_dealt = 0
            drone.damage_taken = 0

        # Reset episode stats
        self.episode_stats = {k: 0 for k in self.episode_stats}

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: Agent's action

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get agent drone
        agent = self.drones[self.agent_drone_id]

        # Apply agent's action
        if agent.is_alive:
            self._apply_action(agent, action)

        # Get and apply opponent actions
        self._step_opponents()

        # Update physics for all drones
        for drone in self.drones.values():
            if drone.is_alive:
                self._update_physics(drone)

        # Process combat (weapons, damage)
        combat_events = self._process_combat()

        # Check boundaries and crashes
        for drone in self.drones.values():
            if drone.is_alive:
                self._check_boundaries(drone)
                self._check_altitude(drone)

        # Update match time
        self.match_time += self.dt

        # Calculate reward
        reward = self._calculate_reward(agent, combat_events)

        # Check termination
        terminated, truncated = self._check_termination()

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["combat_events"] = combat_events

        return obs, reward, terminated, truncated, info

    def _apply_action(self, drone: CombatDrone, action: np.ndarray):
        """Apply action to a drone."""
        roll_rate_cmd = action[0] * np.radians(180)  # Max 180 deg/s
        pitch_rate_cmd = action[1] * np.radians(90)  # Max 90 deg/s
        throttle = np.clip(action[2], 0, 1)
        weapon_idx = int(np.clip(action[3], 0, len(drone.weapons) - 1))
        fire = action[4] > 0.5
        target_idx = int(np.clip(action[5], 0, self._get_num_enemies() - 1))

        # Apply flight controls
        drone.angular_velocity[0] = roll_rate_cmd
        drone.angular_velocity[1] = pitch_rate_cmd

        # Store throttle for physics
        drone._throttle = throttle
        drone._selected_weapon = weapon_idx
        drone._fire_command = fire
        drone._target_idx = target_idx

    def _step_opponents(self):
        """Step all opponent drones using tactical CombatAI."""
        for drone in self.drones.values():
            if drone.drone_id == self.agent_drone_id:
                continue
            if not drone.is_alive:
                continue

            if self.opponent_policy is not None:
                # Use provided RL policy
                obs = self._get_observation_for_drone(drone)
                action, _ = self.opponent_policy.predict(obs, deterministic=False)
            else:
                # Use tactical CombatAI for realistic dogfighting
                action, maneuver_name = self._get_tactical_action(drone)
                # Store current maneuver for visualization
                drone._current_maneuver = maneuver_name

            self._apply_action(drone, action)

    def _get_tactical_action(self, drone: CombatDrone) -> Tuple[np.ndarray, str]:
        """Get tactical action from CombatAI for a drone."""
        combat_ai = self.combat_ais.get(drone.drone_id)
        if combat_ai is None:
            return self._simple_ai_action(drone), ""

        # Gather enemy and ally data
        enemies = []
        allies = []
        for d in self.drones.values():
            if not d.is_alive:
                continue
            data = {
                'id': d.drone_id,
                'pos': d.position.copy(),
                'vel': d.velocity.copy(),
                'health': d.health,
            }
            if d.team != drone.team:
                enemies.append(data)
            elif d.drone_id != drone.drone_id:
                allies.append(data)

        # Get action from combat AI
        action, maneuver_name = combat_ai.get_action(
            current_time=self.match_time,
            drone_pos=drone.position,
            drone_vel=drone.velocity,
            drone_orient=drone.orientation,
            drone_health=drone.health,
            enemies=enemies,
            allies=allies,
        )

        return action, maneuver_name

    def _simple_ai_action(self, drone: CombatDrone) -> np.ndarray:
        """Dynamic AI for opponents - creates varied dogfighting behavior."""
        # Find nearest enemy
        enemies = [d for d in self.drones.values() if d.team != drone.team and d.is_alive]
        if not enemies:
            # No enemies - do a patrol pattern
            t = self.match_time + drone.drone_id * 10  # Offset per drone
            roll_cmd = 0.3 * np.sin(t * 0.5)
            pitch_cmd = 0.2 * np.sin(t * 0.3)
            return np.array([roll_cmd, pitch_cmd, 0.6, 0, 0, 0])

        nearest = min(enemies, key=lambda e: np.linalg.norm(e.position - drone.position))

        # Calculate pursuit vector
        to_enemy = nearest.position - drone.position
        distance = np.linalg.norm(to_enemy)
        to_enemy_normalized = to_enemy / (distance + 1e-6)

        # Get current heading and speed
        yaw = drone.orientation[2]
        pitch = drone.orientation[1]
        heading = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
        speed = np.linalg.norm(drone.velocity)

        # Alignment with enemy
        alignment = np.dot(heading, to_enemy_normalized)

        # Add time-based variation for dynamic movement
        t = self.match_time + drone.drone_id * 3.14159
        variation = np.sin(t * 2) * 0.3

        # Different behaviors based on situation
        if distance < 200:
            # Too close - break away with evasive maneuver
            roll_cmd = 0.8 * np.sign(np.sin(t * 5)) + variation
            pitch_cmd = 0.5 + 0.3 * np.sin(t * 3)
            throttle = 1.0  # Full power to escape
            fire = 1 if alignment > 0.8 else 0

        elif distance < 500 and alignment > 0.7:
            # In firing position - steady pursuit with slight adjustments
            cross = np.cross(heading, to_enemy_normalized)
            roll_cmd = np.clip(cross[2] * 1.5 + variation * 0.2, -1, 1)
            alt_diff = nearest.position[2] - drone.position[2]
            pitch_cmd = np.clip(alt_diff / 150 + variation * 0.1, -0.5, 0.5)
            throttle = 0.7
            fire = 1

        elif alignment < 0.3:
            # Enemy behind or to side - perform defensive turn
            # Determine turn direction based on enemy position
            cross = np.cross(heading, to_enemy_normalized)
            turn_dir = np.sign(cross[2]) if abs(cross[2]) > 0.1 else np.sign(np.sin(t))

            roll_cmd = turn_dir * 0.9  # Hard turn
            pitch_cmd = 0.3 + variation * 0.2  # Pull up while turning
            throttle = 0.9
            fire = 0

        else:
            # Medium range pursuit with maneuvering
            cross = np.cross(heading, to_enemy_normalized)
            roll_cmd = np.clip(cross[2] * 2 + variation * 0.3, -1, 1)

            # Varied altitude changes
            alt_diff = nearest.position[2] - drone.position[2]
            pitch_cmd = np.clip(alt_diff / 100 + np.sin(t * 1.5) * 0.2, -0.6, 0.6)

            throttle = 0.8 if speed < self.config.max_speed * 0.7 else 0.5
            fire = 1 if distance < 400 and alignment > 0.85 else 0

        # Altitude safety - avoid ground and ceiling
        if drone.position[2] < self.config.arena_height_min + 150:
            pitch_cmd = max(pitch_cmd, 0.5)  # Pull up
        elif drone.position[2] > self.config.arena_height_max - 100:
            pitch_cmd = min(pitch_cmd, -0.3)  # Push down

        # Speed management
        if speed < self.config.min_speed * 1.2:
            pitch_cmd = min(pitch_cmd, 0)  # Nose down to gain speed
            throttle = 1.0

        return np.array([roll_cmd, pitch_cmd, throttle, 0, fire, 0])

    def _update_physics(self, drone: CombatDrone):
        """Update drone physics - simplified arcade-style for responsive combat."""
        # Get current state
        pos = drone.position
        vel = drone.velocity
        orient = drone.orientation  # roll, pitch, yaw

        speed = np.linalg.norm(vel)
        throttle = getattr(drone, '_throttle', 0.5)

        # === ORIENTATION UPDATE ===
        # Apply angular velocity to change heading (yaw changes based on roll)
        roll_rate = drone.angular_velocity[0]
        pitch_rate = drone.angular_velocity[1]

        # Bank-to-turn: roll creates yaw change (like a real aircraft)
        yaw_rate = roll_rate * 0.8  # Roll induces turn

        # G-force from turn rate
        if speed > 10:
            max_turn_rate = self.config.max_g_force * self.gravity / speed
            if abs(yaw_rate) > max_turn_rate:
                yaw_rate = np.sign(yaw_rate) * max_turn_rate

        drone._actual_g_force = 1.0 + abs(speed * yaw_rate / self.gravity)
        drone._actual_g_force = min(drone._actual_g_force, self.config.max_g_force + 1.0)

        # Update orientation
        drone.orientation[0] += roll_rate * self.dt  # Roll
        drone.orientation[1] += pitch_rate * self.dt  # Pitch
        drone.orientation[2] += yaw_rate * self.dt    # Yaw (from banking)

        # Clamp orientation
        drone.orientation[0] = np.clip(drone.orientation[0], -np.pi/2, np.pi/2)
        drone.orientation[1] = np.clip(drone.orientation[1], -np.pi/3, np.pi/3)

        # Roll returns to level naturally
        drone.orientation[0] *= 0.95  # Damping

        # === VELOCITY UPDATE ===
        roll = drone.orientation[0]
        pitch = drone.orientation[1]
        yaw = drone.orientation[2]

        # Forward direction based on heading
        heading = np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch)
        ])

        # Target speed based on throttle
        target_speed = self.config.min_speed + throttle * (self.config.max_speed - self.config.min_speed)

        # Smoothly adjust velocity to point in heading direction at target speed
        target_vel = heading * target_speed
        blend = 0.1  # How quickly velocity aligns with heading
        drone.velocity = drone.velocity * (1 - blend) + target_vel * blend

        # Apply gravity effect based on pitch
        if pitch < 0:  # Nose down - gain speed
            speed_gain = -pitch * 20
            current_speed = np.linalg.norm(drone.velocity)
            if current_speed < self.config.max_speed:
                drone.velocity *= (1 + speed_gain * self.dt / current_speed)
        elif pitch > 0.3:  # Nose up - lose speed
            speed_loss = pitch * 15
            current_speed = np.linalg.norm(drone.velocity)
            if current_speed > self.config.min_speed:
                drone.velocity *= (1 - speed_loss * self.dt / current_speed)

        # Altitude changes from velocity
        drone.velocity[2] -= self.gravity * 0.3 * self.dt  # Some gravity effect

        # === POSITION UPDATE ===
        drone.position = pos + drone.velocity * self.dt

        # === SPEED LIMITS ===
        speed = np.linalg.norm(drone.velocity)
        if speed < self.config.min_speed:
            drone.velocity = drone.velocity / (speed + 1e-6) * self.config.min_speed
            drone.velocity[2] -= 5  # Stall effect
        elif speed > self.config.max_speed:
            drone.velocity = drone.velocity / speed * self.config.max_speed

        # Update weapons
        for weapon in drone.weapons:
            weapon.update(self.dt)

    def _process_combat(self) -> List[Dict[str, Any]]:
        """Process combat actions and damage."""
        events = []

        for drone in self.drones.values():
            if not drone.is_alive:
                continue

            fire_cmd = getattr(drone, '_fire_command', False)
            weapon_idx = getattr(drone, '_selected_weapon', 0)
            target_idx = getattr(drone, '_target_idx', 0)

            if not fire_cmd:
                continue

            if weapon_idx >= len(drone.weapons):
                continue

            weapon = drone.weapons[weapon_idx]

            if not weapon.can_fire():
                continue

            # Get target
            enemies = [d for d in self.drones.values() if d.team != drone.team and d.is_alive]
            if target_idx >= len(enemies):
                continue

            target = enemies[target_idx]

            # Check range
            distance = np.linalg.norm(target.position - drone.position)
            if distance > weapon.range:
                continue

            # Check lock for missiles
            if weapon.lock_required:
                # Simplified lock check
                to_target = target.position - drone.position
                heading = np.array([np.cos(drone.orientation[2]), np.sin(drone.orientation[2]), 0])
                alignment = np.dot(heading, to_target / (distance + 1e-6))

                if alignment < 0.95:  # Need 95% alignment
                    weapon.current_lock = max(0, weapon.current_lock - self.dt)
                    continue

                weapon.current_lock += self.dt
                if weapon.current_lock < weapon.lock_time:
                    continue

            # Fire weapon
            if weapon.fire():
                # Calculate hit probability based on distance and alignment
                to_target = target.position - drone.position
                yaw = drone.orientation[2]
                pitch = drone.orientation[1]
                # Full 3D heading vector
                heading = np.array([
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch)
                ])
                alignment = np.dot(heading, to_target / (distance + 1e-6))

                # More generous hit calculation for exciting combat
                range_factor = 1 - (distance / weapon.range) * 0.5  # Less penalty for range
                hit_prob = max(0.1, alignment) * range_factor

                if weapon.weapon_type == WeaponType.MISSILE_IR:
                    hit_prob *= 0.95  # Very high hit rate for missiles
                elif weapon.weapon_type == WeaponType.MISSILE_RADAR:
                    hit_prob *= 0.90
                elif weapon.weapon_type == WeaponType.GUN:
                    hit_prob *= 0.6  # Better gun accuracy for more action

                hit = np.random.random() < hit_prob

                self.episode_stats["shots_fired"] += 1
                if weapon.weapon_type in [WeaponType.MISSILE_IR, WeaponType.MISSILE_RADAR]:
                    self.episode_stats["missiles_fired"] += 1

                if hit:
                    killed = target.take_damage(weapon.damage)
                    drone.damage_dealt += weapon.damage

                    self.episode_stats["shots_hit"] += 1
                    if weapon.weapon_type in [WeaponType.MISSILE_IR, WeaponType.MISSILE_RADAR]:
                        self.episode_stats["missiles_hit"] += 1

                    event = {
                        "type": "hit",
                        "attacker": drone.drone_id,
                        "target": target.drone_id,
                        "weapon": weapon.weapon_type.value,
                        "damage": weapon.damage,
                        "distance": distance,
                    }

                    if killed:
                        drone.kills += 1
                        if drone.team == 0:
                            self.red_kills += 1
                        else:
                            self.blue_kills += 1

                        event["type"] = "kill"

                        # Respawn if enabled
                        if self.config.respawn_enabled:
                            spawn_pos = self._get_spawn_position(
                                target.team,
                                target.drone_id % max(self.config.num_red, self.config.num_blue)
                            )
                            spawn_vel = np.array([
                                self.config.min_speed * 1.5 * (1 if target.team == 0 else -1),
                                0, 0
                            ])
                            target.respawn(spawn_pos, spawn_vel)

                    events.append(event)
                    self.combat_log.append(event)

        return events

    def _check_boundaries(self, drone: CombatDrone):
        """Check if drone is out of bounds and enforce hard limits."""
        pos = drone.position
        half_size = self.config.arena_size / 2
        warning_zone = half_size * 0.9  # 90% of boundary

        # Soft boundary - warn and turn back
        if abs(pos[0]) > warning_zone or abs(pos[1]) > warning_zone:
            # Apply force pushing back toward center
            center_dir = -pos[:2] / (np.linalg.norm(pos[:2]) + 1e-6)
            drone.velocity[:2] += center_dir * 10  # Push toward center

        # Hard boundary - enforce strict limits
        if abs(pos[0]) > half_size:
            drone.take_damage(5)
            drone.velocity[0] *= -0.8
            drone.position[0] = np.sign(pos[0]) * (half_size - 50)
            # Force turn toward center
            drone.orientation[2] = np.arctan2(-pos[1], -pos[0])

        if abs(pos[1]) > half_size:
            drone.take_damage(5)
            drone.velocity[1] *= -0.8
            drone.position[1] = np.sign(pos[1]) * (half_size - 50)
            # Force turn toward center
            drone.orientation[2] = np.arctan2(-pos[1], -pos[0])

        # Absolute hard limit - teleport back if somehow escaped
        max_distance = half_size * 1.2
        if abs(pos[0]) > max_distance or abs(pos[1]) > max_distance:
            drone.position[0] = np.clip(drone.position[0], -half_size + 100, half_size - 100)
            drone.position[1] = np.clip(drone.position[1], -half_size + 100, half_size - 100)
            drone.velocity[:2] = drone.velocity[:2] * 0.5  # Slow down

    def _check_altitude(self, drone: CombatDrone):
        """Check altitude limits."""
        if drone.position[2] < self.config.arena_height_min:
            # Crashed
            drone.is_alive = False
            drone.deaths += 1

            if self.config.respawn_enabled:
                spawn_pos = self._get_spawn_position(
                    drone.team,
                    drone.drone_id % max(self.config.num_red, self.config.num_blue)
                )
                spawn_vel = np.array([
                    self.config.min_speed * 1.5 * (1 if drone.team == 0 else -1),
                    0, 0
                ])
                drone.respawn(spawn_pos, spawn_vel)

        elif drone.position[2] > self.config.arena_height_max:
            drone.velocity[2] = -abs(drone.velocity[2])
            drone.position[2] = self.config.arena_height_max - 10

    def _calculate_reward(self, agent: CombatDrone, events: List[Dict[str, Any]]) -> float:
        """Calculate reward for the agent."""
        reward = 0.0

        # Time penalty (encourages action)
        reward += self.config.reward_per_second * self.dt

        # Combat events
        for event in events:
            if event["attacker"] == agent.drone_id:
                if event["type"] == "kill":
                    reward += self.config.reward_kill
                    self.episode_stats["kills"] += 1
                elif event["type"] == "hit":
                    reward += self.config.reward_hit

            if event["target"] == agent.drone_id:
                if event["type"] == "kill":
                    reward += self.config.reward_death
                    self.episode_stats["deaths"] += 1

        # Pursuit reward - getting closer to enemies
        enemies = [d for d in self.drones.values() if d.team != agent.team and d.is_alive]
        if enemies and agent.is_alive:
            nearest_dist = min(np.linalg.norm(e.position - agent.position) for e in enemies)
            if nearest_dist < 500:
                reward += self.config.reward_pursuit * (500 - nearest_dist) / 500 * self.dt

        # Track stats
        self.episode_stats["damage_dealt"] = agent.damage_dealt
        self.episode_stats["damage_taken"] = agent.damage_taken
        if agent.is_alive:
            self.episode_stats["time_alive"] += self.dt

        return reward

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if match should end."""
        terminated = False
        truncated = False

        # Time limit
        if self.match_time >= self.config.max_match_time:
            truncated = True

        # Win conditions
        if self.config.win_condition == "kills":
            if self.red_kills >= self.config.kills_to_win:
                terminated = True
            elif self.blue_kills >= self.config.kills_to_win:
                terminated = True

        elif self.config.win_condition == "last_alive":
            red_alive = sum(1 for d in self.drones.values() if d.team == 0 and d.is_alive)
            blue_alive = sum(1 for d in self.drones.values() if d.team == 1 and d.is_alive)

            if not self.config.respawn_enabled:
                if red_alive == 0 or blue_alive == 0:
                    terminated = True

        return terminated, truncated

    def _get_observation(self) -> np.ndarray:
        """Get observation for the learning agent."""
        return self._get_observation_for_drone(self.drones[self.agent_drone_id])

    def _get_observation_for_drone(self, drone: CombatDrone) -> np.ndarray:
        """Get observation from a specific drone's perspective."""
        obs = []

        # Own state (14)
        obs.extend(drone.position / self.config.arena_size)  # Normalized position
        obs.extend(drone.velocity / self.config.max_speed)  # Normalized velocity
        obs.extend(drone.orientation / np.pi)  # Normalized orientation
        obs.extend(drone.angular_velocity / np.pi)  # Normalized angular velocity
        obs.append(drone.health / drone.max_health)
        obs.append(np.linalg.norm(drone.velocity) / self.config.max_speed)

        # Enemy states (7 per enemy)
        enemies = [d for d in self.drones.values() if d.team != drone.team]
        for enemy in enemies:
            if enemy.is_alive:
                rel_pos = (enemy.position - drone.position) / self.config.arena_size
                rel_vel = (enemy.velocity - drone.velocity) / self.config.max_speed
                distance = np.linalg.norm(enemy.position - drone.position)
                threat = 1.0 - min(distance / 1000, 1.0)  # Closer = higher threat
            else:
                rel_pos = np.zeros(3)
                rel_vel = np.zeros(3)
                threat = 0.0

            obs.extend(rel_pos)
            obs.extend(rel_vel)
            obs.append(threat)

        # Weapon states (3 per weapon)
        for weapon in drone.weapons:
            obs.append(weapon.ammo / weapon.max_ammo)
            obs.append(1.0 - weapon.cooldown / weapon.cooldown_time)
            obs.append(weapon.current_lock / weapon.lock_time if weapon.lock_required else 1.0)

        # Tactical info (3)
        obs.append((drone.position[2] - self.config.arena_height_min) /
                   (self.config.arena_height_max - self.config.arena_height_min))
        obs.append(1.0 - min(abs(drone.position[0]), abs(drone.position[1])) /
                   (self.config.arena_size / 2))

        # G-force (use actual computed value from physics, capped realistically)
        actual_g = getattr(drone, '_actual_g_force', 1.0)
        obs.append(min(actual_g / self.config.max_g_force, 1.0))

        return np.array(obs, dtype=np.float32)

    def _get_num_enemies(self) -> int:
        """Get number of enemies for the agent."""
        return self.config.num_blue if self.agent_team == 0 else self.config.num_red

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        agent = self.drones[self.agent_drone_id]

        return {
            "match_time": self.match_time,
            "red_kills": self.red_kills,
            "blue_kills": self.blue_kills,
            "agent_kills": agent.kills,
            "agent_deaths": agent.deaths,
            "agent_health": agent.health,
            "agent_alive": agent.is_alive,
            "episode_stats": self.episode_stats.copy(),
        }

    def set_opponent_policy(self, policy: Callable):
        """Set the opponent policy for self-play."""
        self.opponent_policy = policy

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Could integrate with PyBullet visualization
            pass
        return None

    def close(self):
        """Clean up resources."""
        pass


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_1v1_dogfight(render_mode: Optional[str] = None) -> DogfightEnv:
    """Create a 1v1 dogfight environment."""
    config = DogfightConfig(
        num_red=1,
        num_blue=1,
        arena_size=2000.0,
        respawn_enabled=True,
        kills_to_win=5,
        max_match_time=180.0,
    )
    return DogfightEnv(config=config, render_mode=render_mode)


def create_2v2_dogfight(render_mode: Optional[str] = None) -> DogfightEnv:
    """Create a 2v2 team dogfight environment."""
    config = DogfightConfig(
        num_red=2,
        num_blue=2,
        arena_size=3000.0,
        respawn_enabled=True,
        kills_to_win=10,
        max_match_time=300.0,
    )
    return DogfightEnv(config=config, render_mode=render_mode)


def create_swarm_battle(
    red_count: int = 4,
    blue_count: int = 4,
    render_mode: Optional[str] = None,
) -> DogfightEnv:
    """Create a swarm vs swarm battle."""
    config = DogfightConfig(
        num_red=red_count,
        num_blue=blue_count,
        arena_size=5000.0,
        respawn_enabled=False,  # Last team standing
        win_condition="last_alive",
        max_match_time=600.0,
    )
    return DogfightEnv(config=config, render_mode=render_mode)


def create_tournament_match(render_mode: Optional[str] = None) -> DogfightEnv:
    """Create a tournament-style 1v1 match."""
    config = DogfightConfig(
        num_red=1,
        num_blue=1,
        arena_size=2000.0,
        respawn_enabled=False,
        win_condition="last_alive",
        max_match_time=120.0,
        weapons_config=[
            {"type": "gun", "ammo": 200, "range": 300, "damage": 15, "cooldown": 0.1},
            {"type": "missile_ir", "ammo": 2, "range": 800, "damage": 100, "cooldown": 3.0, "lock_time": 2.0},
        ],
    )
    return DogfightEnv(config=config, render_mode=render_mode)

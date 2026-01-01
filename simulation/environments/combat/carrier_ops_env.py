"""
Carrier Operations Environment.

Training environment for carrier-capable UAVs (X-47B) with:
- Catapult launch physics
- Carrier approach patterns (Case I, II, III)
- Arrested landing with tailhook
- Moving deck compensation
- Bolter and wave-off mechanics

This is one of the most challenging fixed-wing control tasks.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum, auto
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base_fixed_wing_env import BaseFixedWingEnv, FixedWingObservation


class CarrierPhase(Enum):
    """Carrier operation phases."""
    DECK = auto()          # On deck, pre-launch
    CATAPULT = auto()      # Catapult stroke
    CLIMBOUT = auto()      # Initial climb after launch
    MISSION = auto()       # Mission phase (simplified)
    MARSHAL = auto()       # Holding pattern for approach
    APPROACH = auto()      # Final approach
    GROOVE = auto()        # In the groove (final 18 seconds)
    TOUCHDOWN = auto()     # Wheels on deck
    ARRESTED = auto()      # Caught the wire
    BOLTER = auto()        # Missed wire, going around
    WAVEOFF = auto()       # LSO waved off


class ApproachCase(Enum):
    """Carrier approach cases."""
    CASE_I = "case_i"      # Day VFR (ceiling > 3000ft, vis > 5nm)
    CASE_II = "case_ii"    # Day/Night (ceiling 1000-3000ft)
    CASE_III = "case_iii"  # Night/IMC (ceiling < 1000ft or vis < 5nm)


@dataclass
class CarrierState:
    """Aircraft carrier state."""
    position: np.ndarray        # [x, y, z] world position
    heading: float              # Ship heading [rad]
    speed: float                # Ship speed [m/s]

    # Deck motion (ship motion in waves)
    pitch: float = 0.0          # Deck pitch [rad]
    roll: float = 0.0           # Deck roll [rad]
    heave: float = 0.0          # Vertical motion [m]

    # Wind over deck
    wind_over_deck: float = 0.0  # [m/s]

    def update(self, dt: float, sea_state: int = 3):
        """Update carrier motion based on sea state."""
        # Simple sinusoidal deck motion model
        t = dt  # Would need cumulative time

        # Sea state affects motion amplitude
        amplitude = sea_state * 0.5  # degrees
        period = 8.0 + sea_state  # seconds

        self.pitch = np.radians(amplitude) * np.sin(2 * np.pi * t / period)
        self.roll = np.radians(amplitude * 0.7) * np.sin(2 * np.pi * t / (period * 1.3))
        self.heave = amplitude * 0.5 * np.sin(2 * np.pi * t / (period * 0.8))

        # Update position based on heading and speed
        self.position[0] += self.speed * np.cos(self.heading) * dt
        self.position[1] += self.speed * np.sin(self.heading) * dt

    def get_deck_position(self) -> np.ndarray:
        """Get current deck touchdown point in world frame."""
        # Deck is at ship position, adjusted for motion
        deck_pos = self.position.copy()
        deck_pos[2] += self.heave + 20.0  # 20m deck height
        return deck_pos

    def get_glideslope_point(self, distance: float) -> np.ndarray:
        """Get point on ideal glideslope at given distance."""
        # 3.5 degree glideslope for carrier
        glideslope = np.radians(3.5)

        # Start from deck
        point = self.get_deck_position()

        # Go back along approach path
        approach_heading = self.heading + np.pi  # Opposite of ship heading
        point[0] += distance * np.cos(approach_heading)
        point[1] += distance * np.sin(approach_heading)
        point[2] += distance * np.tan(glideslope)

        return point


@dataclass
class CatapultState:
    """Catapult launch state."""
    stroke_length: float = 95.0     # EMALS stroke [m]
    max_acceleration: float = 250.0  # Max accel [m/s^2] (~25g)
    launch_speed: float = 75.0      # End of stroke speed [m/s]
    stroke_time: float = 2.0        # Time for full stroke [s]
    current_position: float = 0.0   # Position along stroke
    is_launching: bool = False


class CarrierOpsEnv(BaseFixedWingEnv):
    """
    Carrier operations training environment.

    Supports:
    - Catapult launch (EMALS physics)
    - Case I/II/III approaches
    - Arrested landing with 4-wire system
    - Moving deck compensation
    - LSO guidance (simplified)
    - Bolter and wave-off handling

    This is extremely challenging - carrier landing has been called
    "the most difficult regularly scheduled task in aviation."

    Observation (30 dims):
    - Base flight state (21)
    - Carrier-relative state (6)
    - Approach state (3)

    Action space:
    - [aileron, elevator, rudder, throttle, hook_deploy]
    """

    def __init__(
        self,
        platform_config: Dict[str, Any],
        approach_case: str = "case_i",
        sea_state: int = 3,
        enable_deck_motion: bool = True,
        **kwargs
    ):
        """
        Initialize carrier ops environment.

        Args:
            platform_config: Platform configuration
            approach_case: "case_i", "case_ii", or "case_iii"
            sea_state: Sea state 1-6 (affects deck motion)
            enable_deck_motion: Enable ship motion simulation
        """
        super().__init__(platform_config, **kwargs)

        self.approach_case = ApproachCase(approach_case)
        self.sea_state = sea_state
        self.enable_deck_motion = enable_deck_motion

        # Verify carrier capability
        physics = platform_config.get('physics_params', {})
        if not physics.get('carrier_capable', False):
            print("Warning: Platform may not be carrier capable")

        # Carrier state
        self._carrier = CarrierState(
            position=np.array([10000.0, 0.0, 0.0]),
            heading=0.0,  # Heading into wind
            speed=15.0,   # ~30 knots
        )

        # Catapult
        self._catapult = CatapultState()

        # Mission state
        self._phase = CarrierPhase.DECK
        self._hook_deployed = False
        self._wire_caught = 0  # Which wire (1-4) or 0 for none
        self._bolter_count = 0
        self._approach_time = 0.0

        # Landing parameters
        self._glideslope = np.radians(3.5)
        self._on_speed_aoa = np.radians(8.0)  # On-speed AOA
        self._lineup_tolerance = np.radians(2.0)

        # Wire positions (relative to deck edge, positive = closer)
        self._wire_positions = [35.0, 45.0, 55.0, 65.0]  # meters from ramp

        # Extended observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32
        )

        # Action space with hook deploy
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def _create_carrier_visual(self):
        """Create visual representation of the aircraft carrier."""
        if self._using_airsim or not hasattr(self, '_p'):
            return

        p = self._p

        # Carrier deck dimensions (simplified Nimitz-class)
        deck_length = 330.0  # meters
        deck_width = 78.0    # meters (flight deck)
        deck_height = 20.0   # height above water

        # Main deck (gray)
        deck_half = [deck_length / 2, deck_width / 2, 1.0]
        deck_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=deck_half)
        deck_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=deck_half,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]  # Dark gray deck
        )

        # Hull (darker)
        hull_half = [deck_length / 2, deck_width / 3, deck_height / 2]
        hull_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=hull_half)
        hull_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=hull_half,
            rgbaColor=[0.2, 0.2, 0.25, 1.0]  # Darker hull
        )

        # Island superstructure
        island_half = [20.0, 10.0, 15.0]
        island_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=island_half)
        island_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=island_half,
            rgbaColor=[0.4, 0.4, 0.45, 1.0]
        )

        # Landing area marking (angled deck line - yellow stripe)
        stripe_half = [100.0, 1.0, 0.1]
        stripe_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=stripe_half,
            rgbaColor=[1.0, 0.9, 0.0, 1.0]  # Yellow
        )

        # Arresting wire markers (4 wires)
        wire_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, deck_width / 3, 0.2],
            rgbaColor=[1.0, 0.0, 0.0, 1.0]  # Red
        )

        # Create carrier as multi-body
        carrier_pos = self._carrier.position.copy()
        carrier_pos[2] = deck_height / 2

        self._carrier_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=hull_col,
            baseVisualShapeIndex=hull_vis,
            basePosition=carrier_pos.tolist(),
            linkMasses=[0, 0, 0, 0, 0, 0, 0],
            linkCollisionShapeIndices=[deck_col, island_col, -1, -1, -1, -1, -1],
            linkVisualShapeIndices=[deck_vis, island_vis, stripe_vis, wire_vis, wire_vis, wire_vis, wire_vis],
            linkPositions=[
                [0, 0, deck_height / 2 + 1],      # Deck on top of hull
                [80, -deck_width / 3, deck_height / 2 + 15],  # Island starboard side
                [-50, 10, deck_height / 2 + 1.5],  # Landing stripe
                [-deck_length / 2 + 35, 0, deck_height / 2 + 1.2],   # Wire 1
                [-deck_length / 2 + 45, 0, deck_height / 2 + 1.2],   # Wire 2
                [-deck_length / 2 + 55, 0, deck_height / 2 + 1.2],   # Wire 3
                [-deck_length / 2 + 65, 0, deck_height / 2 + 1.2],   # Wire 4
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                p.getQuaternionFromEuler([0, 0, np.radians(-9)]),  # Angled deck
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkInertialFramePositions=[[0, 0, 0]] * 7,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 7,
            linkParentIndices=[0, 0, 0, 0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED] * 7,
            linkJointAxis=[[0, 0, 1]] * 7
        )

        # Create water plane (blue)
        water_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[5000, 5000, 0.1],
            rgbaColor=[0.0, 0.2, 0.5, 0.8]
        )
        self._water_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=water_vis,
            basePosition=[carrier_pos[0], carrier_pos[1], -0.1]
        )

    def _update_carrier_visual(self):
        """Update carrier visual position and orientation."""
        if not hasattr(self, '_carrier_id') or self._using_airsim:
            return

        p = self._p

        # Update carrier position/orientation with deck motion
        carrier_pos = self._carrier.position.copy()
        carrier_pos[2] = 10.0 + self._carrier.heave  # Base height + heave

        quat = p.getQuaternionFromEuler([
            self._carrier.roll,
            self._carrier.pitch,
            self._carrier.heading
        ])

        p.resetBasePositionAndOrientation(
            self._carrier_id,
            carrier_pos.tolist(),
            quat
        )

        # Update water to follow carrier roughly
        if hasattr(self, '_water_id'):
            p.resetBasePositionAndOrientation(
                self._water_id,
                [carrier_pos[0], carrier_pos[1], -0.1],
                [0, 0, 0, 1]
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset for carrier operations."""
        obs_array, info = super().reset(seed=seed, options=options)

        # Reset carrier
        self._carrier = CarrierState(
            position=np.array([10000.0, 0.0, 0.0]),
            heading=self.np_random.uniform(-0.1, 0.1),  # Slight heading variation
            speed=15.0,
        )

        # Reset mission state
        self._phase = CarrierPhase.DECK
        self._hook_deployed = False
        self._wire_caught = 0
        self._bolter_count = 0
        self._approach_time = 0.0

        # Determine starting phase based on options
        start_phase = options.get('start_phase', 'approach') if options else 'approach'

        if start_phase == 'catapult':
            self._setup_catapult_launch()
        elif start_phase == 'approach':
            self._setup_approach()
        else:
            self._setup_approach()

        # Create carrier visual (only once on first reset)
        if not hasattr(self, '_carrier_id'):
            self._create_carrier_visual()
        else:
            # Update carrier visual position
            self._update_carrier_visual()

        obs = self._get_carrier_observation()

        info.update({
            'phase': self._phase.name,
            'carrier_position': self._carrier.position.copy(),
            'hook_deployed': self._hook_deployed,
        })

        return obs, info

    def _setup_catapult_launch(self):
        """Setup for catapult launch."""
        self._phase = CarrierPhase.DECK
        self._catapult = CatapultState()

        # Position on catapult
        deck_pos = self._carrier.get_deck_position()
        self._position = deck_pos + np.array([-50.0, 0.0, 5.0])  # On cat
        self._velocity = np.array([self._carrier.speed, 0.0, 0.0])
        self._attitude = np.array([0.0, 0.0, self._carrier.heading])

    def _setup_approach(self):
        """Setup for carrier approach."""
        self._phase = CarrierPhase.APPROACH

        # Position 1nm behind carrier on glideslope (closer for faster visual)
        approach_dist = 1852.0  # 1nm in meters
        start_pos = self._carrier.get_glideslope_point(approach_dist)

        self._position = start_pos

        # Velocity toward carrier (along carrier heading direction)
        approach_speed = 70.0  # m/s
        descent_rate = approach_speed * np.tan(self._glideslope)
        self._velocity = np.array([
            approach_speed * np.cos(self._carrier.heading),
            approach_speed * np.sin(self._carrier.heading),
            -descent_rate
        ])

        # Attitude: facing carrier (same as carrier heading)
        self._attitude = np.array([0.0, -self._glideslope, self._carrier.heading])

        self._hook_deployed = True

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute carrier ops step."""
        self._current_step += 1
        dt = 1.0 / self.control_hz

        # Hook deployment
        if action[4] > 0.5 and not self._hook_deployed:
            self._hook_deployed = True

        # Update carrier
        if self.enable_deck_motion:
            self._carrier.update(dt, self.sea_state)

        # Phase-specific handling
        if self._phase == CarrierPhase.DECK:
            # Waiting for catapult
            if action[3] > 0.9:  # Full throttle = launch
                self._phase = CarrierPhase.CATAPULT
                self._catapult.is_launching = True

        elif self._phase == CarrierPhase.CATAPULT:
            # Catapult acceleration
            self._catapult_step(dt)

        else:
            # Normal flight physics
            surfaces = self._mixer.mix(
                roll_cmd=float(action[0]),
                pitch_cmd=float(action[1]),
                yaw_cmd=float(action[2]),
                throttle=float(action[3]),
                dt=dt
            )

            self._current_surfaces = np.array([
                surfaces.aileron, surfaces.elevator,
                surfaces.rudder, surfaces.flaps
            ])
            self._throttle = float(action[3])

            for _ in range(self.physics_steps_per_control):
                self._physics_step(surfaces, float(action[3]))

        # Update phase
        self._update_carrier_phase()

        # Check for landing/wire catch
        self._check_landing()

        # Update visuals
        self._update_carrier_visual()
        self._update_aircraft_visual()

        obs = self._get_carrier_observation()
        reward = self._compute_carrier_reward(action)
        terminated = self._check_carrier_terminated()
        truncated = self._current_step >= self._max_episode_steps

        info = {
            'phase': self._phase.name,
            'hook_deployed': self._hook_deployed,
            'wire_caught': self._wire_caught,
            'bolter_count': self._bolter_count,
            'glideslope_error': self._compute_glideslope_error(),
            'lineup_error': self._compute_lineup_error(),
        }

        return obs, reward, terminated, truncated, info

    def _catapult_step(self, dt: float):
        """Execute catapult launch physics."""
        cat = self._catapult

        if not cat.is_launching:
            return

        # Catapult acceleration profile (trapezoidal)
        progress = cat.current_position / cat.stroke_length

        if progress < 0.1:
            accel = cat.max_acceleration * (progress / 0.1)
        elif progress < 0.9:
            accel = cat.max_acceleration
        else:
            accel = cat.max_acceleration * ((1.0 - progress) / 0.1)

        # Update velocity and position
        self._velocity[0] += accel * dt
        cat.current_position += self._velocity[0] * dt

        # Update aircraft position
        self._position[0] += self._velocity[0] * dt

        # End of stroke
        if cat.current_position >= cat.stroke_length:
            self._phase = CarrierPhase.CLIMBOUT
            cat.is_launching = False

            # Add deck motion to velocity
            self._velocity[0] += self._carrier.speed

    def _update_carrier_phase(self):
        """Update carrier operation phase."""
        deck_pos = self._carrier.get_deck_position()
        rel_pos = self._position - deck_pos
        dist_to_deck = np.linalg.norm(rel_pos)

        if self._phase == CarrierPhase.CLIMBOUT:
            if self._position[2] > 150:  # Cleared deck
                self._phase = CarrierPhase.MISSION

        elif self._phase == CarrierPhase.MISSION:
            # Simplified: just track distance
            if dist_to_deck > 30000:  # 30km = time to return
                self._phase = CarrierPhase.MARSHAL

        elif self._phase == CarrierPhase.MARSHAL:
            # Descend and approach
            if dist_to_deck < 10000:
                self._phase = CarrierPhase.APPROACH

        elif self._phase == CarrierPhase.APPROACH:
            if dist_to_deck < 1500:
                self._phase = CarrierPhase.GROOVE
                self._approach_time = 0.0

        elif self._phase == CarrierPhase.GROOVE:
            self._approach_time += 1.0 / self.control_hz

    def _check_landing(self):
        """Check for deck touchdown and wire engagement."""
        if self._phase not in [CarrierPhase.GROOVE, CarrierPhase.TOUCHDOWN]:
            return

        deck_pos = self._carrier.get_deck_position()

        # Height above deck
        height_above_deck = self._position[2] - deck_pos[2]

        # Distance along deck (from ramp)
        deck_relative = self._position - deck_pos
        along_deck = deck_relative[0] * np.cos(self._carrier.heading) + \
                    deck_relative[1] * np.sin(self._carrier.heading)

        # Touchdown detection
        if height_above_deck < 3.0 and self._velocity[2] < 0:
            self._phase = CarrierPhase.TOUCHDOWN

            # Check wire catch
            if self._hook_deployed:
                for i, wire_pos in enumerate(self._wire_positions):
                    if abs(along_deck - wire_pos) < 5.0:
                        self._wire_caught = i + 1
                        self._phase = CarrierPhase.ARRESTED
                        return

            # No wire caught = bolter
            if along_deck > self._wire_positions[-1] + 10:
                self._phase = CarrierPhase.BOLTER
                self._bolter_count += 1

    def _compute_glideslope_error(self) -> float:
        """Compute deviation from glideslope."""
        deck_pos = self._carrier.get_deck_position()
        rel_pos = self._position - deck_pos

        # Distance to deck
        horiz_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)

        # Ideal height on glideslope
        ideal_height = horiz_dist * np.tan(self._glideslope)

        return rel_pos[2] - ideal_height

    def _compute_lineup_error(self) -> float:
        """Compute lateral deviation from centerline."""
        deck_pos = self._carrier.get_deck_position()
        rel_pos = self._position - deck_pos

        # Lateral offset from centerline
        lateral = rel_pos[1] * np.cos(self._carrier.heading) - \
                 rel_pos[0] * np.sin(self._carrier.heading)

        return lateral

    def _get_carrier_observation(self) -> np.ndarray:
        """Get observation with carrier-relative info."""
        base_obs = self._get_observation()
        base_array = base_obs.to_array()  # 21 dims

        # Carrier-relative state (6 dims)
        deck_pos = self._carrier.get_deck_position()
        rel_pos = (self._position - deck_pos) / 1000.0  # Normalized

        carrier_state = np.array([
            rel_pos[0],
            rel_pos[1],
            rel_pos[2],
            self._carrier.pitch,
            self._carrier.roll,
            self._carrier.heave / 10.0,
        ])

        # Approach state (3 dims)
        approach_state = np.array([
            self._compute_glideslope_error() / 50.0,
            self._compute_lineup_error() / 50.0,
            float(self._hook_deployed),
        ])

        return np.concatenate([
            base_array,      # 21
            carrier_state,   # 6
            approach_state,  # 3
        ]).astype(np.float32)  # Total: 30

    def _compute_carrier_reward(self, action: np.ndarray) -> float:
        """Compute reward for carrier operations."""
        reward = 0.0

        # Phase-specific rewards
        if self._phase == CarrierPhase.CATAPULT:
            reward += 1.0  # Launching

        elif self._phase == CarrierPhase.CLIMBOUT:
            # Reward altitude gain
            reward += 0.01 * self._position[2]

        elif self._phase == CarrierPhase.APPROACH:
            # Reward glideslope adherence
            gs_error = abs(self._compute_glideslope_error())
            reward += 1.0 / (1.0 + gs_error / 10.0)

            # Reward lineup
            lineup_error = abs(self._compute_lineup_error())
            reward += 1.0 / (1.0 + lineup_error / 10.0)

        elif self._phase == CarrierPhase.GROOVE:
            # High precision required
            gs_error = abs(self._compute_glideslope_error())
            lineup_error = abs(self._compute_lineup_error())

            # Tighter tolerances in groove
            reward += 5.0 / (1.0 + gs_error / 3.0)
            reward += 5.0 / (1.0 + lineup_error / 3.0)

            # Reward stable approach
            reward -= 0.5 * np.sum(action[:3] ** 2)

        elif self._phase == CarrierPhase.ARRESTED:
            # Success! Big reward
            reward += 200.0

            # Bonus for catching wire 3 (optimal)
            if self._wire_caught == 3:
                reward += 50.0
            elif self._wire_caught in [2, 4]:
                reward += 25.0

        elif self._phase == CarrierPhase.BOLTER:
            # Survived but missed
            reward -= 20.0

        # General penalties
        # Hook not deployed on approach
        if self._phase in [CarrierPhase.APPROACH, CarrierPhase.GROOVE]:
            if not self._hook_deployed:
                reward -= 5.0

        # Excessive sink rate
        if self._velocity[2] < -10.0:
            reward -= 2.0

        return reward

    def _check_carrier_terminated(self) -> bool:
        """Check if carrier ops should terminate."""
        # Success
        if self._phase == CarrierPhase.ARRESTED:
            return True

        # Crash
        if self._position[2] < 0:
            return True

        # Ramp strike (below deck level at deck edge)
        deck_pos = self._carrier.get_deck_position()
        if self._position[2] < deck_pos[2] - 5:
            # Check if near carrier
            dist = np.linalg.norm(self._position[:2] - deck_pos[:2])
            if dist < 200:
                return True

        # Base termination
        if super()._check_terminated():
            return True

        # Too many bolters
        if self._bolter_count >= 3:
            return True

        return False

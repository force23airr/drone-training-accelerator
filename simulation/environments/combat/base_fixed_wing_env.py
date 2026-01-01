"""
Base Fixed-Wing UAV Environment.

Gymnasium-compatible environment for fixed-wing jet UAV training.
Supports both PyBullet (fast training) and AirSim (visualization) backends.

Features:
- Full 6-DOF aerodynamic simulation
- Jet propulsion with fuel consumption
- Control surface dynamics
- Environmental effects (wind, weather)
- Weapon system state machine
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WeaponState(Enum):
    """Weapon system state machine for strike missions."""
    SAFE = "safe"           # Weapon not armed, cannot release
    ARMED = "armed"         # Ready for release when in envelope
    TERMINAL = "terminal"   # Final attack dive (loitering munitions)
    RELEASED = "released"   # Weapon has been released
    DETONATED = "detonated"  # Impact confirmed


@dataclass
class FixedWingObservation:
    """
    Observation state for fixed-wing UAV.

    Extended observation includes aerodynamic state variables
    not present in quadcopter observations.
    """
    # Position (world frame) [m]
    position: np.ndarray        # [x, y, z]

    # Velocity (world frame) [m/s]
    velocity: np.ndarray        # [vx, vy, vz]

    # Attitude [rad]
    attitude: np.ndarray        # [roll, pitch, yaw]

    # Angular velocity (body frame) [rad/s]
    angular_velocity: np.ndarray  # [p, q, r]

    # Aerodynamic state
    airspeed: float             # True airspeed [m/s]
    alpha: float                # Angle of attack [rad]
    beta: float                 # Sideslip angle [rad]
    mach: float                 # Mach number

    # Fuel state
    fuel_fraction: float        # Remaining fuel (0-1)

    # Control surfaces [rad]
    control_surfaces: np.ndarray  # [aileron, elevator, rudder, flaps]

    def to_array(self) -> np.ndarray:
        """Convert to flat numpy array for RL."""
        return np.concatenate([
            self.position,              # 3
            self.velocity,              # 3
            self.attitude,              # 3
            self.angular_velocity,      # 3
            [self.airspeed],            # 1
            [self.alpha],               # 1
            [self.beta],                # 1
            [self.mach],                # 1
            [self.fuel_fraction],       # 1
            self.control_surfaces,      # 4
        ])  # Total: 21

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FixedWingObservation':
        """Create from flat array."""
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            attitude=arr[6:9],
            angular_velocity=arr[9:12],
            airspeed=arr[12],
            alpha=arr[13],
            beta=arr[14],
            mach=arr[15],
            fuel_fraction=arr[16],
            control_surfaces=arr[17:21],
        )


class BaseFixedWingEnv(gym.Env):
    """
    Base environment for fixed-wing jet UAV simulation.

    Designed for RL training of military jet UAVs with:
    - Realistic 6-DOF aerodynamics
    - Jet propulsion physics
    - Control surface dynamics
    - Fuel consumption
    - Optional weapon systems

    Supports dual backends:
    - PyBullet: Fast simulation (~5000 steps/sec) for training
    - AirSim: Photorealistic rendering for visualization
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        platform_config: Dict[str, Any],
        backend: str = "pybullet",
        render_mode: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        enable_weapon: bool = False,
        physics_hz: float = 240.0,
        control_hz: float = 48.0,
        **kwargs
    ):
        """
        Initialize fixed-wing environment.

        Args:
            platform_config: Platform configuration dictionary
            backend: Physics backend ("pybullet" or "airsim")
            render_mode: Rendering mode (None, "human", or "rgb_array")
            target_position: Optional target position for navigation
            enable_weapon: Enable weapon system state machine
            physics_hz: Physics simulation frequency
            control_hz: Control loop frequency
        """
        super().__init__()

        self.platform_config = platform_config
        self.backend_type = backend
        self.render_mode = render_mode
        self.enable_weapon = enable_weapon
        self.physics_hz = physics_hz
        self.control_hz = control_hz
        self.physics_steps_per_control = int(physics_hz / control_hz)

        # Extract parameters
        self._mass = platform_config.get('mass', 1000.0)
        self._fuel_capacity = platform_config.get('physics_params', {}).get(
            'fuel_capacity', 1000.0
        )
        self._max_episode_steps = platform_config.get('max_episode_steps', 5000)

        # Initialize physics backend
        self._init_backend()

        # Initialize aerodynamics and propulsion
        self._init_physics_models()

        # State
        self._current_step = 0
        self._fuel_remaining = self._fuel_capacity
        self._weapon_state = WeaponState.SAFE
        self._target_position = target_position or np.array([1000.0, 0.0, 500.0])

        # Control surfaces current state
        self._current_surfaces = np.zeros(4)  # [aileron, elevator, rudder, flaps]
        self._throttle = 0.5

        # Define action space: [aileron, elevator, rudder, throttle, flaps]
        # All normalized to [-1, 1] except throttle [0, 1] and flaps [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space (21 dimensions)
        obs_dim = 21
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Bounds for termination
        self._bounds = {
            'xy_max': 50000.0,    # 50 km
            'z_min': 10.0,        # Minimum altitude
            'z_max': 20000.0,     # Maximum altitude
        }

    def _init_backend(self):
        """Initialize physics backend."""
        if self.backend_type == "airsim":
            try:
                from simulation.physics.airsim_integration import AirSimBackend
                self.physics = AirSimBackend()
                self._using_airsim = True
            except ImportError:
                print("Warning: AirSim not available, falling back to PyBullet")
                self._init_pybullet()
                self._using_airsim = False
        else:
            self._init_pybullet()
            self._using_airsim = False

    def _init_pybullet(self):
        """Initialize PyBullet backend."""
        try:
            import pybullet as p
            import pybullet_data

            # Determine connection mode
            if self.render_mode == "human":
                self._physics_client = p.connect(p.GUI)
            else:
                self._physics_client = p.connect(p.DIRECT)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / self.physics_hz)

            self._p = p
            self._using_airsim = False

            # Load ground plane
            p.loadURDF("plane.urdf")

            # Create aircraft visual body
            self._create_aircraft_visual()

            # Set up camera for better viewing
            p.resetDebugVisualizerCamera(
                cameraDistance=50,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 100]
            )

        except ImportError:
            raise RuntimeError("PyBullet is required but not installed")

    def _create_aircraft_visual(self):
        """Create a visual representation of the jet aircraft."""
        p = self._p

        # Get aircraft dimensions from config
        physics_params = self.platform_config.get('physics_params', {})
        wingspan = physics_params.get('wingspan', 10.0)
        length = physics_params.get('mean_chord', 3.0) * 3  # Approximate fuselage length

        # Create fuselage (elongated box)
        fuselage_half = [length / 2, 0.5, 0.3]
        fuselage_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=fuselage_half)
        fuselage_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=fuselage_half,
            rgbaColor=[0.3, 0.3, 0.35, 1.0]  # Dark gray
        )

        # Create wings (flat box)
        wing_half = [1.0, wingspan / 2, 0.1]
        wing_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=wing_half)
        wing_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=wing_half,
            rgbaColor=[0.35, 0.35, 0.4, 1.0]
        )

        # Create the aircraft as a multi-body
        self._aircraft_id = p.createMultiBody(
            baseMass=0,  # Kinematic body (we control position directly)
            baseCollisionShapeIndex=fuselage_col,
            baseVisualShapeIndex=fuselage_vis,
            basePosition=[0, 0, 100],
            linkMasses=[0],
            linkCollisionShapeIndices=[wing_col],
            linkVisualShapeIndices=[wing_vis],
            linkPositions=[[0, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]]
        )

    def _update_aircraft_visual(self):
        """Update aircraft visual position and orientation."""
        if not hasattr(self, '_aircraft_id') or self._using_airsim:
            return

        p = self._p

        # Convert attitude (roll, pitch, yaw) to quaternion
        roll, pitch, yaw = self._attitude
        quat = p.getQuaternionFromEuler([roll, pitch, yaw])

        # Update position and orientation
        p.resetBasePositionAndOrientation(
            self._aircraft_id,
            self._position.tolist(),
            quat
        )

        # Update camera to follow aircraft
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=80,
                cameraYaw=np.degrees(yaw) + 180,
                cameraPitch=-20,
                cameraTargetPosition=self._position.tolist()
            )

    def _init_physics_models(self):
        """Initialize aerodynamics and propulsion models."""
        from simulation.physics.aerodynamics import (
            FixedWingAerodynamics,
            ISAAtmosphere,
            GroundEffectModel,
            AircraftGeometry,
            StabilityDerivatives,
        )
        from simulation.physics.propulsion import (
            JetEngine,
            create_jet_engine_from_config,
        )
        from simulation.control.fixed_wing import (
            ControlSurfaceMixer,
            FixedWingController,
        )

        physics_params = self.platform_config.get('physics_params', {})

        # Create aircraft geometry
        self._geometry = AircraftGeometry(
            wingspan=physics_params.get('wingspan', 10.0),
            wing_area=physics_params.get('wing_area', 20.0),
            mean_chord=physics_params.get('mean_chord', 2.0),
            aspect_ratio=physics_params.get('aspect_ratio', None),
        )

        # Create stability derivatives from config
        deriv_dict = {
            'CL_0': physics_params.get('CL_0', 0.2),
            'CL_alpha': physics_params.get('CL_alpha', 5.0),
            'CD_0': physics_params.get('CD_0', 0.02),
            'Cm_alpha': physics_params.get('Cm_alpha', -0.5),
        }
        self._derivatives = StabilityDerivatives.from_dict(deriv_dict)

        # Create aerodynamic model
        from simulation.physics.aerodynamics.fixed_wing_aero import (
            FixedWingAerodynamics,
            StallModel,
            create_jet_uav_aero_model,
        )

        self._aero_model = create_jet_uav_aero_model(self.platform_config)

        # Create atmosphere model
        self._atmosphere = ISAAtmosphere()

        # Create ground effect model
        from simulation.physics.aerodynamics.ground_effect import GroundEffectConfig
        self._ground_effect = GroundEffectModel(GroundEffectConfig(
            wingspan=self._geometry.wingspan
        ))

        # Create propulsion model
        self._engine = create_jet_engine_from_config(self.platform_config)

        # Create control surface mixer
        self._mixer = ControlSurfaceMixer.create_from_config(self.platform_config)

        # Inertia tensor
        Ixx = physics_params.get('Ixx', 5000)
        Iyy = physics_params.get('Iyy', 10000)
        Izz = physics_params.get('Izz', 12000)
        self._inertia = np.diag([Ixx, Iyy, Izz])
        self._inertia_inv = np.linalg.inv(self._inertia)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Reset options (e.g., initial_position)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset step counter
        self._current_step = 0

        # Reset fuel
        self._fuel_remaining = self._fuel_capacity

        # Reset weapon state
        self._weapon_state = WeaponState.SAFE

        # Initial state
        options = options or {}
        initial_pos = options.get('initial_position', np.array([0.0, 0.0, 500.0]))
        initial_vel = options.get('initial_velocity', np.array([100.0, 0.0, 0.0]))
        initial_att = options.get('initial_attitude', np.array([0.0, 0.0, 0.0]))

        # Add randomization if enabled
        if options.get('randomize', True):
            initial_pos += self.np_random.uniform(-50, 50, 3)
            initial_pos[2] = max(initial_pos[2], 100)  # Minimum altitude
            initial_att += self.np_random.uniform(-0.1, 0.1, 3)

        # Set initial state
        self._position = initial_pos.copy()
        self._velocity = initial_vel.copy()
        self._attitude = initial_att.copy()
        self._angular_velocity = np.zeros(3)

        # Reset control surfaces
        self._current_surfaces = np.zeros(4)
        self._throttle = 0.5

        # Reset engine
        self._engine.reset()

        # Reset physics backend
        self._reset_physics_backend()

        # Get observation
        obs = self._get_observation()

        info = {
            'fuel_remaining': self._fuel_remaining,
            'weapon_state': self._weapon_state.value,
        }

        return obs.to_array().astype(np.float32), info

    def _reset_physics_backend(self):
        """Reset physics backend state."""
        if self._using_airsim:
            # AirSim reset
            pass
        else:
            # PyBullet reset - create/reset aircraft body
            # Simplified: use a basic body shape
            pass

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: [aileron, elevator, rudder, throttle, flaps]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._current_step += 1

        # Parse action
        aileron_cmd = float(action[0])
        elevator_cmd = float(action[1])
        rudder_cmd = float(action[2])
        throttle = float(action[3])
        flaps_cmd = float(action[4])

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
            surfaces.aileron,
            surfaces.elevator,
            surfaces.rudder,
            surfaces.flaps
        ])
        self._throttle = throttle

        # Run physics simulation
        for _ in range(self.physics_steps_per_control):
            self._physics_step(surfaces, throttle)

        # Update visual
        self._update_aircraft_visual()

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs, action)

        # Check termination
        terminated = self._check_terminated()
        truncated = self._current_step >= self._max_episode_steps

        # Info
        info = {
            'fuel_remaining': self._fuel_remaining,
            'fuel_fraction': self._fuel_remaining / self._fuel_capacity,
            'weapon_state': self._weapon_state.value,
            'airspeed': obs.airspeed,
            'altitude': self._position[2],
            'mach': obs.mach,
        }

        return obs.to_array().astype(np.float32), reward, terminated, truncated, info

    def _physics_step(self, surfaces, throttle: float):
        """Execute one physics step."""
        dt = 1.0 / self.physics_hz

        # Get atmospheric conditions
        altitude = self._position[2]
        atm = self._atmosphere.get_state(altitude)

        # Compute airspeed and aerodynamic angles
        V = np.linalg.norm(self._velocity)
        if V > 1.0:
            # Transform velocity to body frame
            v_body = self._world_to_body(self._velocity)
            u, v, w = v_body

            alpha = np.arctan2(w, max(u, 1.0))
            beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
        else:
            alpha = 0.0
            beta = 0.0
            v_body = np.array([V, 0.0, 0.0])

        # Mach number
        mach = V / atm.speed_of_sound

        # Dynamic pressure
        q_bar = 0.5 * atm.density * V ** 2

        # Create aerodynamic state
        from simulation.physics.aerodynamics import AeroState, ControlSurfaceState
        aero_state = AeroState(
            position=self._position,
            orientation=self._attitude_to_quat(self._attitude),
            velocity_body=v_body,
            angular_velocity=self._angular_velocity,
            altitude=altitude,
            airspeed=V,
            alpha=alpha,
            beta=beta,
            mach=mach,
            dynamic_pressure=q_bar
        )

        control_surfaces = ControlSurfaceState(
            aileron=surfaces.aileron,
            elevator=surfaces.elevator,
            rudder=surfaces.rudder,
            flaps=surfaces.flaps
        )

        # Compute aerodynamic forces and moments
        aero_forces, aero_moments = self._aero_model.compute_forces_moments(
            aero_state, control_surfaces
        )

        # Apply ground effect if close to ground
        if altitude < self._geometry.wingspan:
            lift_factor, drag_factor = self._ground_effect.compute_ground_effect_factors(
                altitude, self._geometry.aspect_ratio
            )
            # Modify forces (simplified)
            aero_forces[2] *= lift_factor  # Increase lift

        # Compute propulsion
        from simulation.physics.propulsion import EngineState
        engine_state = EngineState(
            throttle=throttle,
            altitude=altitude,
            mach=mach,
            airspeed=V,
            fuel_remaining=self._fuel_remaining
        )
        prop_output = self._engine.compute_thrust(engine_state, dt)

        # Thrust force (body x-axis forward)
        thrust_force = np.array([prop_output.thrust, 0.0, 0.0])

        # Update fuel
        self._fuel_remaining -= prop_output.fuel_flow * dt
        self._fuel_remaining = max(0.0, self._fuel_remaining)

        # Total forces in body frame
        total_force_body = aero_forces + thrust_force

        # Add gravity in world frame, transform to body
        gravity_world = np.array([0.0, 0.0, -self._mass * 9.81])
        gravity_body = self._world_to_body(gravity_world)
        total_force_body += gravity_body

        # Linear acceleration (body frame)
        accel_body = total_force_body / self._mass

        # Angular acceleration
        # tau = I * alpha + omega x (I * omega)
        omega = self._angular_velocity
        angular_accel = self._inertia_inv @ (
            aero_moments - np.cross(omega, self._inertia @ omega)
        )

        # Integrate velocities (body frame → world frame for position)
        accel_world = self._body_to_world(accel_body)
        self._velocity += accel_world * dt
        self._position += self._velocity * dt

        # Integrate angular velocity and attitude
        self._angular_velocity += angular_accel * dt

        # Update attitude (simplified Euler integration)
        roll, pitch, yaw = self._attitude
        p, q, r = self._angular_velocity

        roll_dot = p + np.tan(pitch) * (q * np.sin(roll) + r * np.cos(roll))
        pitch_dot = q * np.cos(roll) - r * np.sin(roll)
        yaw_dot = (q * np.sin(roll) + r * np.cos(roll)) / np.cos(pitch) if np.cos(pitch) > 0.01 else 0

        self._attitude[0] += roll_dot * dt
        self._attitude[1] += pitch_dot * dt
        self._attitude[2] += yaw_dot * dt

        # Wrap angles
        self._attitude[0] = np.arctan2(np.sin(self._attitude[0]), np.cos(self._attitude[0]))
        self._attitude[1] = np.clip(self._attitude[1], -np.pi/2 + 0.01, np.pi/2 - 0.01)
        self._attitude[2] = np.arctan2(np.sin(self._attitude[2]), np.cos(self._attitude[2]))

    def _get_observation(self) -> FixedWingObservation:
        """Get current observation."""
        V = np.linalg.norm(self._velocity)
        altitude = self._position[2]

        # Compute aero angles
        if V > 1.0:
            v_body = self._world_to_body(self._velocity)
            u, v, w = v_body
            alpha = np.arctan2(w, max(u, 1.0))
            beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
        else:
            alpha = 0.0
            beta = 0.0

        # Mach number
        atm = self._atmosphere.get_state(altitude)
        mach = V / atm.speed_of_sound

        return FixedWingObservation(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            attitude=self._attitude.copy(),
            angular_velocity=self._angular_velocity.copy(),
            airspeed=V,
            alpha=alpha,
            beta=beta,
            mach=mach,
            fuel_fraction=self._fuel_remaining / self._fuel_capacity,
            control_surfaces=self._current_surfaces.copy()
        )

    def _compute_reward(
        self,
        obs: FixedWingObservation,
        action: np.ndarray
    ) -> float:
        """
        Compute reward for current state.

        Override this in subclasses for mission-specific rewards.
        """
        reward = 0.0

        # Alive bonus
        reward += 1.0

        # Altitude penalty (stay above minimum)
        if obs.position[2] < 100:
            reward -= 5.0 * (100 - obs.position[2]) / 100

        # Distance to target
        dist_to_target = np.linalg.norm(obs.position - self._target_position)
        reward -= 0.001 * dist_to_target

        # Airspeed maintenance
        stall_speed = self.platform_config.get('physics_params', {}).get('stall_speed', 50) / 3.6
        if obs.airspeed < stall_speed * 1.2:
            reward -= 5.0  # Approaching stall

        # Action smoothness
        reward -= 0.01 * np.sum(action[:3] ** 2)

        # Fuel efficiency
        fuel_rate = 1.0 - obs.fuel_fraction
        reward -= 0.001 * fuel_rate

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Ground collision
        if self._position[2] < 0:
            return True

        # Out of bounds
        if abs(self._position[0]) > self._bounds['xy_max']:
            return True
        if abs(self._position[1]) > self._bounds['xy_max']:
            return True
        if self._position[2] > self._bounds['z_max']:
            return True

        # Excessive attitude (loss of control)
        if abs(self._attitude[1]) > np.radians(85):  # Pitch > 85 degrees
            return True

        # Fuel exhaustion (optional - might want to continue for glide)
        # if self._fuel_remaining <= 0:
        #     return True

        return False

    def _world_to_body(self, v_world: np.ndarray) -> np.ndarray:
        """Transform vector from world to body frame."""
        roll, pitch, yaw = self._attitude

        # Rotation matrix (world to body)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp*cy, cp*sy, -sp],
            [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
            [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]
        ])

        return R @ v_world

    def _body_to_world(self, v_body: np.ndarray) -> np.ndarray:
        """Transform vector from body to world frame."""
        roll, pitch, yaw = self._attitude

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_T = np.array([
            [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
            [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
            [-sp, sr*cp, cr*cp]
        ])

        return R_T @ v_body

    def _attitude_to_quat(self, attitude: np.ndarray) -> np.ndarray:
        """Convert Euler angles to quaternion [x, y, z, w]."""
        roll, pitch, yaw = attitude

        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)

        return np.array([
            sr*cp*cy - cr*sp*sy,  # x
            cr*sp*cy + sr*cp*sy,  # y
            cr*cp*sy - sr*sp*cy,  # z
            cr*cp*cy + sr*sp*sy,  # w
        ])

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._using_airsim:
                pass  # AirSim handles its own rendering
            else:
                pass  # PyBullet GUI mode
        elif self.render_mode == "rgb_array":
            # Return camera image
            pass

    def close(self):
        """Clean up resources."""
        if not self._using_airsim:
            try:
                self._p.disconnect()
            except:
                pass

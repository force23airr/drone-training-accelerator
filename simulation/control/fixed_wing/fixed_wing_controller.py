"""
Fixed-Wing Flight Controller.

Implements a cascaded control architecture for fixed-wing UAVs:
- Outer loop: Path/Trajectory following
- Middle loop: Altitude, heading, airspeed control
- Inner loop: Attitude rate control

Supports multiple autopilot modes for different flight phases.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from enum import Enum, auto
import numpy as np

from simulation.control.pid_controller import PIDController, PIDGains
from .control_surface_mixer import ControlSurfaceMixer, ControlSurfaceLimits


class AutopilotMode(Enum):
    """Autopilot operating modes."""
    MANUAL = auto()           # Direct pilot control
    STABILIZE = auto()        # Attitude stabilization only
    ALTITUDE_HOLD = auto()    # Hold altitude, manual heading/roll
    HEADING_HOLD = auto()     # Hold heading, manual altitude
    CRUISE = auto()           # Hold altitude + heading + airspeed
    WAYPOINT = auto()         # Navigate to waypoint
    ORBIT = auto()            # Orbit around point
    TAKEOFF = auto()          # Automatic takeoff
    LANDING = auto()          # Automatic landing approach
    RTL = auto()              # Return to launch


@dataclass
class FixedWingControllerConfig:
    """
    Configuration for fixed-wing controller.
    """

    # Inner loop gains (attitude rate control)
    roll_rate_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=3.0, ki=0.5, kd=0.1, output_min=-1.0, output_max=1.0
    ))
    pitch_rate_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=2.0, ki=0.3, kd=0.05, output_min=-1.0, output_max=1.0
    ))
    yaw_rate_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=1.5, ki=0.1, kd=0.02, output_min=-1.0, output_max=1.0
    ))

    # Middle loop gains (attitude control)
    roll_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=5.0, ki=0.0, kd=0.5, output_min=-3.0, output_max=3.0
    ))
    pitch_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=4.0, ki=0.0, kd=0.3, output_min=-2.0, output_max=2.0
    ))

    # Outer loop gains (navigation)
    altitude_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.1, ki=0.01, kd=0.05,
        output_min=np.radians(-15), output_max=np.radians(15)
    ))
    airspeed_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.5, ki=0.1, kd=0.0, output_min=0.0, output_max=1.0
    ))
    heading_gains: PIDGains = field(default_factory=lambda: PIDGains(
        kp=2.0, ki=0.0, kd=0.2,
        output_min=np.radians(-45), output_max=np.radians(45)
    ))

    # Flight envelope limits
    max_bank_angle: float = np.radians(45.0)
    max_pitch_angle: float = np.radians(30.0)
    min_pitch_angle: float = np.radians(-20.0)
    max_climb_rate: float = 15.0  # m/s
    min_sink_rate: float = -10.0  # m/s

    # Airspeed limits
    stall_speed: float = 50.0   # m/s
    max_speed: float = 250.0    # m/s
    cruise_speed: float = 150.0  # m/s

    # Coordinated turn parameters
    coordinated_turn_gain: float = 1.0

    @classmethod
    def from_platform_config(cls, config: Dict[str, Any]) -> 'FixedWingControllerConfig':
        """Create controller config from platform configuration."""
        physics = config.get('physics_params', {})

        ctrl_config = cls()

        # Update speed limits from platform
        ctrl_config.stall_speed = physics.get('stall_speed', 50.0) / 3.6  # km/h to m/s
        ctrl_config.max_speed = physics.get('max_speed', 500.0) / 3.6
        ctrl_config.cruise_speed = physics.get('cruise_speed', 200.0) / 3.6

        return ctrl_config


class FixedWingController:
    """
    Complete fixed-wing flight controller.

    Implements cascaded control:
    1. Outer loop (navigation): Position/waypoint → altitude/heading targets
    2. Middle loop (guidance): Altitude/heading → pitch/roll targets
    3. Inner loop (stabilization): Pitch/roll targets → surface commands

    The inner loop runs at high rate (50+ Hz) for stability.
    Outer loops can run slower (10-20 Hz).
    """

    def __init__(
        self,
        config: FixedWingControllerConfig = None,
        mixer: ControlSurfaceMixer = None
    ):
        """
        Initialize fixed-wing controller.

        Args:
            config: Controller configuration
            mixer: Control surface mixer
        """
        self.config = config or FixedWingControllerConfig()
        self.mixer = mixer or ControlSurfaceMixer()

        # Initialize PID controllers
        # Inner loop (rate control)
        self.roll_rate_pid = PIDController(self.config.roll_rate_gains)
        self.pitch_rate_pid = PIDController(self.config.pitch_rate_gains)
        self.yaw_rate_pid = PIDController(self.config.yaw_rate_gains)

        # Middle loop (attitude control)
        self.roll_pid = PIDController(self.config.roll_gains)
        self.pitch_pid = PIDController(self.config.pitch_gains)

        # Outer loop (navigation control)
        self.altitude_pid = PIDController(self.config.altitude_gains)
        self.airspeed_pid = PIDController(self.config.airspeed_gains)
        self.heading_pid = PIDController(self.config.heading_gains)

        # Current mode
        self.mode = AutopilotMode.MANUAL

        # Targets
        self._target_altitude = 0.0
        self._target_airspeed = self.config.cruise_speed
        self._target_heading = 0.0
        self._target_roll = 0.0
        self._target_pitch = 0.0

        # State tracking
        self._last_throttle = 0.5

    def compute_control(
        self,
        state: Dict[str, Any],
        target: Dict[str, Any],
        dt: float
    ) -> Tuple[np.ndarray, float]:
        """
        Compute control outputs for the current mode.

        Args:
            state: Current aircraft state
                - position: [x, y, z] in world frame
                - velocity: [vx, vy, vz] in world frame
                - attitude: [roll, pitch, yaw] in radians
                - angular_velocity: [p, q, r] in rad/s
                - airspeed: true airspeed in m/s
            target: Control targets (varies by mode)
                - altitude: target altitude (for ALT_HOLD, CRUISE)
                - airspeed: target airspeed (for CRUISE)
                - heading: target heading (for HDG_HOLD, CRUISE)
                - roll: target roll (for STABILIZE)
                - pitch: target pitch (for STABILIZE)
            dt: Timestep in seconds

        Returns:
            Tuple of (surface_commands [aileron, elevator, rudder, flaps], throttle)
        """
        # Extract state
        roll, pitch, yaw = state['attitude']
        p, q, r = state['angular_velocity']
        altitude = state['position'][2]
        airspeed = state.get('airspeed', np.linalg.norm(state['velocity']))

        # Ensure minimum airspeed for controllability
        airspeed = max(airspeed, self.config.stall_speed * 0.5)

        # Mode-specific outer loop processing
        if self.mode == AutopilotMode.MANUAL:
            # Direct passthrough of target attitudes
            target_roll = target.get('roll', 0.0)
            target_pitch = target.get('pitch', 0.0)
            throttle = target.get('throttle', 0.5)

        elif self.mode == AutopilotMode.STABILIZE:
            # Attitude stabilization
            target_roll = target.get('roll', 0.0)
            target_pitch = target.get('pitch', 0.0)
            throttle = target.get('throttle', 0.5)

        elif self.mode == AutopilotMode.ALTITUDE_HOLD:
            target_alt = target.get('altitude', altitude)
            target_roll = target.get('roll', 0.0)

            # Altitude → pitch
            alt_error = target_alt - altitude
            target_pitch = self.altitude_pid.update(alt_error, dt)

            # Airspeed control
            throttle = self._compute_airspeed_control(
                airspeed, target.get('airspeed', self.config.cruise_speed), dt
            )

        elif self.mode == AutopilotMode.HEADING_HOLD:
            target_hdg = target.get('heading', yaw)
            target_pitch = target.get('pitch', 0.0)

            # Heading → roll
            heading_error = self._wrap_angle(target_hdg - yaw)
            target_roll = self.heading_pid.update(heading_error, dt)
            target_roll = np.clip(target_roll, -self.config.max_bank_angle,
                                 self.config.max_bank_angle)

            throttle = target.get('throttle', 0.5)

        elif self.mode == AutopilotMode.CRUISE:
            target_alt = target.get('altitude', altitude)
            target_hdg = target.get('heading', yaw)
            target_spd = target.get('airspeed', self.config.cruise_speed)

            # Altitude → pitch
            alt_error = target_alt - altitude
            target_pitch = self.altitude_pid.update(alt_error, dt)
            target_pitch = np.clip(target_pitch, self.config.min_pitch_angle,
                                  self.config.max_pitch_angle)

            # Heading → roll (coordinated turn)
            heading_error = self._wrap_angle(target_hdg - yaw)
            target_roll = self.heading_pid.update(heading_error, dt)
            target_roll = np.clip(target_roll, -self.config.max_bank_angle,
                                 self.config.max_bank_angle)

            # Airspeed → throttle
            throttle = self._compute_airspeed_control(airspeed, target_spd, dt)

        elif self.mode == AutopilotMode.ORBIT:
            # Orbit mode: maintain constant bank and altitude
            orbit_radius = target.get('radius', 500.0)
            orbit_alt = target.get('altitude', altitude)

            # Bank angle for coordinated turn at given radius
            # bank = atan(V^2 / (g * R))
            g = 9.81
            target_roll = np.arctan2(airspeed ** 2, g * orbit_radius)
            target_roll = np.clip(target_roll, -self.config.max_bank_angle,
                                 self.config.max_bank_angle)

            # Direction of orbit
            if target.get('clockwise', True):
                target_roll = -target_roll

            # Altitude hold
            alt_error = orbit_alt - altitude
            target_pitch = self.altitude_pid.update(alt_error, dt)

            throttle = self._compute_airspeed_control(
                airspeed, target.get('airspeed', self.config.cruise_speed), dt
            )

        else:
            # Default to stabilize
            target_roll = 0.0
            target_pitch = 0.0
            throttle = 0.5

        # --- Middle Loop: Attitude → Rate Commands ---
        roll_error = target_roll - roll
        pitch_error = target_pitch - pitch

        target_p = self.roll_pid.update(roll_error, dt)
        target_q = self.pitch_pid.update(pitch_error, dt)

        # Coordinated turn: compute required yaw rate
        if abs(roll) > np.radians(5.0) and airspeed > self.config.stall_speed:
            # Required yaw rate for coordinated turn
            g = 9.81
            target_r = g * np.tan(roll) / airspeed * self.config.coordinated_turn_gain
        else:
            target_r = 0.0

        # --- Inner Loop: Rate → Surface Commands ---
        p_error = target_p - p
        q_error = target_q - q
        r_error = target_r - r

        roll_cmd = self.roll_rate_pid.update(p_error, dt)
        pitch_cmd = self.pitch_rate_pid.update(q_error, dt)
        yaw_cmd = self.yaw_rate_pid.update(r_error, dt)

        # Mix to control surfaces
        surfaces = self.mixer.mix(
            roll_cmd=roll_cmd,
            pitch_cmd=pitch_cmd,
            yaw_cmd=yaw_cmd,
            throttle=throttle,
            dt=dt
        )

        # Return as array [aileron, elevator, rudder, flaps] and throttle
        surface_array = np.array([
            surfaces.aileron,
            surfaces.elevator,
            surfaces.rudder,
            surfaces.flaps
        ])

        self._last_throttle = throttle

        return surface_array, throttle

    def _compute_airspeed_control(
        self,
        current: float,
        target: float,
        dt: float
    ) -> float:
        """Compute throttle for airspeed control."""
        # Clamp target to valid range
        target = np.clip(target, self.config.stall_speed * 1.2, self.config.max_speed)

        airspeed_error = target - current
        throttle_correction = self.airspeed_pid.update(airspeed_error, dt)

        # Bias throttle based on current vs target
        base_throttle = 0.5 + 0.3 * (target / self.config.cruise_speed - 1.0)
        throttle = base_throttle + throttle_correction

        return np.clip(throttle, 0.0, 1.0)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def set_mode(self, mode: AutopilotMode):
        """
        Set autopilot mode.

        Resets integrators when changing modes to prevent windup.
        """
        if mode != self.mode:
            self._reset_integrators()
            self.mode = mode

    def _reset_integrators(self):
        """Reset all PID integrators."""
        self.roll_rate_pid.reset()
        self.pitch_rate_pid.reset()
        self.yaw_rate_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.altitude_pid.reset()
        self.airspeed_pid.reset()
        self.heading_pid.reset()

    def reset(self):
        """Full controller reset."""
        self._reset_integrators()
        self.mode = AutopilotMode.MANUAL
        self.mixer.reset()
        self._last_throttle = 0.5

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for telemetry/logging."""
        return {
            'mode': self.mode.name,
            'target_roll': self._target_roll,
            'target_pitch': self._target_pitch,
            'target_altitude': self._target_altitude,
            'target_airspeed': self._target_airspeed,
            'target_heading': self._target_heading,
            'throttle': self._last_throttle,
        }


def create_controller_for_platform(config: Dict[str, Any]) -> FixedWingController:
    """
    Factory function to create controller from platform config.

    Args:
        config: Platform configuration dictionary

    Returns:
        Configured FixedWingController
    """
    # Create controller config from platform
    ctrl_config = FixedWingControllerConfig.from_platform_config(config)

    # Create mixer from platform
    mixer = ControlSurfaceMixer.create_from_config(config)

    # Check for flying wing / elevon configuration
    physics = config.get('physics_params', {})
    if physics.get('control_surface_type', '').lower() == 'elevon':
        mixer = ControlSurfaceMixer.create_for_flying_wing()

    return FixedWingController(config=ctrl_config, mixer=mixer)

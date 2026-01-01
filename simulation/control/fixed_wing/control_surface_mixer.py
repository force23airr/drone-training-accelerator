"""
Control Surface Mixer for Fixed-Wing Aircraft.

Maps pilot/autopilot commands to control surface deflections,
handling limits, mixing, and special configurations like elevons.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np

from simulation.physics.aerodynamics.aerodynamic_model import ControlSurfaceState


class ControlSurfaceType(Enum):
    """Types of control surface configurations."""
    CONVENTIONAL = "conventional"     # Separate aileron, elevator, rudder
    ELEVON = "elevon"                 # Flying wing (combined elevator/aileron)
    V_TAIL = "v_tail"                 # V-tail (combined elevator/rudder)
    TAILERONS = "tailerons"           # All-moving horizontal tail for roll+pitch
    DELTA = "delta"                   # Delta wing configuration


@dataclass
class ControlSurfaceLimits:
    """
    Control surface deflection limits and rates.

    All angles in radians.
    """

    # Maximum deflections (symmetric about zero)
    aileron_max: float = np.radians(25.0)
    elevator_max: float = np.radians(25.0)
    rudder_max: float = np.radians(30.0)
    flap_max: float = np.radians(40.0)
    speedbrake_max: float = np.radians(60.0)

    # Deflection rates [rad/s]
    aileron_rate: float = np.radians(80.0)
    elevator_rate: float = np.radians(80.0)
    rudder_rate: float = np.radians(60.0)
    flap_rate: float = np.radians(5.0)
    speedbrake_rate: float = np.radians(30.0)

    # Trim range (for autopilot trim systems)
    elevator_trim_max: float = np.radians(10.0)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ControlSurfaceLimits':
        """Create from configuration dictionary."""
        return cls(
            aileron_max=np.radians(config.get('aileron_max_deg', 25.0)),
            elevator_max=np.radians(config.get('elevator_max_deg', 25.0)),
            rudder_max=np.radians(config.get('rudder_max_deg', 30.0)),
            flap_max=np.radians(config.get('flap_max_deg', 40.0)),
            speedbrake_max=np.radians(config.get('speedbrake_max_deg', 60.0)),
            aileron_rate=np.radians(config.get('aileron_rate_deg_s', 80.0)),
            elevator_rate=np.radians(config.get('elevator_rate_deg_s', 80.0)),
            rudder_rate=np.radians(config.get('rudder_rate_deg_s', 60.0)),
        )


@dataclass
class MixerGains:
    """
    Mixing gains for control coupling.

    Used to implement features like:
    - Adverse yaw compensation (aileron-rudder interconnect)
    - Coordinated turn mixing
    - Differential aileron
    """

    # Aileron-rudder interconnect (ARI)
    # Applies rudder proportional to aileron to reduce adverse yaw
    aileron_to_rudder: float = 0.0

    # Roll-yaw coupling compensation
    roll_to_yaw: float = 0.0

    # Differential aileron (more up than down)
    aileron_differential: float = 0.0  # 0 = symmetric, 0.5 = 50% more up

    # Flaperon mixing (aileron deflection with flaps)
    flaperon_gain: float = 0.0

    # Elevon mixing gains (for flying wings)
    elevon_pitch_gain: float = 1.0
    elevon_roll_gain: float = 1.0


class ControlSurfaceMixer:
    """
    Maps normalized commands to control surface deflections.

    Handles:
    - Command scaling to physical deflections
    - Control surface limits
    - Rate limiting
    - Control mixing (ARI, elevons, etc.)
    - Trim integration
    """

    def __init__(
        self,
        limits: ControlSurfaceLimits = None,
        mixer_gains: MixerGains = None,
        surface_type: ControlSurfaceType = ControlSurfaceType.CONVENTIONAL,
    ):
        """
        Initialize control surface mixer.

        Args:
            limits: Control surface deflection limits
            mixer_gains: Mixing gains for control coupling
            surface_type: Type of control surface configuration
        """
        self.limits = limits or ControlSurfaceLimits()
        self.gains = mixer_gains or MixerGains()
        self.surface_type = surface_type

        # Current surface positions (for rate limiting)
        self._current_state = ControlSurfaceState()

        # Trim values (added to commands)
        self._trim = ControlSurfaceState()

    def mix(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yaw_cmd: float,
        throttle: float,
        flap_cmd: float = 0.0,
        speedbrake_cmd: float = 0.0,
        dt: float = 0.0
    ) -> ControlSurfaceState:
        """
        Mix pilot commands to control surface deflections.

        Args:
            roll_cmd: Roll command (-1 to 1, positive = roll right)
            pitch_cmd: Pitch command (-1 to 1, positive = pitch up)
            yaw_cmd: Yaw command (-1 to 1, positive = yaw right)
            throttle: Throttle setting (0 to 1, or 0 to 2 with AB)
            flap_cmd: Flap command (0 to 1)
            speedbrake_cmd: Speedbrake command (0 to 1)
            dt: Timestep for rate limiting (0 = no rate limit)

        Returns:
            ControlSurfaceState with surface deflections in radians
        """
        # Apply mixing based on surface type
        if self.surface_type == ControlSurfaceType.ELEVON:
            surfaces = self._mix_elevon(roll_cmd, pitch_cmd, yaw_cmd)
        elif self.surface_type == ControlSurfaceType.V_TAIL:
            surfaces = self._mix_v_tail(roll_cmd, pitch_cmd, yaw_cmd)
        else:
            surfaces = self._mix_conventional(roll_cmd, pitch_cmd, yaw_cmd)

        # Add flaps and speedbrake
        surfaces.flaps = self._scale_and_limit(
            flap_cmd, 0.0, self.limits.flap_max
        )
        surfaces.speedbrake = self._scale_and_limit(
            speedbrake_cmd, 0.0, self.limits.speedbrake_max
        )

        # Add trim
        surfaces = self._apply_trim(surfaces)

        # Apply rate limiting if timestep provided
        if dt > 0:
            surfaces = self._apply_rate_limits(surfaces, dt)

        # Update current state
        self._current_state = surfaces

        return surfaces

    def _mix_conventional(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yaw_cmd: float
    ) -> ControlSurfaceState:
        """
        Mix for conventional control surfaces.
        """
        # Direct mapping with limits
        aileron = self._scale_and_limit(
            roll_cmd,
            -self.limits.aileron_max,
            self.limits.aileron_max
        )

        elevator = self._scale_and_limit(
            pitch_cmd,
            -self.limits.elevator_max,
            self.limits.elevator_max
        )

        # Yaw with aileron-rudder interconnect (ARI)
        ari_contribution = self.gains.aileron_to_rudder * roll_cmd
        yaw_total = yaw_cmd + ari_contribution

        rudder = self._scale_and_limit(
            yaw_total,
            -self.limits.rudder_max,
            self.limits.rudder_max
        )

        return ControlSurfaceState(
            aileron=aileron,
            elevator=elevator,
            rudder=rudder
        )

    def _mix_elevon(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yaw_cmd: float
    ) -> ControlSurfaceState:
        """
        Mix for elevon (flying wing) configuration.

        Elevons combine elevator and aileron function:
        - Both up = pitch up
        - Differential = roll
        """
        # Combine pitch and roll into elevon commands
        pitch_contribution = pitch_cmd * self.gains.elevon_pitch_gain
        roll_contribution = roll_cmd * self.gains.elevon_roll_gain

        # Left elevon: pitch up + roll left = up
        elevon_left = pitch_contribution - roll_contribution

        # Right elevon: pitch up + roll right = up
        elevon_right = pitch_contribution + roll_contribution

        # Scale to limits (use elevator limit for elevons)
        max_deflection = self.limits.elevator_max

        elevon_left = np.clip(elevon_left, -1.0, 1.0) * max_deflection
        elevon_right = np.clip(elevon_right, -1.0, 1.0) * max_deflection

        # Flying wings may use split rudders or drag rudders for yaw
        # Simplified: use a small drag-based yaw
        rudder = self._scale_and_limit(
            yaw_cmd,
            -self.limits.rudder_max * 0.5,  # Reduced authority
            self.limits.rudder_max * 0.5
        )

        return ControlSurfaceState(
            aileron=elevon_left - elevon_right,  # Store differential
            elevator=(elevon_left + elevon_right) / 2,  # Store average
            rudder=rudder,
            elevon_left=elevon_left,
            elevon_right=elevon_right
        )

    def _mix_v_tail(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yaw_cmd: float
    ) -> ControlSurfaceState:
        """
        Mix for V-tail (ruddervator) configuration.

        V-tail surfaces combine elevator and rudder:
        - Both up = pitch up
        - Differential = yaw
        """
        # V-tail mixing
        ruddervator_left = pitch_cmd - yaw_cmd
        ruddervator_right = pitch_cmd + yaw_cmd

        # Scale to elevator limits
        max_deflection = self.limits.elevator_max

        ruddervator_left = np.clip(ruddervator_left, -1.0, 1.0) * max_deflection
        ruddervator_right = np.clip(ruddervator_right, -1.0, 1.0) * max_deflection

        # Ailerons are separate
        aileron = self._scale_and_limit(
            roll_cmd,
            -self.limits.aileron_max,
            self.limits.aileron_max
        )

        return ControlSurfaceState(
            aileron=aileron,
            elevator=(ruddervator_left + ruddervator_right) / 2,
            rudder=(ruddervator_right - ruddervator_left) / 2
        )

    def _scale_and_limit(
        self,
        cmd: float,
        min_val: float,
        max_val: float
    ) -> float:
        """Scale normalized command to physical limits."""
        if min_val < 0:
            # Symmetric limits: cmd in [-1, 1]
            return np.clip(cmd, -1.0, 1.0) * max_val
        else:
            # One-sided limit: cmd in [0, 1]
            return np.clip(cmd, 0.0, 1.0) * max_val

    def _apply_trim(self, surfaces: ControlSurfaceState) -> ControlSurfaceState:
        """Add trim deflections to surfaces."""
        return ControlSurfaceState(
            aileron=surfaces.aileron + self._trim.aileron,
            elevator=surfaces.elevator + self._trim.elevator,
            rudder=surfaces.rudder + self._trim.rudder,
            flaps=surfaces.flaps,
            speedbrake=surfaces.speedbrake,
            elevon_left=surfaces.elevon_left,
            elevon_right=surfaces.elevon_right
        )

    def _apply_rate_limits(
        self,
        target: ControlSurfaceState,
        dt: float
    ) -> ControlSurfaceState:
        """Apply rate limiting to surface movements."""
        def rate_limit(current: float, target: float, max_rate: float) -> float:
            delta = target - current
            max_delta = max_rate * dt
            return current + np.clip(delta, -max_delta, max_delta)

        return ControlSurfaceState(
            aileron=rate_limit(
                self._current_state.aileron,
                target.aileron,
                self.limits.aileron_rate
            ),
            elevator=rate_limit(
                self._current_state.elevator,
                target.elevator,
                self.limits.elevator_rate
            ),
            rudder=rate_limit(
                self._current_state.rudder,
                target.rudder,
                self.limits.rudder_rate
            ),
            flaps=rate_limit(
                self._current_state.flaps,
                target.flaps,
                self.limits.flap_rate
            ),
            speedbrake=rate_limit(
                self._current_state.speedbrake,
                target.speedbrake,
                self.limits.speedbrake_rate
            ),
            elevon_left=target.elevon_left,
            elevon_right=target.elevon_right
        )

    def set_trim(
        self,
        aileron_trim: float = 0.0,
        elevator_trim: float = 0.0,
        rudder_trim: float = 0.0
    ):
        """
        Set trim deflections.

        Args:
            aileron_trim: Aileron trim [rad]
            elevator_trim: Elevator trim [rad]
            rudder_trim: Rudder trim [rad]
        """
        self._trim = ControlSurfaceState(
            aileron=np.clip(aileron_trim, -self.limits.aileron_max * 0.5,
                          self.limits.aileron_max * 0.5),
            elevator=np.clip(elevator_trim, -self.limits.elevator_trim_max,
                           self.limits.elevator_trim_max),
            rudder=np.clip(rudder_trim, -self.limits.rudder_max * 0.3,
                         self.limits.rudder_max * 0.3)
        )

    def get_trim(self) -> ControlSurfaceState:
        """Get current trim settings."""
        return self._trim

    def reset(self):
        """Reset mixer state."""
        self._current_state = ControlSurfaceState()
        self._trim = ControlSurfaceState()

    @staticmethod
    def create_for_flying_wing() -> 'ControlSurfaceMixer':
        """Create mixer configured for flying wing (elevon) aircraft."""
        limits = ControlSurfaceLimits(
            aileron_max=np.radians(30.0),  # Elevons have more travel
            elevator_max=np.radians(30.0),
            rudder_max=np.radians(20.0),   # Limited yaw authority
        )

        gains = MixerGains(
            elevon_pitch_gain=1.0,
            elevon_roll_gain=0.8,  # Slightly less roll authority
        )

        return ControlSurfaceMixer(
            limits=limits,
            mixer_gains=gains,
            surface_type=ControlSurfaceType.ELEVON
        )

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> 'ControlSurfaceMixer':
        """
        Create mixer from platform configuration.

        Args:
            config: Platform config with physics_params

        Returns:
            Configured mixer
        """
        physics = config.get('physics_params', {})

        # Determine surface type
        surface_type_str = physics.get('control_surface_type', 'conventional')
        surface_type = ControlSurfaceType(surface_type_str.lower())

        # Create limits
        limits = ControlSurfaceLimits.from_config(physics)

        # Create gains
        gains = MixerGains(
            aileron_to_rudder=physics.get('ari_gain', 0.1),
            aileron_differential=physics.get('aileron_differential', 0.0),
        )

        return ControlSurfaceMixer(
            limits=limits,
            mixer_gains=gains,
            surface_type=surface_type
        )

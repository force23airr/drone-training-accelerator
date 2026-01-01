"""
Propulsion Model Base Classes.

Defines the abstract interface for aircraft propulsion systems,
including state tracking and output computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EngineState:
    """
    Current state of the propulsion system.

    Attributes:
        throttle: Throttle setting (0-1 for military power, 1-2 for afterburner)
        altitude: Current altitude [m]
        mach: Current Mach number
        airspeed: True airspeed [m/s]
        fuel_remaining: Remaining fuel mass [kg]
        temperature_ambient: Outside air temperature [K]
        n1: Fan/compressor spool speed (% of max)
        n2: Core spool speed (% of max) - for two-spool engines
    """

    throttle: float = 0.0
    altitude: float = 0.0
    mach: float = 0.0
    airspeed: float = 0.0
    fuel_remaining: float = 1000.0
    temperature_ambient: float = 288.15

    # Internal engine state
    n1: float = 0.0  # Low-pressure spool (%)
    n2: float = 0.0  # High-pressure spool (%)

    @property
    def is_afterburner_active(self) -> bool:
        """Check if afterburner is engaged."""
        return self.throttle > 1.0

    @property
    def afterburner_fraction(self) -> float:
        """Get afterburner usage fraction (0 if not engaged)."""
        if self.throttle > 1.0:
            return min(self.throttle - 1.0, 1.0)
        return 0.0

    @property
    def military_throttle(self) -> float:
        """Get throttle in military power range (0-1)."""
        return min(self.throttle, 1.0)


@dataclass
class PropulsionOutput:
    """
    Output from propulsion system computation.

    Attributes:
        thrust: Net thrust [N]
        fuel_flow: Fuel consumption rate [kg/s]
        sfc: Specific fuel consumption [kg/(N*s)]
        exhaust_velocity: Exhaust gas velocity [m/s]
        thrust_available: Maximum thrust available at current conditions
        n1_actual: Actual N1 spool speed (%)
        n2_actual: Actual N2 spool speed (%)
    """

    thrust: float = 0.0
    fuel_flow: float = 0.0
    sfc: float = 0.0
    exhaust_velocity: float = 0.0
    thrust_available: float = 0.0
    n1_actual: float = 0.0
    n2_actual: float = 0.0

    @property
    def is_producing_thrust(self) -> bool:
        """Check if engine is producing significant thrust."""
        return self.thrust > 100.0  # 100 N threshold

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for logging/analysis."""
        return np.array([
            self.thrust,
            self.fuel_flow,
            self.sfc,
            self.thrust_available,
            self.n1_actual,
            self.n2_actual
        ])


class PropulsionModel(ABC):
    """
    Abstract base class for propulsion system models.

    Subclasses implement specific engine types:
    - Turbofan (high bypass for efficiency)
    - Turbojet (low bypass for speed)
    - Turboprop
    - Electric
    """

    @abstractmethod
    def compute_thrust(
        self,
        engine_state: EngineState,
        dt: float = 0.0
    ) -> PropulsionOutput:
        """
        Compute thrust output for current engine state.

        Args:
            engine_state: Current engine and flight state
            dt: Timestep for dynamics (0 for steady-state)

        Returns:
            PropulsionOutput with thrust, fuel flow, etc.
        """
        pass

    @abstractmethod
    def get_max_thrust(self, altitude: float, mach: float) -> float:
        """
        Get maximum available thrust at conditions.

        Args:
            altitude: Altitude [m]
            mach: Mach number

        Returns:
            Maximum thrust [N]
        """
        pass

    @abstractmethod
    def get_idle_thrust(self, altitude: float, mach: float) -> float:
        """
        Get idle thrust at conditions.

        Args:
            altitude: Altitude [m]
            mach: Mach number

        Returns:
            Idle thrust [N]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset engine state to default/startup condition."""
        pass

    def compute_range_factor(
        self,
        altitude: float,
        mach: float,
        CL: float,
        CD: float
    ) -> float:
        """
        Compute Breguet range factor for cruise optimization.

        Range = (V / SFC) * (L/D) * ln(W_initial / W_final)

        Args:
            altitude: Cruise altitude [m]
            mach: Cruise Mach number
            CL: Lift coefficient
            CD: Drag coefficient

        Returns:
            Range factor (V / SFC) * (L/D) in meters
        """
        # This is a default implementation; subclasses may override
        from simulation.physics.aerodynamics.atmosphere_model import ISAAtmosphere

        atm = ISAAtmosphere()
        state = atm.get_state(altitude)
        velocity = mach * state.speed_of_sound

        # Get SFC at cruise
        engine_state = EngineState(
            throttle=0.7,  # Typical cruise throttle
            altitude=altitude,
            mach=mach,
            airspeed=velocity
        )
        output = self.compute_thrust(engine_state)

        if output.sfc > 0 and CD > 0:
            L_D = CL / CD
            return (velocity / output.sfc) * L_D
        return 0.0

    def estimate_fuel_to_climb(
        self,
        altitude_start: float,
        altitude_end: float,
        mach: float,
        weight: float
    ) -> float:
        """
        Estimate fuel required for climb.

        Simplified model assuming constant Mach climb.

        Args:
            altitude_start: Starting altitude [m]
            altitude_end: Target altitude [m]
            mach: Climb Mach number
            weight: Aircraft weight [N]

        Returns:
            Estimated fuel consumption [kg]
        """
        from simulation.physics.aerodynamics.atmosphere_model import ISAAtmosphere

        atm = ISAAtmosphere()
        altitude_mid = (altitude_start + altitude_end) / 2
        state = atm.get_state(altitude_mid)
        velocity = mach * state.speed_of_sound

        # Estimate climb rate (simplified)
        thrust = self.get_max_thrust(altitude_mid, mach)
        drag_estimate = weight * 0.1  # Rough drag estimate
        excess_thrust = max(thrust - drag_estimate, 0)

        if excess_thrust > 0:
            climb_rate = excess_thrust * velocity / weight  # m/s
            climb_time = (altitude_end - altitude_start) / max(climb_rate, 1.0)

            # Get fuel flow at climb power
            engine_state = EngineState(
                throttle=1.0,  # Max power climb
                altitude=altitude_mid,
                mach=mach,
                airspeed=velocity
            )
            output = self.compute_thrust(engine_state)

            return output.fuel_flow * climb_time

        return float('inf')  # Cannot climb

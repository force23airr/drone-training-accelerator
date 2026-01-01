"""
Jet Engine Propulsion Model.

Implements turbofan and turbojet engine models with:
- Altitude and Mach number thrust variation
- Afterburner augmentation
- Spool-up/down dynamics
- Fuel consumption modeling
- Temperature effects
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .propulsion_model import PropulsionModel, EngineState, PropulsionOutput
from simulation.physics.aerodynamics.atmosphere_model import ISAAtmosphere, AtmosphereState


@dataclass
class JetEngineConfig:
    """
    Jet engine configuration parameters.

    Supports turbofan (bypass_ratio > 0) and turbojet (bypass_ratio = 0) engines.
    """

    # Basic thrust parameters
    max_thrust_sl: float = 50000.0      # Sea level static thrust [N]
    idle_thrust_fraction: float = 0.05   # Idle thrust as fraction of max

    # Afterburner (optional)
    has_afterburner: bool = False
    afterburner_thrust_ratio: float = 1.5  # AB thrust / military thrust

    # Engine type
    bypass_ratio: float = 0.3           # 0 = turbojet, >0 = turbofan
    overall_pressure_ratio: float = 25.0  # Compressor pressure ratio

    # Fuel consumption
    sfc_military: float = 0.00008       # Specific fuel consumption at mil power [kg/(N*s)]
    sfc_afterburner: float = 0.00020    # SFC with afterburner [kg/(N*s)]
    sfc_idle: float = 0.00015           # SFC at idle

    # Altitude/Mach lapse
    altitude_lapse_exponent: float = 0.7  # Thrust ~ (rho/rho_sl)^n
    mach_ram_coefficient: float = 0.2     # Ram effect coefficient

    # Thrust-Mach curve (optional custom curve)
    mach_thrust_curve: List[Tuple[float, float]] = None  # [(M, factor), ...]

    # Operating limits
    max_altitude: float = 18000.0       # Service ceiling [m]
    max_mach: float = 2.0               # Maximum operating Mach
    max_turbine_inlet_temp: float = 1700.0  # TIT limit [K]

    # Spool dynamics
    spool_time_constant: float = 2.0    # Time constant for spool-up [s]
    spool_rate_limit: float = 0.3       # Max spool rate [%/s]

    # Bleed air effects
    bleed_thrust_penalty: float = 0.02  # Thrust loss per unit bleed

    @classmethod
    def turbofan_high_bypass(cls, max_thrust: float) -> 'JetEngineConfig':
        """Create config for high-bypass turbofan (commercial/HALE UAV)."""
        return cls(
            max_thrust_sl=max_thrust,
            bypass_ratio=5.0,
            overall_pressure_ratio=35.0,
            sfc_military=0.00005,
            sfc_idle=0.0001,
            altitude_lapse_exponent=0.8,
            mach_ram_coefficient=0.1,
            max_mach=0.9,
            has_afterburner=False,
        )

    @classmethod
    def turbofan_low_bypass(cls, max_thrust: float, with_ab: bool = True) -> 'JetEngineConfig':
        """Create config for low-bypass turbofan (fighter/UCAV)."""
        return cls(
            max_thrust_sl=max_thrust,
            bypass_ratio=0.3,
            overall_pressure_ratio=25.0,
            sfc_military=0.00008,
            sfc_afterburner=0.0002,
            sfc_idle=0.00012,
            altitude_lapse_exponent=0.7,
            mach_ram_coefficient=0.2,
            max_mach=2.0 if with_ab else 1.2,
            has_afterburner=with_ab,
            afterburner_thrust_ratio=1.6 if with_ab else 1.0,
        )

    @classmethod
    def turbojet(cls, max_thrust: float) -> 'JetEngineConfig':
        """Create config for pure turbojet engine."""
        return cls(
            max_thrust_sl=max_thrust,
            bypass_ratio=0.0,
            overall_pressure_ratio=15.0,
            sfc_military=0.0001,
            sfc_idle=0.00015,
            altitude_lapse_exponent=0.65,
            mach_ram_coefficient=0.25,
            max_mach=2.5,
            has_afterburner=False,
        )

    @classmethod
    def small_turbine(cls, max_thrust: float) -> 'JetEngineConfig':
        """Create config for small turbine (loitering munition)."""
        return cls(
            max_thrust_sl=max_thrust,
            bypass_ratio=0.0,
            overall_pressure_ratio=8.0,
            sfc_military=0.00012,
            sfc_idle=0.0002,
            altitude_lapse_exponent=0.6,
            mach_ram_coefficient=0.15,
            max_mach=0.7,
            max_altitude=9000.0,
            has_afterburner=False,
            spool_time_constant=1.0,
        )


class JetEngine(PropulsionModel):
    """
    Jet engine model with dynamics and fuel consumption.

    Models thrust variation with:
    - Altitude (density ratio)
    - Mach number (ram effect)
    - Throttle setting (including afterburner)
    - Spool dynamics (response lag)
    - Temperature effects
    """

    def __init__(self, config: JetEngineConfig):
        """
        Initialize jet engine model.

        Args:
            config: Engine configuration parameters
        """
        self.config = config
        self.atmosphere = ISAAtmosphere()

        # Internal state
        self._n1_actual = 0.0  # Current N1 spool position (0-1)
        self._n1_commanded = 0.0
        self._fuel_consumed_total = 0.0

        # Pre-compute common values
        self._thrust_at_idle = config.max_thrust_sl * config.idle_thrust_fraction

    def compute_thrust(
        self,
        engine_state: EngineState,
        dt: float = 0.0
    ) -> PropulsionOutput:
        """
        Compute engine thrust output.

        Args:
            engine_state: Current engine and flight state
            dt: Timestep for dynamics (0 for instantaneous)

        Returns:
            PropulsionOutput with thrust and fuel flow
        """
        # Get atmospheric conditions
        atm = self.atmosphere.get_state(engine_state.altitude)

        # Update spool dynamics if dt provided
        if dt > 0:
            self._update_spool(engine_state.throttle, dt)
            effective_throttle = self._n1_actual
        else:
            effective_throttle = min(engine_state.throttle, 1.0)

        # Compute altitude effect (density ratio)
        density_ratio = atm.density / self.atmosphere.RHO0
        altitude_factor = density_ratio ** self.config.altitude_lapse_exponent

        # Compute Mach effect (ram pressure)
        mach = engine_state.mach
        mach_factor = self._compute_mach_factor(mach)

        # Temperature correction
        temp_factor = self._compute_temperature_factor(atm.temperature)

        # Base thrust at current conditions
        thrust_available = (self.config.max_thrust_sl *
                           altitude_factor *
                           mach_factor *
                           temp_factor)

        # Apply throttle
        if engine_state.is_afterburner_active and self.config.has_afterburner:
            # Afterburner regime
            military_thrust = thrust_available * effective_throttle
            ab_thrust_max = thrust_available * self.config.afterburner_thrust_ratio
            ab_fraction = engine_state.afterburner_fraction
            thrust = military_thrust + ab_fraction * (ab_thrust_max - military_thrust)
            sfc = self.config.sfc_afterburner
        else:
            # Military/partial power
            idle_thrust = self._thrust_at_idle * altitude_factor * mach_factor
            thrust = idle_thrust + effective_throttle * (thrust_available - idle_thrust)

            # Interpolate SFC based on throttle
            if effective_throttle < 0.1:
                sfc = self.config.sfc_idle
            else:
                sfc = (self.config.sfc_idle +
                       effective_throttle * (self.config.sfc_military - self.config.sfc_idle))

        # Ensure non-negative thrust
        thrust = max(thrust, 0.0)

        # Fuel flow
        fuel_flow = thrust * sfc

        # Check fuel availability
        if engine_state.fuel_remaining <= 0:
            thrust = 0.0
            fuel_flow = 0.0

        # Update total fuel consumed
        if dt > 0:
            self._fuel_consumed_total += fuel_flow * dt

        return PropulsionOutput(
            thrust=thrust,
            fuel_flow=fuel_flow,
            sfc=sfc,
            exhaust_velocity=self._estimate_exhaust_velocity(thrust, fuel_flow),
            thrust_available=thrust_available,
            n1_actual=self._n1_actual * 100,
            n2_actual=self._n1_actual * 100 * 0.95,  # Simplified N2
        )

    def _compute_mach_factor(self, mach: float) -> float:
        """
        Compute Mach number effect on thrust.

        Ram effect increases thrust at high Mach.
        """
        # Use custom curve if provided
        if self.config.mach_thrust_curve:
            mach_points = [p[0] for p in self.config.mach_thrust_curve]
            factors = [p[1] for p in self.config.mach_thrust_curve]
            return np.interp(mach, mach_points, factors)

        # Default: linear ram effect with saturation
        if mach < 0.1:
            return 1.0
        elif mach < 0.8:
            # Slight increase due to ram effect
            return 1.0 + self.config.mach_ram_coefficient * mach
        elif mach < 1.2:
            # Transonic - ram effect continues but with drag rise
            return 1.0 + self.config.mach_ram_coefficient * 0.8
        else:
            # Supersonic - significant ram effect
            return 1.0 + self.config.mach_ram_coefficient * (0.8 + 0.3 * (mach - 1.2))

    def _compute_temperature_factor(self, temperature: float) -> float:
        """
        Compute temperature effect on thrust.

        Hot day reduces thrust, cold day increases it.
        """
        # Reference: ISA sea level = 288.15 K
        temp_ratio = temperature / 288.15

        # Thrust varies inversely with temperature (approximately)
        # Hotter air = less dense = less thrust
        # Also affects compressor efficiency
        return np.sqrt(1.0 / temp_ratio)

    def _update_spool(self, throttle_commanded: float, dt: float):
        """
        Update engine spool dynamics.

        Models the lag between throttle command and thrust response.
        """
        # Target N1 based on throttle
        target_n1 = min(throttle_commanded, 1.0)

        # First-order lag response
        tau = self.config.spool_time_constant

        # Spool-up is faster than spool-down
        if target_n1 > self._n1_actual:
            tau_effective = tau * 0.7  # Faster spool-up
        else:
            tau_effective = tau * 1.3  # Slower spool-down

        # Rate limit
        n1_error = target_n1 - self._n1_actual
        rate = n1_error / tau_effective
        rate = np.clip(rate, -self.config.spool_rate_limit, self.config.spool_rate_limit)

        # Update spool position
        self._n1_actual += rate * dt
        self._n1_actual = np.clip(self._n1_actual, 0.0, 1.0)

    def _estimate_exhaust_velocity(self, thrust: float, fuel_flow: float) -> float:
        """
        Estimate exhaust velocity from thrust and fuel flow.

        V_exhaust ≈ Thrust / (fuel_flow + air_mass_flow)
        Simplified estimation.
        """
        if fuel_flow < 1e-6:
            return 0.0

        # Approximate air-to-fuel ratio
        afr = 50.0  # Typical for jet engines

        total_mass_flow = fuel_flow * (1 + afr)
        if total_mass_flow > 0:
            return thrust / total_mass_flow
        return 0.0

    def get_max_thrust(self, altitude: float, mach: float) -> float:
        """Get maximum thrust at conditions."""
        state = EngineState(
            throttle=1.0,
            altitude=altitude,
            mach=mach
        )
        output = self.compute_thrust(state, dt=0)
        return output.thrust_available

    def get_idle_thrust(self, altitude: float, mach: float) -> float:
        """Get idle thrust at conditions."""
        state = EngineState(
            throttle=0.0,
            altitude=altitude,
            mach=mach
        )
        output = self.compute_thrust(state, dt=0)
        return output.thrust

    def get_afterburner_thrust(self, altitude: float, mach: float) -> float:
        """Get maximum afterburner thrust at conditions."""
        if not self.config.has_afterburner:
            return self.get_max_thrust(altitude, mach)

        state = EngineState(
            throttle=2.0,  # Full afterburner
            altitude=altitude,
            mach=mach
        )
        output = self.compute_thrust(state, dt=0)
        return output.thrust

    def reset(self):
        """Reset engine to startup state."""
        self._n1_actual = 0.0
        self._n1_commanded = 0.0
        self._fuel_consumed_total = 0.0

    def get_fuel_consumed(self) -> float:
        """Get total fuel consumed since last reset [kg]."""
        return self._fuel_consumed_total

    def estimate_endurance(
        self,
        fuel_available: float,
        altitude: float,
        mach: float,
        throttle: float = 0.7
    ) -> float:
        """
        Estimate flight endurance at given conditions.

        Args:
            fuel_available: Available fuel [kg]
            altitude: Cruise altitude [m]
            mach: Cruise Mach
            throttle: Throttle setting (0-1)

        Returns:
            Estimated endurance [hours]
        """
        atm = self.atmosphere.get_state(altitude)
        airspeed = mach * atm.speed_of_sound

        state = EngineState(
            throttle=throttle,
            altitude=altitude,
            mach=mach,
            airspeed=airspeed,
            fuel_remaining=fuel_available
        )

        output = self.compute_thrust(state, dt=0)

        if output.fuel_flow > 0:
            endurance_seconds = fuel_available / output.fuel_flow
            return endurance_seconds / 3600.0  # Convert to hours

        return float('inf')

    def estimate_range(
        self,
        fuel_available: float,
        altitude: float,
        mach: float,
        throttle: float = 0.7
    ) -> float:
        """
        Estimate flight range at given conditions.

        Simplified Breguet range equation.

        Args:
            fuel_available: Available fuel [kg]
            altitude: Cruise altitude [m]
            mach: Cruise Mach
            throttle: Throttle setting

        Returns:
            Estimated range [km]
        """
        atm = self.atmosphere.get_state(altitude)
        airspeed = mach * atm.speed_of_sound

        endurance_hours = self.estimate_endurance(
            fuel_available, altitude, mach, throttle
        )

        if endurance_hours < float('inf'):
            return airspeed * endurance_hours * 3600 / 1000  # km

        return float('inf')


def create_jet_engine_from_config(config: Dict[str, Any]) -> JetEngine:
    """
    Factory function to create jet engine from platform config.

    Args:
        config: Platform configuration with physics_params

    Returns:
        Configured JetEngine instance
    """
    physics = config.get('physics_params', {})
    engine_type = physics.get('engine_type', 'turbofan')

    max_thrust = physics.get('max_thrust_sl', 50000.0)

    if engine_type == 'turbofan':
        bypass = physics.get('bypass_ratio', 0.3)
        if bypass > 2.0:
            engine_config = JetEngineConfig.turbofan_high_bypass(max_thrust)
        else:
            has_ab = physics.get('afterburner_thrust_sl', 0) > 0
            engine_config = JetEngineConfig.turbofan_low_bypass(max_thrust, has_ab)
            if has_ab:
                ab_thrust = physics.get('afterburner_thrust_sl', max_thrust * 1.5)
                engine_config.afterburner_thrust_ratio = ab_thrust / max_thrust

    elif engine_type == 'turbojet':
        engine_config = JetEngineConfig.turbojet(max_thrust)

    elif engine_type in ('turbine_small', 'small_turbine'):
        engine_config = JetEngineConfig.small_turbine(max_thrust)

    elif engine_type == 'turboprop':
        # Treat turboprop similarly to high-bypass turbofan for thrust
        engine_config = JetEngineConfig.turbofan_high_bypass(max_thrust)
        engine_config.max_mach = 0.6

    else:
        # Default to low-bypass turbofan
        engine_config = JetEngineConfig.turbofan_low_bypass(max_thrust, False)

    # Override with any explicit config values
    if 'sfc' in physics:
        engine_config.sfc_military = physics['sfc']
    if 'max_altitude' in physics or 'service_ceiling' in physics:
        engine_config.max_altitude = physics.get('service_ceiling',
                                                  physics.get('max_altitude', 18000))
    if 'max_mach' in physics:
        engine_config.max_mach = physics['max_mach']

    return JetEngine(engine_config)

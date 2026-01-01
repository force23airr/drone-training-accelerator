"""
International Standard Atmosphere (ISA) Model.

Implements the ISA model for computing atmospheric properties
(temperature, pressure, density, speed of sound) as a function of altitude.

Reference: ICAO Standard Atmosphere (ISO 2533:1975)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AtmosphereState:
    """Atmospheric state at a given altitude."""

    temperature: float      # Temperature [K]
    pressure: float         # Pressure [Pa]
    density: float          # Density [kg/m^3]
    speed_of_sound: float   # Speed of sound [m/s]
    dynamic_viscosity: float  # Dynamic viscosity [Pa*s]

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity [m^2/s]."""
        return self.dynamic_viscosity / self.density


class ISAAtmosphere:
    """
    International Standard Atmosphere (ISA) model.

    Computes atmospheric properties from sea level to 86 km altitude.
    Uses the standard temperature profile with discrete layers:
    - Troposphere (0-11 km): Linear temperature decrease
    - Tropopause (11-20 km): Isothermal
    - Stratosphere (20-32 km): Linear temperature increase
    - Stratosphere (32-47 km): Linear temperature increase (steeper)
    - Stratopause (47-51 km): Isothermal
    - Mesosphere (51-71 km): Linear temperature decrease
    - Mesosphere (71-86 km): Linear temperature decrease (steeper)

    Attributes:
        T0: Sea level temperature [K]
        P0: Sea level pressure [Pa]
        RHO0: Sea level density [kg/m^3]
        GAMMA: Ratio of specific heats for air
        R: Gas constant for air [J/(kg*K)]
        G0: Gravitational acceleration [m/s^2]
    """

    # Sea level standard conditions
    T0 = 288.15     # K (15°C)
    P0 = 101325.0   # Pa
    RHO0 = 1.225    # kg/m^3

    # Physical constants
    GAMMA = 1.4                    # Ratio of specific heats (cp/cv)
    R = 287.05287                  # Gas constant for air [J/(kg*K)]
    G0 = 9.80665                   # Standard gravity [m/s^2]
    SUTHERLAND_C = 120.0           # Sutherland's constant [K]
    MU0 = 1.716e-5                 # Reference viscosity [Pa*s]

    # Atmospheric layer definitions: (base_altitude, lapse_rate, base_temp)
    # Lapse rate in K/m, negative means temperature decreases with altitude
    LAYERS = [
        (0.0,     -0.0065, 288.15),    # Troposphere
        (11000.0,  0.0,    216.65),    # Tropopause (isothermal)
        (20000.0,  0.001,  216.65),    # Stratosphere lower
        (32000.0,  0.0028, 228.65),    # Stratosphere upper
        (47000.0,  0.0,    270.65),    # Stratopause (isothermal)
        (51000.0, -0.0028, 270.65),    # Mesosphere lower
        (71000.0, -0.002,  214.65),    # Mesosphere upper
        (86000.0,  0.0,    186.946),   # Upper limit
    ]

    def __init__(self, sea_level_temp_offset: float = 0.0):
        """
        Initialize atmosphere model.

        Args:
            sea_level_temp_offset: Temperature deviation from ISA [K]
                                   Positive = hotter than standard
        """
        self.temp_offset = sea_level_temp_offset

    def get_state(self, altitude_m: float) -> AtmosphereState:
        """
        Get atmospheric state at given altitude.

        Args:
            altitude_m: Geometric altitude above mean sea level [m]

        Returns:
            AtmosphereState with all atmospheric properties

        Note:
            For altitudes below 0 or above 86 km, values are extrapolated
            but may be physically inaccurate.
        """
        # Clamp altitude to valid range (with warning for out of range)
        h = max(0.0, min(altitude_m, 86000.0))

        # Find the appropriate atmospheric layer
        T, P = self._compute_temp_pressure(h)

        # Apply temperature offset (non-standard atmosphere)
        T += self.temp_offset

        # Compute density from ideal gas law: P = rho * R * T
        rho = P / (self.R * T)

        # Speed of sound: a = sqrt(gamma * R * T)
        a = np.sqrt(self.GAMMA * self.R * T)

        # Dynamic viscosity using Sutherland's law
        mu = self._compute_viscosity(T)

        return AtmosphereState(
            temperature=T,
            pressure=P,
            density=rho,
            speed_of_sound=a,
            dynamic_viscosity=mu
        )

    def _compute_temp_pressure(self, h: float) -> tuple:
        """
        Compute temperature and pressure at altitude.

        Uses the barometric formula for each layer.
        """
        # Start from sea level
        T = self.T0
        P = self.P0

        for i in range(len(self.LAYERS) - 1):
            h_base, lapse, T_base = self.LAYERS[i]
            h_top = self.LAYERS[i + 1][0]

            if h <= h_base:
                break

            # Altitude within this layer
            h_in_layer = min(h, h_top) - h_base

            if abs(lapse) < 1e-10:
                # Isothermal layer
                T = T_base
                P = P * np.exp(-self.G0 * h_in_layer / (self.R * T))
            else:
                # Linear temperature gradient
                T_new = T_base + lapse * h_in_layer
                P = P * (T_new / T) ** (-self.G0 / (lapse * self.R))
                T = T_new

            if h <= h_top:
                break

        return T, P

    def _compute_viscosity(self, T: float) -> float:
        """
        Compute dynamic viscosity using Sutherland's law.

        mu = mu0 * (T/T0)^(3/2) * (T0 + C) / (T + C)

        where C is Sutherland's constant.
        """
        T0_ref = 273.15  # Reference temperature for Sutherland
        return (self.MU0 *
                (T / T0_ref) ** 1.5 *
                (T0_ref + self.SUTHERLAND_C) / (T + self.SUTHERLAND_C))

    def compute_mach(self, velocity: float, altitude: float) -> float:
        """
        Compute Mach number.

        Args:
            velocity: True airspeed [m/s]
            altitude: Altitude [m]

        Returns:
            Mach number (dimensionless)
        """
        atm = self.get_state(altitude)
        return velocity / atm.speed_of_sound

    def compute_dynamic_pressure(
        self,
        velocity: float,
        altitude: float
    ) -> float:
        """
        Compute dynamic pressure.

        q = 0.5 * rho * V^2

        Args:
            velocity: True airspeed [m/s]
            altitude: Altitude [m]

        Returns:
            Dynamic pressure [Pa]
        """
        atm = self.get_state(altitude)
        return 0.5 * atm.density * velocity ** 2

    def compute_reynolds(
        self,
        velocity: float,
        characteristic_length: float,
        altitude: float
    ) -> float:
        """
        Compute Reynolds number.

        Re = rho * V * L / mu = V * L / nu

        Args:
            velocity: True airspeed [m/s]
            characteristic_length: Reference length (e.g., chord) [m]
            altitude: Altitude [m]

        Returns:
            Reynolds number (dimensionless)
        """
        atm = self.get_state(altitude)
        return (atm.density * velocity * characteristic_length /
                atm.dynamic_viscosity)

    def compute_calibrated_airspeed(
        self,
        true_airspeed: float,
        altitude: float
    ) -> float:
        """
        Convert true airspeed (TAS) to calibrated airspeed (CAS).

        CAS is the airspeed that would produce the same dynamic pressure
        at sea level as the TAS produces at altitude.

        Args:
            true_airspeed: True airspeed [m/s]
            altitude: Altitude [m]

        Returns:
            Calibrated airspeed [m/s]
        """
        q = self.compute_dynamic_pressure(true_airspeed, altitude)
        # CAS = sqrt(2 * q / rho0)
        return np.sqrt(2 * q / self.RHO0)

    def compute_true_airspeed(
        self,
        calibrated_airspeed: float,
        altitude: float
    ) -> float:
        """
        Convert calibrated airspeed (CAS) to true airspeed (TAS).

        Args:
            calibrated_airspeed: Calibrated airspeed [m/s]
            altitude: Altitude [m]

        Returns:
            True airspeed [m/s]
        """
        atm = self.get_state(altitude)
        # TAS = CAS * sqrt(rho0 / rho)
        return calibrated_airspeed * np.sqrt(self.RHO0 / atm.density)

    def density_altitude(self, pressure_altitude: float, temperature: float) -> float:
        """
        Compute density altitude from pressure altitude and actual temperature.

        Density altitude is the altitude in the standard atmosphere
        that has the same air density as the current conditions.

        Args:
            pressure_altitude: Pressure altitude [m]
            temperature: Actual outside air temperature [K]

        Returns:
            Density altitude [m]
        """
        # Get standard temperature at pressure altitude
        std_state = self.get_state(pressure_altitude)
        T_std = std_state.temperature - self.temp_offset  # Remove any offset

        # Density altitude correction
        # DA = PA + 120 * (T - T_std) approximately for troposphere
        # More accurate: use density ratio
        delta_T = temperature - T_std

        # Approximate correction (valid in troposphere)
        return pressure_altitude + 120.0 * delta_T


# Pre-instantiated standard atmosphere for convenience
STANDARD_ATMOSPHERE = ISAAtmosphere()


def get_air_properties(altitude_m: float) -> AtmosphereState:
    """
    Convenience function to get air properties at altitude.

    Args:
        altitude_m: Altitude above mean sea level [m]

    Returns:
        AtmosphereState with temperature, pressure, density, etc.
    """
    return STANDARD_ATMOSPHERE.get_state(altitude_m)

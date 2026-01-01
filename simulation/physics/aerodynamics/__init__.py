"""
Aerodynamics module for fixed-wing UAV simulation.

This module provides:
- ISA atmosphere model for altitude-dependent air properties
- Full 6-DOF aerodynamic force/moment computation
- Ground effect modeling for takeoff/landing
- RCS (radar cross-section) modeling for stealth aircraft
"""

from .atmosphere_model import (
    AtmosphereState,
    ISAAtmosphere,
)
from .aerodynamic_model import (
    AeroState,
    AeroCoefficients,
    ControlSurfaceState,
    AerodynamicModel,
    AircraftGeometry,
)
from .fixed_wing_aero import FixedWingAerodynamics, StabilityDerivatives
from .ground_effect import GroundEffectModel

__all__ = [
    # Atmosphere
    "AtmosphereState",
    "ISAAtmosphere",
    # Aerodynamics
    "AeroState",
    "AeroCoefficients",
    "ControlSurfaceState",
    "AerodynamicModel",
    "AircraftGeometry",
    "FixedWingAerodynamics",
    "StabilityDerivatives",
    # Ground effect
    "GroundEffectModel",
]

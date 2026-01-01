"""
Fixed-Wing Control Module.

Provides control systems for fixed-wing UAVs:
- Control surface mixer (aileron/elevator/rudder)
- Fixed-wing flight controller with autopilot modes
- Terrain following controller
"""

from .control_surface_mixer import (
    ControlSurfaceLimits,
    ControlSurfaceMixer,
)
from .fixed_wing_controller import (
    FixedWingController,
    FixedWingControllerConfig,
    AutopilotMode,
)

__all__ = [
    # Mixer
    "ControlSurfaceLimits",
    "ControlSurfaceMixer",
    # Controller
    "FixedWingController",
    "FixedWingControllerConfig",
    "AutopilotMode",
]

"""
Propulsion module for UAV simulation.

This module provides:
- Abstract propulsion model interface
- Turbofan/turbojet jet engine model
- Fuel system modeling
- Engine dynamics (spool-up/down)
"""

from .propulsion_model import (
    EngineState,
    PropulsionOutput,
    PropulsionModel,
)
from .jet_engine import (
    JetEngineConfig,
    JetEngine,
    create_jet_engine_from_config,
)

__all__ = [
    # Base classes
    "EngineState",
    "PropulsionOutput",
    "PropulsionModel",
    # Jet engine
    "JetEngineConfig",
    "JetEngine",
    "create_jet_engine_from_config",
]

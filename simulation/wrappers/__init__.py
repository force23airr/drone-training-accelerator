"""
Simulation Wrappers Module

Gym wrappers for simulation enhancement.
"""

from simulation.wrappers.randomization_wrapper import (
    DomainRandomizationWrapper,
    CurriculumRandomizationWrapper,
    make_randomized_env,
)

from simulation.wrappers.safety_shield import (
    ShieldAction,
    SafetyLimits,
    ShieldConfig,
    ShieldState,
    SafetyShieldWrapper,
    make_shielded_env,
)
from simulation.wrappers.action_adapter import (
    ActionAdapterConfig,
    ActionAdapterWrapper,
    make_action_adapted_env,
)
from simulation.wrappers.observation_adapter import (
    ObservationAdapterConfig,
    ObservationAdapterWrapper,
    make_observation_adapted_env,
)

__all__ = [
    # Domain randomization
    "DomainRandomizationWrapper",
    "CurriculumRandomizationWrapper",
    "make_randomized_env",
    # Safety shield
    "ShieldAction",
    "SafetyLimits",
    "ShieldConfig",
    "ShieldState",
    "SafetyShieldWrapper",
    "make_shielded_env",
    # Action adapter
    "ActionAdapterConfig",
    "ActionAdapterWrapper",
    "make_action_adapted_env",
    # Observation adapter
    "ObservationAdapterConfig",
    "ObservationAdapterWrapper",
    "make_observation_adapted_env",
]

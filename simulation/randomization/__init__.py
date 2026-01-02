"""
Domain Randomization Module

Sim2Real transfer via physics, sensor, and actuator randomization.
"""

from simulation.randomization.domain_randomization import (
    DistributionType,
    RandomizationRange,
    PhysicsRandomizationConfig,
    ActuatorRandomizationConfig,
    SensorRandomizationConfig,
    EnvironmentRandomizationConfig,
    DomainRandomizationConfig,
    SampledDomainParams,
    DomainRandomizer,
    create_curriculum_configs,
)

__all__ = [
    "DistributionType",
    "RandomizationRange",
    "PhysicsRandomizationConfig",
    "ActuatorRandomizationConfig",
    "SensorRandomizationConfig",
    "EnvironmentRandomizationConfig",
    "DomainRandomizationConfig",
    "SampledDomainParams",
    "DomainRandomizer",
    "create_curriculum_configs",
]

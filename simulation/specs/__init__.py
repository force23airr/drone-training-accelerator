"""
Customer Drone Specification System

Enables startups, universities, and manufacturers to test their
custom drone designs in the combat simulator.

Usage:
    from simulation.specs import DroneSpec, DroneSpecLoader

    # Load customer specification
    loader = DroneSpecLoader()
    spec = loader.load_from_yaml("my_drone.yaml")

    # Validate
    result = loader.validate(spec)
    if result.errors:
        print("Errors:", result.errors)

    # Convert to simulator config
    config = loader.to_dogfight_config(spec)
"""

from .drone_spec import (
    DroneSpec,
    PropulsionType,
    ControlSurfaceType,
    DroneCategory,
)

from .aero_estimator import (
    AeroEstimator,
    EstimatedAero,
)

from .validator import (
    SpecValidator,
    ValidationResult,
    ValidationError,
    ValidationWarning,
)

from .spec_loader import (
    DroneSpecLoader,
)

__all__ = [
    # Spec
    "DroneSpec",
    "PropulsionType",
    "ControlSurfaceType",
    "DroneCategory",
    # Estimator
    "AeroEstimator",
    "EstimatedAero",
    # Validator
    "SpecValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    # Loader
    "DroneSpecLoader",
]

"""
Custom Drone Onboarding System

Import YOUR drone into the training platform.
Supports any drone: quadcopters, fixed-wing, VTOLs, jets, custom builds.

Workflow:
1. Define your drone's specs (or import from CAD/URDF)
2. System validates and auto-completes missing parameters
3. Generates simulation model matched to your hardware
4. Train autonomous behaviors
5. Export to your flight controller

Usage:
    from onboarding import DroneSpecification, onboard_drone

    # Define your drone
    my_drone = DroneSpecification(
        name="My Custom Drone",
        airframe_type="quadcopter",
        mass_kg=2.5,
        ...
    )

    # Onboard to platform
    sim_config = onboard_drone(my_drone)

    # Train!
    train_drone(sim_config, mission="hover_stability")
"""

from onboarding.drone_specification import (
    DroneSpecification,
    AirframeType,
    MotorConfiguration,
    PropulsionType,
    FlightControllerType,
    SensorSuite,
    BatterySpecification,
    MotorSpecification,
    PropellerSpecification,
    AerodynamicSurfaces,
    WingSpecification,
    ControlSurfaceSpecification,
    JetEngineSpecification,
    PerformanceEnvelope,
    PayloadSpecification,
    CommunicationSystem,
    InertialProperties,
)

from onboarding.spec_validator import (
    validate_specification,
    auto_complete_specs,
    estimate_missing_parameters,
)

from onboarding.cad_importer import (
    import_from_urdf,
    import_from_step,
    import_from_stl,
    import_from_json,
    export_to_urdf,
)

from onboarding.simulation_generator import (
    generate_simulation_config,
    onboard_drone,
)

from onboarding.onboarding_cli import (
    interactive_onboarding,
)

__all__ = [
    # Core specification
    "DroneSpecification",
    "AirframeType",
    "MotorConfiguration",
    "PropulsionType",
    "FlightControllerType",
    "SensorSuite",
    "BatterySpecification",
    "MotorSpecification",
    "PropellerSpecification",
    "AerodynamicSurfaces",
    "WingSpecification",
    "ControlSurfaceSpecification",
    "JetEngineSpecification",
    "PerformanceEnvelope",
    "PayloadSpecification",
    "CommunicationSystem",
    "InertialProperties",
    # Validation
    "validate_specification",
    "auto_complete_specs",
    "estimate_missing_parameters",
    # Import/Export
    "import_from_urdf",
    "import_from_step",
    "import_from_stl",
    "import_from_json",
    "export_to_urdf",
    # Onboarding
    "generate_simulation_config",
    "onboard_drone",
    "interactive_onboarding",
]

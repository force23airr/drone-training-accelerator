"""
Simulation Model Generator

Converts a DroneSpecification into a simulation-ready configuration
that can be used with PyBullet, AirSim, or any supported backend.

This is the bridge between user specs and actual training.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

from onboarding.drone_specification import (
    DroneSpecification,
    AirframeType,
    PropulsionType,
    MotorConfiguration,
)
from onboarding.spec_validator import full_process, ValidationResult


def generate_simulation_config(
    spec: DroneSpecification,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate simulation configuration from drone specification.

    This creates the configuration dict used by BaseDroneEnv and other
    simulation environments.

    Args:
        spec: Validated drone specification
        output_dir: Optional directory to save config files

    Returns:
        Configuration dictionary ready for simulation
    """
    config = {
        "name": spec.name,
        "platform_type": _map_airframe_to_platform_type(spec.airframe_type),
        "manufacturer": spec.manufacturer,
        "model": spec.model,

        # Physical properties
        "mass": spec.mass_kg,
        "arm_length": spec.arm_length_m,
        "dimensions": {
            "length": spec.length_m,
            "width": spec.width_m,
            "height": spec.height_m,
        },

        # Inertia
        "inertia": {
            "ixx": spec.inertial.ixx,
            "iyy": spec.inertial.iyy,
            "izz": spec.inertial.izz,
            "ixy": spec.inertial.ixy,
            "ixz": spec.inertial.ixz,
            "iyz": spec.inertial.iyz,
        },
        "center_of_gravity": {
            "x": spec.inertial.cg_x,
            "y": spec.inertial.cg_y,
            "z": spec.inertial.cg_z,
        },

        # Propulsion
        "num_motors": spec.num_motors,
        "motor_config": spec.motor_config.value,
        "motors": [],
        "propellers": [],

        # Thrust characteristics
        "max_thrust_per_motor": 0.0,
        "max_rpm": 0.0,
        "thrust_coefficient": 0.0,
        "torque_coefficient": 0.0,

        # Aerodynamics
        "drag_coefficient_xy": spec.drag_coefficient_xy,
        "drag_coefficient_z": spec.drag_coefficient_z,
        "frontal_area": spec.frontal_area_m2,
        "ground_effect_height": spec.ground_effect_height_m,
        "ground_effect_coefficient": spec.ground_effect_coefficient,

        # Performance
        "max_speed": spec.max_speed_m_s,
        "max_climb_rate": spec.max_climb_rate_m_s,
        "max_thrust_to_weight": spec.max_thrust_to_weight,
        "hover_throttle": spec.hover_throttle,

        # Control rates
        "max_roll_rate": spec.max_roll_rate,
        "max_pitch_rate": spec.max_pitch_rate,
        "max_yaw_rate": spec.max_yaw_rate,

        # Battery
        "battery": {
            "capacity_mah": spec.battery.capacity_mah,
            "cell_count": spec.battery.cell_count,
            "voltage_nominal": spec.battery.pack_voltage_nominal,
        },

        # Sensors
        "sensors": {
            "has_gps": spec.sensors.has_gps,
            "has_imu": spec.sensors.has_imu,
            "has_barometer": spec.sensors.has_barometer,
            "has_magnetometer": spec.sensors.has_magnetometer,
            "has_optical_flow": spec.sensors.has_optical_flow,
            "has_lidar": spec.sensors.has_lidar,
            "has_camera": spec.sensors.has_camera,
            "gps_accuracy_m": spec.sensors.gps_accuracy_m,
        },

        # Physics parameters (for PyBullet)
        "physics_params": {
            "motor_time_constant": 0.05,
            "motor_response_delay": 0.01,
        },

        # Source specification
        "source": "custom_onboarding",
        "spec_version": spec.version,
    }

    # Process motors
    for motor in spec.motors:
        motor_config = {
            "id": motor.motor_id,
            "position": [motor.position_x, motor.position_y, motor.position_z],
            "spin_direction": motor.spin_direction,
            "kv": motor.kv_rating,
            "max_rpm": motor.max_rpm,
            "thrust_coefficient": motor.thrust_coefficient_kf,
            "torque_coefficient": motor.torque_coefficient_km,
            "time_constant": motor.time_constant_s,
        }

        if motor.can_tilt:
            motor_config["tilt"] = {
                "enabled": True,
                "min_angle": motor.min_tilt_deg,
                "max_angle": motor.max_tilt_deg,
                "rate": motor.tilt_rate_deg_s,
            }

        config["motors"].append(motor_config)

        # Update global max values
        if motor.max_rpm > config["max_rpm"]:
            config["max_rpm"] = motor.max_rpm
        if motor.thrust_coefficient_kf > config["thrust_coefficient"]:
            config["thrust_coefficient"] = motor.thrust_coefficient_kf
            config["torque_coefficient"] = motor.torque_coefficient_km

    # Process propellers
    for prop in spec.propellers:
        config["propellers"].append({
            "id": prop.propeller_id,
            "motor_id": prop.motor_id,
            "diameter_m": prop.diameter_m,
            "pitch_inch": prop.pitch_inch,
            "num_blades": prop.num_blades,
            "ct": prop.ct,
            "cp": prop.cp,
        })

    # Calculate max thrust per motor
    if config["max_rpm"] > 0 and config["thrust_coefficient"] > 0:
        config["max_thrust_per_motor"] = config["thrust_coefficient"] * (config["max_rpm"] ** 2)
    else:
        # Estimate from thrust-to-weight
        total_thrust = spec.mass_kg * 9.81 * spec.max_thrust_to_weight
        config["max_thrust_per_motor"] = total_thrust / spec.num_motors if spec.num_motors > 0 else 0

    # Fixed-wing specific
    if "fixed_wing" in spec.airframe_type.value:
        config["aerodynamics"] = {
            "reference_area": spec.aerodynamics.reference_area_m2,
            "reference_chord": spec.aerodynamics.reference_chord_m,
            "reference_span": spec.aerodynamics.reference_span_m,
            "cl_alpha": spec.aerodynamics.cl_alpha,
            "cd_0": spec.aerodynamics.cd_0,
            "cm_alpha": spec.aerodynamics.cm_alpha,
            "oswald_efficiency": 0.8,
        }

        # Wings
        config["wings"] = []
        for wing in spec.aerodynamics.wings:
            config["wings"].append({
                "id": wing.wing_id,
                "type": wing.wing_type,
                "span": wing.span_m,
                "area": wing.area_m2,
                "aspect_ratio": wing.aspect_ratio,
                "sweep": wing.sweep_deg,
                "dihedral": wing.dihedral_deg,
                "cl_alpha": wing.cl_alpha,
                "cl_max": wing.cl_max,
                "stall_angle": wing.stall_angle_deg,
            })

        # Control surfaces
        config["control_surfaces"] = []
        for surface in spec.aerodynamics.control_surfaces:
            config["control_surfaces"].append({
                "id": surface.surface_id,
                "type": surface.surface_type,
                "wing_id": surface.wing_id,
                "area": surface.area_m2,
                "max_deflection_up": surface.max_deflection_up_deg,
                "max_deflection_down": surface.max_deflection_down_deg,
                "effectiveness": surface.effectiveness,
            })

    # Jet engines (if any)
    if len(spec.jet_engines) > 0:
        config["jet_engines"] = []
        for engine in spec.jet_engines:
            config["jet_engines"].append({
                "id": engine.engine_id,
                "type": engine.engine_type.value,
                "position": [engine.position_x, engine.position_y, engine.position_z],
                "thrust_vector": [engine.thrust_vector_x, engine.thrust_vector_y, engine.thrust_vector_z],
                "max_thrust_sl": engine.max_thrust_sea_level_n,
                "sfc": engine.sfc_kg_per_n_hr,
                "spool_time": engine.spool_time_s,
            })

    # Save config if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        config_file = output_path / f"{spec.name.lower().replace(' ', '_')}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved simulation config to: {config_file}")

    return config


def _map_airframe_to_platform_type(airframe: AirframeType) -> str:
    """Map airframe type to platform type string."""
    mapping = {
        AirframeType.QUADCOPTER: "quadcopter",
        AirframeType.QUADCOPTER_X: "quadcopter",
        AirframeType.QUADCOPTER_PLUS: "quadcopter",
        AirframeType.QUADCOPTER_H: "quadcopter",
        AirframeType.HEXACOPTER: "hexacopter",
        AirframeType.HEXACOPTER_X: "hexacopter",
        AirframeType.OCTOCOPTER: "octocopter",
        AirframeType.OCTOCOPTER_X: "octocopter",
        AirframeType.FIXED_WING_CONVENTIONAL: "fixed_wing",
        AirframeType.FIXED_WING_FLYING_WING: "fixed_wing",
        AirframeType.VTOL_TILTROTOR: "vtol",
        AirframeType.VTOL_QUADPLANE: "vtol",
        AirframeType.VTOL_TAILSITTER: "vtol",
    }
    return mapping.get(airframe, "custom")


class OnboardingResult:
    """Result of drone onboarding process."""

    def __init__(self):
        self.success: bool = False
        self.spec: Optional[DroneSpecification] = None
        self.simulation_config: Optional[Dict[str, Any]] = None
        self.validation: Optional[ValidationResult] = None
        self.errors: list = []
        self.warnings: list = []

    def summary(self) -> str:
        lines = []
        if self.success:
            lines.append("ONBOARDING SUCCESSFUL")
            lines.append(f"\nDrone: {self.spec.name}")
            lines.append(f"Type: {self.spec.airframe_type.value}")
            lines.append(f"Mass: {self.spec.mass_kg:.2f} kg")
            lines.append(f"Motors: {self.spec.num_motors}")
        else:
            lines.append("ONBOARDING FAILED")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.validation and self.validation.auto_completed:
            lines.append(f"\nAuto-completed fields ({len(self.validation.auto_completed)}):")
            for a in self.validation.auto_completed:
                lines.append(f"  - {a}")

        return "\n".join(lines)


def onboard_drone(
    spec: DroneSpecification,
    output_dir: Optional[str] = None,
    auto_complete: bool = True,
) -> OnboardingResult:
    """
    Complete onboarding process for a custom drone.

    This is the main entry point for users to bring their drones
    into the training platform.

    Args:
        spec: Drone specification (can be partial)
        output_dir: Optional directory to save generated files
        auto_complete: Whether to estimate missing parameters

    Returns:
        OnboardingResult with status and generated configs
    """
    result = OnboardingResult()

    try:
        # Step 1: Process and validate specification
        print(f"Processing drone specification: {spec.name}")

        if auto_complete:
            spec, validation = full_process(spec)
        else:
            from onboarding.spec_validator import validate_specification
            validation = validate_specification(spec)

        result.validation = validation
        result.errors = validation.errors
        result.warnings = validation.warnings

        if not validation.is_valid:
            print(f"Validation failed with {len(validation.errors)} errors")
            return result

        result.spec = spec

        # Step 2: Generate simulation configuration
        print("Generating simulation configuration...")
        sim_config = generate_simulation_config(spec, output_dir)
        result.simulation_config = sim_config

        # Step 3: Register with platform
        print("Registering drone with training platform...")
        _register_custom_drone(spec, sim_config)

        result.success = True
        print(f"Onboarding complete! Drone '{spec.name}' is ready for training.")

    except Exception as e:
        result.errors.append(f"Onboarding error: {str(e)}")
        import traceback
        traceback.print_exc()

    return result


def _register_custom_drone(spec: DroneSpecification, config: Dict[str, Any]):
    """Register the custom drone with the platform's platform_configs."""
    # This integrates with the existing platform system
    try:
        from simulation.platforms.platform_configs import register_platform

        # Create platform config compatible with existing system
        platform_config = {
            "name": spec.name,
            "platform_type": config["platform_type"],
            "num_motors": spec.num_motors,
            "mass": spec.mass_kg,
            "arm_length": spec.arm_length_m,
            "max_thrust_per_motor": config["max_thrust_per_motor"],
            "max_rpm": config["max_rpm"],
            "physics_params": {
                "ixx": spec.inertial.ixx,
                "iyy": spec.inertial.iyy,
                "izz": spec.inertial.izz,
                "thrust_coefficient": config["thrust_coefficient"],
                "torque_coefficient": config["torque_coefficient"],
                "drag_coefficient": spec.drag_coefficient_xy,
            },
            # Custom fields
            "custom": True,
            "source_spec": spec.name,
        }

        # Generate platform ID from name
        platform_id = spec.name.lower().replace(" ", "_").replace("-", "_")

        register_platform(platform_id, platform_config)
        print(f"Registered as platform: '{platform_id}'")

    except ImportError:
        print("Warning: Could not register with platform system")
    except Exception as e:
        print(f"Warning: Platform registration failed: {e}")


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def quick_onboard_quadcopter(
    name: str,
    mass_kg: float,
    arm_length_m: Optional[float] = None,
    motor_kv: Optional[float] = None,
) -> OnboardingResult:
    """
    Quickly onboard a quadcopter with minimal information.

    Only mass is required - everything else will be estimated.
    """
    from onboarding.drone_specification import create_quadcopter_spec

    arm_length = arm_length_m or (0.1 * (mass_kg ** 0.4))
    kv = motor_kv or (3000 * (mass_kg ** -0.5))

    spec = create_quadcopter_spec(
        name=name,
        mass_kg=mass_kg,
        arm_length_m=arm_length,
        motor_kv=kv,
    )

    return onboard_drone(spec)


def quick_onboard_fixed_wing(
    name: str,
    mass_kg: float,
    wingspan_m: float,
) -> OnboardingResult:
    """
    Quickly onboard a fixed-wing aircraft with minimal information.
    """
    from onboarding.drone_specification import create_fixed_wing_spec

    # Estimate wing area from wingspan (typical AR ~ 8)
    wing_area = (wingspan_m ** 2) / 8

    spec = create_fixed_wing_spec(
        name=name,
        mass_kg=mass_kg,
        wingspan_m=wingspan_m,
        wing_area_m2=wing_area,
    )

    return onboard_drone(spec)

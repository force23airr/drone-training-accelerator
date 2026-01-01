"""
Specification Validator and Auto-Completion

Validates drone specifications and estimates missing parameters using:
- Physics-based estimation
- Empirical correlations from existing drone data
- Machine learning models (optional)

The goal: User provides what they know, we figure out the rest.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import fields
import warnings

from onboarding.drone_specification import (
    DroneSpecification,
    AirframeType,
    PropulsionType,
    MotorConfiguration,
    InertialProperties,
    MotorSpecification,
    PropellerSpecification,
    BatterySpecification,
    WingSpecification,
    PerformanceEnvelope,
)


# =============================================================================
# VALIDATION
# =============================================================================

class ValidationResult:
    """Result of validation check."""
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.auto_completed: List[str] = []
        self.confidence_score: float = 1.0

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_auto_completed(self, field: str, value: Any, method: str):
        self.auto_completed.append(f"{field} = {value} (estimated via {method})")

    def summary(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("VALIDATION PASSED")
        else:
            lines.append("VALIDATION FAILED")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.auto_completed:
            lines.append(f"\nAuto-completed ({len(self.auto_completed)}):")
            for a in self.auto_completed:
                lines.append(f"  - {a}")

        lines.append(f"\nConfidence Score: {self.confidence_score:.1%}")
        return "\n".join(lines)


def validate_specification(spec: DroneSpecification) -> ValidationResult:
    """
    Validate a drone specification for completeness and physical consistency.

    Returns:
        ValidationResult with errors, warnings, and confidence score
    """
    result = ValidationResult()

    # Required fields
    if not spec.name:
        result.add_error("Drone name is required")

    if spec.mass_kg <= 0:
        result.add_error("Mass must be positive")

    # Physical consistency checks
    if spec.airframe_type in [AirframeType.QUADCOPTER, AirframeType.QUADCOPTER_X,
                               AirframeType.QUADCOPTER_PLUS, AirframeType.QUADCOPTER_H]:
        if spec.num_motors != 4:
            result.add_warning(f"Quadcopter should have 4 motors, got {spec.num_motors}")

    if spec.airframe_type in [AirframeType.HEXACOPTER, AirframeType.HEXACOPTER_X]:
        if spec.num_motors != 6:
            result.add_warning(f"Hexacopter should have 6 motors, got {spec.num_motors}")

    if spec.airframe_type in [AirframeType.OCTOCOPTER, AirframeType.OCTOCOPTER_X]:
        if spec.num_motors != 8:
            result.add_warning(f"Octocopter should have 8 motors, got {spec.num_motors}")

    # Thrust-to-weight check
    if spec.max_thrust_to_weight > 0 and spec.max_thrust_to_weight < 1.0:
        result.add_error(f"Thrust-to-weight ratio {spec.max_thrust_to_weight} < 1.0 - cannot fly!")
    elif spec.max_thrust_to_weight > 0 and spec.max_thrust_to_weight < 1.5:
        result.add_warning(f"Low thrust-to-weight ratio {spec.max_thrust_to_weight} - limited maneuverability")

    # Battery sanity check
    if spec.battery.capacity_mah > 0 and spec.battery.cell_count > 0:
        if spec.battery.cell_count > 12:
            result.add_warning(f"Very high cell count ({spec.battery.cell_count}S) - verify this is correct")

    # Motor positions check for multirotors
    if spec.num_motors > 0 and len(spec.motors) > 0:
        if len(spec.motors) != spec.num_motors:
            result.add_warning(f"Motor count mismatch: {len(spec.motors)} defined vs {spec.num_motors} expected")

    # Speed sanity check
    if spec.max_speed_m_s > 100:
        result.add_warning(f"Very high max speed ({spec.max_speed_m_s} m/s) - verify this is correct")

    # Fixed-wing specific checks
    if "fixed_wing" in spec.airframe_type.value.lower():
        if len(spec.aerodynamics.wings) == 0:
            result.add_error("Fixed-wing aircraft requires wing specification")

        if spec.performance.stall_speed_m_s <= 0:
            result.add_warning("Stall speed not specified for fixed-wing")

    # Calculate confidence score based on completeness
    total_fields = 0
    filled_fields = 0

    critical_fields = [
        ('mass_kg', spec.mass_kg > 0),
        ('num_motors', spec.num_motors > 0),
        ('arm_length_m', spec.arm_length_m > 0 or "fixed_wing" in spec.airframe_type.value),
        ('max_speed_m_s', spec.max_speed_m_s > 0),
    ]

    for name, is_filled in critical_fields:
        total_fields += 1
        if is_filled:
            filled_fields += 1

    result.confidence_score = filled_fields / total_fields if total_fields > 0 else 0

    # Store results in spec
    spec.validation_errors = result.errors
    spec.validation_warnings = result.warnings
    spec.is_validated = result.is_valid

    return result


# =============================================================================
# AUTO-COMPLETION / PARAMETER ESTIMATION
# =============================================================================

class WeightClass:
    """UAV weight classifications."""
    MICRO = "micro"           # < 250g (0.55 lbs)
    MINI = "mini"             # 250g - 2kg (0.55 - 4.4 lbs)
    SMALL = "small"           # 2kg - 25kg (4.4 - 55 lbs)
    MEDIUM = "medium"         # 25kg - 150kg (55 - 330 lbs)
    LARGE = "large"           # 150kg - 600kg (330 - 1320 lbs)
    HEAVY = "heavy"           # > 600kg (> 1320 lbs)

    @staticmethod
    def classify(mass_kg: float) -> str:
        if mass_kg < 0.25:
            return WeightClass.MICRO
        elif mass_kg < 2:
            return WeightClass.MINI
        elif mass_kg < 25:
            return WeightClass.SMALL
        elif mass_kg < 150:
            return WeightClass.MEDIUM
        elif mass_kg < 600:
            return WeightClass.LARGE
        else:
            return WeightClass.HEAVY


class ParameterEstimator:
    """
    Estimates missing drone parameters using physics and empirical data.

    Supports full range from micro drones (250g) to heavy military UAVs (5000+ kg).

    Methods:
    - Physics-based: Use equations of motion
    - Empirical: Use correlations from drone database
    - Lookup: Use manufacturer data tables
    """

    # Empirical data from existing drones - FULL SPECTRUM
    # Format: (mass_kg, arm_length_m, typical_motor_kv, typical_prop_inch, propulsion_type)
    QUADCOPTER_DATA = [
        # Micro/Mini (electric)
        (0.025, 0.035, 19000, 1.5, "electric"),   # Tiny whoop 25g
        (0.25, 0.065, 3800, 3, "electric"),       # Tiny whoop 250g
        (0.5, 0.12, 2300, 5, "electric"),         # 5" racing quad
        (1.0, 0.18, 920, 10, "electric"),         # 10" quad
        (2.0, 0.25, 700, 13, "electric"),         # Camera drone (DJI Mavic class)
        (5.0, 0.35, 400, 17, "electric"),         # Heavy lift (DJI M600 class)
        (10.0, 0.50, 300, 22, "electric"),        # Industrial (Freefly Alta X class)

        # Medium (25-75 kg / 55-165 lbs) - YOUR RANGE
        (25.0, 0.75, 200, 28, "electric"),        # Large industrial (Skyfront Perimeter)
        (35.0, 0.85, 170, 30, "hybrid"),          # Heavy-lift cargo (~77 lbs)
        (50.0, 1.0, 150, 32, "hybrid"),           # Cargo delivery (~110 lbs)
        (75.0, 1.2, 120, 36, "hybrid"),           # Heavy cargo (~165 lbs)

        # Large (75-200 kg / 165-440 lbs)
        (100.0, 1.4, 100, 40, "hybrid"),          # Large cargo/passenger
        (150.0, 1.6, 80, 48, "turbine"),          # Air taxi class
        (200.0, 1.8, 60, 54, "turbine"),          # Heavy eVTOL

        # Military/Industrial Heavy (200+ kg)
        (500.0, 2.5, 0, 0, "turboprop"),          # MQ-1 Predator class
        (1000.0, 3.0, 0, 0, "turboprop"),         # Medium altitude UAV
        (2000.0, 4.0, 0, 0, "turbofan"),          # Large UAV
        (4760.0, 5.5, 0, 0, "turboprop"),         # MQ-9 Reaper (10,500 lbs)
        (20000.0, 8.0, 0, 0, "turbofan"),         # X-47B UCAV class (44,000 lbs)
    ]

    # Fixed-wing data
    # Format: (mass_kg, wingspan_m, cruise_speed_m_s, propulsion_type)
    FIXED_WING_DATA = [
        (1.0, 1.2, 15, "electric"),               # Small FPV
        (5.0, 2.5, 25, "electric"),               # Survey drone
        (15.0, 4.0, 35, "electric"),              # Mapping UAV
        (50.0, 6.0, 45, "hybrid"),                # Long endurance
        (150.0, 10.0, 60, "turboprop"),           # Tactical UAV
        (500.0, 14.0, 80, "turboprop"),           # MQ-1 Predator (wingspan 14.8m)
        (1000.0, 16.0, 100, "turboprop"),
        (2000.0, 20.0, 120, "turbofan"),
        (4760.0, 20.0, 130, "turboprop"),         # MQ-9 Reaper (wingspan 20m)
        (14000.0, 35.0, 180, "turbofan"),         # RQ-4 Global Hawk
        (20000.0, 19.0, 250, "turbofan"),         # X-47B UCAV
    ]

    # Propulsion system characteristics by type
    PROPULSION_PARAMS = {
        "electric": {
            "power_density_w_kg": 200,            # Motor power per kg motor
            "efficiency": 0.85,
            "energy_density_wh_kg": 200,          # LiPo battery
            "sfc": 0,                             # No fuel consumption
        },
        "hybrid": {
            "power_density_w_kg": 500,
            "efficiency": 0.35,
            "energy_density_wh_kg": 3000,         # Gasoline equivalent
            "sfc": 0.3,                           # kg fuel per kW-hr
        },
        "turboprop": {
            "power_density_w_kg": 2000,
            "efficiency": 0.30,
            "energy_density_wh_kg": 12000,        # Jet fuel
            "sfc": 0.25,
        },
        "turbofan": {
            "power_density_w_kg": 5000,
            "efficiency": 0.35,
            "energy_density_wh_kg": 12000,
            "sfc": 0.20,
        },
    }

    # Typical motor constants (for electric)
    MOTOR_KF_PER_KV = 3.16e-10      # Approximate thrust coefficient scaling
    MOTOR_KM_PER_KF = 0.0245        # Torque to thrust ratio

    def __init__(self, spec: DroneSpecification):
        self.spec = spec
        self.result = ValidationResult()

    def estimate_all(self) -> DroneSpecification:
        """Estimate all missing parameters."""
        # Order matters - some estimates depend on others
        self._estimate_dimensions()
        self._estimate_inertia()
        self._estimate_motors()
        self._estimate_propellers()
        self._estimate_battery()
        self._estimate_performance()
        self._estimate_aerodynamics()

        return self.spec

    def _estimate_dimensions(self):
        """Estimate physical dimensions from mass or arm length."""
        spec = self.spec

        # If arm length is missing, estimate from mass
        if spec.arm_length_m <= 0 and spec.mass_kg > 0:
            # Empirical: arm_length ~ 0.1 * mass^0.4 for multirotors
            spec.arm_length_m = 0.1 * (spec.mass_kg ** 0.4)
            self.result.add_auto_completed(
                "arm_length_m",
                f"{spec.arm_length_m:.3f}",
                "empirical scaling from mass"
            )

        # Estimate overall dimensions
        if spec.length_m <= 0 and spec.arm_length_m > 0:
            if spec.num_motors == 4:
                spec.length_m = spec.arm_length_m * 2.2
                spec.width_m = spec.arm_length_m * 2.2
            elif spec.num_motors == 6:
                spec.length_m = spec.arm_length_m * 2.5
                spec.width_m = spec.arm_length_m * 2.5
            spec.height_m = spec.arm_length_m * 0.4

            self.result.add_auto_completed(
                "dimensions",
                f"{spec.length_m:.2f}x{spec.width_m:.2f}x{spec.height_m:.2f}m",
                "estimated from arm length"
            )

    def _estimate_inertia(self):
        """Estimate moments of inertia from mass and dimensions."""
        spec = self.spec
        inertial = spec.inertial

        if inertial.mass_total_kg <= 0:
            inertial.mass_total_kg = spec.mass_kg

        # Estimate inertia if not provided
        if inertial.ixx <= 0 and spec.mass_kg > 0 and spec.arm_length_m > 0:
            # Simplified model: treat as point masses at motor positions
            # Plus central body contribution
            m = spec.mass_kg
            L = spec.arm_length_m

            # For quadcopter X config
            if spec.num_motors == 4:
                # Motor contribution (4 motors at arm tips)
                motor_mass = m * 0.15 / 4  # ~15% of mass in motors
                motor_inertia = 4 * motor_mass * (L ** 2)

                # Body contribution (approximate as box)
                body_mass = m * 0.5
                body_length = L * 0.5
                body_inertia_xx = body_mass * (body_length ** 2) / 12

                inertial.ixx = motor_inertia + body_inertia_xx
                inertial.iyy = inertial.ixx  # Symmetric
                inertial.izz = 2 * motor_inertia  # Larger for yaw

            else:
                # Generic estimate
                inertial.ixx = 0.5 * m * (L ** 2)
                inertial.iyy = inertial.ixx
                inertial.izz = m * (L ** 2)

            self.result.add_auto_completed(
                "inertia",
                f"Ixx={inertial.ixx:.4f}, Iyy={inertial.iyy:.4f}, Izz={inertial.izz:.4f}",
                "estimated from mass distribution"
            )

    def _estimate_motors(self):
        """Estimate motor parameters."""
        spec = self.spec

        # Estimate motor KV if not provided
        if len(spec.motors) == 0 and spec.num_motors > 0:
            # Estimate KV from mass (empirical)
            kv_estimate = 3000 * (spec.mass_kg ** -0.5)  # Heavier = lower KV
            kv_estimate = np.clip(kv_estimate, 200, 4000)

            # Create motor specs
            if spec.motor_config == MotorConfiguration.QUAD_X:
                positions = self._get_quad_x_positions(spec.arm_length_m)
                spin_dirs = [1, -1, 1, -1]
            else:
                # Generic positions in a circle
                n = spec.num_motors
                positions = []
                for i in range(n):
                    angle = 2 * np.pi * i / n + np.pi / 4
                    x = spec.arm_length_m * np.cos(angle)
                    y = spec.arm_length_m * np.sin(angle)
                    positions.append((x, y, 0))
                spin_dirs = [1 if i % 2 == 0 else -1 for i in range(n)]

            for i, (pos, spin) in enumerate(zip(positions, spin_dirs)):
                motor = MotorSpecification(
                    motor_id=i,
                    position_x=pos[0],
                    position_y=pos[1],
                    position_z=pos[2],
                    kv_rating=kv_estimate,
                    spin_direction=spin,
                    thrust_coefficient_kf=self.MOTOR_KF_PER_KV * kv_estimate,
                    torque_coefficient_km=self.MOTOR_KF_PER_KV * kv_estimate * self.MOTOR_KM_PER_KF,
                )
                spec.motors.append(motor)

            self.result.add_auto_completed(
                "motors",
                f"{spec.num_motors} motors at KV={kv_estimate:.0f}",
                "empirical estimation from mass"
            )

    def _estimate_propellers(self):
        """Estimate propeller parameters."""
        spec = self.spec

        if len(spec.propellers) == 0 and spec.num_motors > 0:
            # Estimate prop size from arm length (leave clearance)
            prop_diameter_m = spec.arm_length_m * 0.9
            prop_diameter_inch = prop_diameter_m / 0.0254

            # Round to common sizes
            common_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 28]
            prop_diameter_inch = min(common_sizes, key=lambda x: abs(x - prop_diameter_inch))

            for i in range(spec.num_motors):
                prop = PropellerSpecification(
                    propeller_id=i,
                    motor_id=i,
                    diameter_inch=prop_diameter_inch,
                    diameter_m=prop_diameter_inch * 0.0254,
                    pitch_inch=prop_diameter_inch * 0.5,  # Typical pitch ratio
                )
                spec.propellers.append(prop)

            self.result.add_auto_completed(
                "propellers",
                f'{prop_diameter_inch}" props',
                "estimated from arm length"
            )

    def _estimate_battery(self):
        """Estimate battery/fuel parameters based on weight class."""
        spec = self.spec
        batt = spec.battery
        weight_class = WeightClass.classify(spec.mass_kg)

        # Determine propulsion type based on weight class
        propulsion_type = self._determine_propulsion_type(spec.mass_kg)

        if propulsion_type == "electric":
            # Electric - estimate battery
            if batt.capacity_mah <= 0 and spec.mass_kg > 0:
                # Battery is typically 30-50% of total mass for electric
                battery_mass_g = spec.mass_kg * 1000 * 0.35

                if batt.cell_count <= 0:
                    # Estimate cell count from mass
                    if spec.mass_kg < 0.5:
                        batt.cell_count = 2
                    elif spec.mass_kg < 1.5:
                        batt.cell_count = 4
                    elif spec.mass_kg < 5:
                        batt.cell_count = 6
                    elif spec.mass_kg < 15:
                        batt.cell_count = 8
                    elif spec.mass_kg < 30:
                        batt.cell_count = 12
                    else:
                        batt.cell_count = 14  # High voltage systems

                batt.capacity_mah = battery_mass_g * 180 / (3.7 * batt.cell_count)
                batt.nominal_voltage = 3.7
                batt.pack_voltage_nominal = 3.7 * batt.cell_count
                batt.mass_g = battery_mass_g

                self.result.add_auto_completed(
                    "battery",
                    f"{batt.capacity_mah:.0f}mAh {batt.cell_count}S",
                    "estimated from mass (electric)"
                )

        elif propulsion_type in ["hybrid", "turboprop", "turbofan"]:
            # Fuel-based systems
            if spec.fuel_capacity_kg <= 0 and spec.mass_kg > 0:
                # Fuel is typically 20-40% of MTOW for longer endurance
                spec.fuel_capacity_kg = spec.mass_kg * 0.25

                # Set fuel type
                if propulsion_type == "hybrid":
                    spec.fuel_type = "avgas"
                else:
                    spec.fuel_type = "jet-a"

                self.result.add_auto_completed(
                    "fuel_capacity",
                    f"{spec.fuel_capacity_kg:.1f} kg ({spec.fuel_type})",
                    f"estimated from mass ({propulsion_type})"
                )

            # Update propulsion type in spec
            if spec.propulsion_type == PropulsionType.ELECTRIC_BRUSHLESS:
                propulsion_map = {
                    "hybrid": PropulsionType.HYBRID_ELECTRIC,
                    "turboprop": PropulsionType.TURBOPROP,
                    "turbofan": PropulsionType.TURBOFAN,
                }
                spec.propulsion_type = propulsion_map.get(propulsion_type, spec.propulsion_type)

                self.result.add_auto_completed(
                    "propulsion_type",
                    spec.propulsion_type.value,
                    f"auto-selected for {weight_class} class UAV"
                )

    def _determine_propulsion_type(self, mass_kg: float) -> str:
        """Determine likely propulsion type based on mass."""
        if mass_kg < 25:
            return "electric"
        elif mass_kg < 100:
            return "hybrid"  # Hybrid-electric for medium
        elif mass_kg < 500:
            return "turboprop"
        else:
            return "turbofan"

    def _estimate_performance(self):
        """Estimate performance envelope."""
        spec = self.spec
        perf = spec.performance

        # Max speed (empirical: heavier = slower for multirotors)
        if spec.max_speed_m_s <= 0:
            spec.max_speed_m_s = 30 * (spec.mass_kg ** -0.2)
            spec.max_speed_m_s = np.clip(spec.max_speed_m_s, 10, 50)
            perf.max_speed_m_s = spec.max_speed_m_s

            self.result.add_auto_completed(
                "max_speed",
                f"{spec.max_speed_m_s:.1f} m/s",
                "empirical estimation"
            )

        # Max climb rate
        if spec.max_climb_rate_m_s <= 0 and spec.max_thrust_to_weight > 0:
            # Excess thrust determines climb rate
            excess_thrust_ratio = spec.max_thrust_to_weight - 1.0
            spec.max_climb_rate_m_s = excess_thrust_ratio * 10  # Rough approximation
            spec.max_climb_rate_m_s = np.clip(spec.max_climb_rate_m_s, 2, 20)
            perf.max_climb_rate_m_s = spec.max_climb_rate_m_s

            self.result.add_auto_completed(
                "max_climb_rate",
                f"{spec.max_climb_rate_m_s:.1f} m/s",
                "estimated from thrust-to-weight"
            )

        # Thrust-to-weight if not provided
        if spec.max_thrust_to_weight <= 0:
            # Default assumption for typical multirotors
            spec.max_thrust_to_weight = 2.0

            self.result.add_auto_completed(
                "thrust_to_weight",
                f"{spec.max_thrust_to_weight:.1f}",
                "default assumption"
            )

        # Hover throttle
        if spec.hover_throttle <= 0:
            spec.hover_throttle = 1.0 / spec.max_thrust_to_weight

            self.result.add_auto_completed(
                "hover_throttle",
                f"{spec.hover_throttle:.2f}",
                "calculated from T/W ratio"
            )

        # Endurance estimate
        if perf.max_endurance_min <= 0 and spec.battery.capacity_mah > 0:
            # Rough estimate: 10-15 min per 1000mAh for typical quad
            hover_current_a = spec.mass_kg * 10  # ~10A per kg at hover
            usable_capacity = spec.battery.capacity_mah * 0.8  # 80% usable
            perf.max_endurance_min = (usable_capacity / 1000) / hover_current_a * 60

            self.result.add_auto_completed(
                "endurance",
                f"{perf.max_endurance_min:.1f} min",
                "estimated from battery capacity"
            )

    def _estimate_aerodynamics(self):
        """Estimate aerodynamic parameters for fixed-wing."""
        spec = self.spec

        # Only for fixed-wing
        if "fixed_wing" not in spec.airframe_type.value:
            return

        aero = spec.aerodynamics

        if len(aero.wings) > 0:
            wing = aero.wings[0]

            # Estimate lift curve slope if not provided
            if wing.cl_alpha <= 0:
                # Thin airfoil theory: ~0.11 per degree
                # Finite wing correction
                AR = wing.aspect_ratio if wing.aspect_ratio > 0 else 8.0
                wing.cl_alpha = 0.11 * AR / (AR + 2)

                self.result.add_auto_completed(
                    "cl_alpha",
                    f"{wing.cl_alpha:.3f} /deg",
                    "finite wing theory"
                )

            # Estimate drag
            if wing.cd_0 <= 0:
                wing.cd_0 = 0.02  # Typical value

            # Stall angle
            if wing.stall_angle_deg <= 0:
                wing.stall_angle_deg = 15.0

    def _get_quad_x_positions(self, arm_length: float) -> List[Tuple[float, float, float]]:
        """Get motor positions for quad X configuration."""
        L = arm_length
        angle = np.pi / 4  # 45 degrees
        return [
            (L * np.cos(angle), L * np.sin(angle), 0),        # Front-right
            (-L * np.cos(angle), L * np.sin(angle), 0),       # Back-right
            (-L * np.cos(angle), -L * np.sin(angle), 0),      # Back-left
            (L * np.cos(angle), -L * np.sin(angle), 0),       # Front-left
        ]


def auto_complete_specs(spec: DroneSpecification) -> ValidationResult:
    """
    Auto-complete missing parameters in a drone specification.

    This is the main entry point for parameter estimation.
    """
    estimator = ParameterEstimator(spec)
    estimator.estimate_all()
    return estimator.result


def estimate_missing_parameters(spec: DroneSpecification) -> Tuple[DroneSpecification, List[str]]:
    """
    Estimate missing parameters and return modified spec with list of estimates.

    Returns:
        Tuple of (modified_spec, list_of_estimated_fields)
    """
    result = auto_complete_specs(spec)
    return spec, result.auto_completed


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate(spec: DroneSpecification) -> bool:
    """Quick validation - returns True if spec is usable."""
    result = validate_specification(spec)
    return result.is_valid


def full_process(spec: DroneSpecification) -> Tuple[DroneSpecification, ValidationResult]:
    """
    Full processing: auto-complete then validate.

    This is the recommended way to process user specifications.
    """
    # First, estimate missing parameters
    auto_result = auto_complete_specs(spec)

    # Then validate
    val_result = validate_specification(spec)

    # Combine results
    val_result.auto_completed = auto_result.auto_completed

    return spec, val_result

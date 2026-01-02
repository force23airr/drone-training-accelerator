"""
Drone Specification Validator

Validates customer drone specifications for physical plausibility.
Catches common errors like impossible performance claims, inconsistent
dimensions, or missing required fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .drone_spec import DroneSpec, PropulsionType


class Severity(Enum):
    """Validation issue severity."""
    ERROR = "error"      # Cannot proceed, must fix
    WARNING = "warning"  # Unusual but allowed
    INFO = "info"        # Informational note


@dataclass
class ValidationError:
    """A validation error that must be fixed."""
    field: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        s = f"ERROR [{self.field}]: {self.message}"
        if self.suggestion:
            s += f" (Suggestion: {self.suggestion})"
        return s


@dataclass
class ValidationWarning:
    """A validation warning (unusual but allowed)."""
    field: str
    message: str
    typical_range: Optional[str] = None

    def __str__(self) -> str:
        s = f"WARNING [{self.field}]: {self.message}"
        if self.typical_range:
            s += f" (Typical: {self.typical_range})"
        return s


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("Validation PASSED")
        else:
            lines.append("Validation FAILED")

        if self.errors:
            lines.append(f"\n{len(self.errors)} Error(s):")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append(f"\n{len(self.warnings)} Warning(s):")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.notes:
            lines.append(f"\nNotes:")
            for n in self.notes:
                lines.append(f"  - {n}")

        return "\n".join(lines)


class SpecValidator:
    """
    Validate drone specifications for physical plausibility.

    Checks:
    - Required fields are provided
    - Values are within physically reasonable ranges
    - Performance claims are consistent with each other
    - Geometric relationships make sense
    """

    # Typical ranges for validation (can be adjusted)
    RANGES = {
        "wing_loading": (50, 800),      # kg/m² (glider to fighter)
        "thrust_to_weight": (0.15, 2.0), # T/W ratio
        "aspect_ratio": (2, 25),         # AR (delta to glider)
        "max_g": (3, 12),                # Structural g-limit
        "CL_max": (0.8, 2.5),            # Max lift coefficient
    }

    def validate(self, spec: DroneSpec) -> ValidationResult:
        """
        Validate a drone specification.

        Args:
            spec: The drone specification to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        notes = []

        # Check required fields
        required_errors = spec.validate_required()
        for err in required_errors:
            errors.append(ValidationError(
                field=err.split()[0],
                message=err,
                suggestion="Provide a positive value"
            ))

        # If basic requirements fail, return early
        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                notes=["Fix required field errors before proceeding"]
            )

        # Speed consistency checks
        self._validate_speeds(spec, errors, warnings)

        # Geometry checks
        self._validate_geometry(spec, errors, warnings, notes)

        # Performance checks
        self._validate_performance(spec, errors, warnings, notes)

        # Propulsion checks
        self._validate_propulsion(spec, errors, warnings)

        # Cross-checks
        self._validate_cross_consistency(spec, errors, warnings, notes)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            notes=notes
        )

    def _validate_speeds(
        self,
        spec: DroneSpec,
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ):
        """Validate speed parameters."""

        # Speed ordering: stall < cruise < max
        if spec.stall_speed_ms >= spec.cruise_speed_ms:
            errors.append(ValidationError(
                field="stall_speed_ms",
                message=f"Stall speed ({spec.stall_speed_ms} m/s) must be less than cruise speed ({spec.cruise_speed_ms} m/s)",
                suggestion="Stall speed is typically 40-60% of cruise speed"
            ))

        if spec.cruise_speed_ms >= spec.max_speed_ms:
            errors.append(ValidationError(
                field="cruise_speed_ms",
                message=f"Cruise speed ({spec.cruise_speed_ms} m/s) must be less than max speed ({spec.max_speed_ms} m/s)",
                suggestion="Cruise is typically 70-85% of max speed"
            ))

        # Check for supersonic claims
        speed_of_sound = 343  # m/s at sea level
        if spec.max_speed_ms > speed_of_sound:
            warnings.append(ValidationWarning(
                field="max_speed_ms",
                message=f"Max speed ({spec.max_speed_ms} m/s) is supersonic - ensure propulsion supports this",
                typical_range="Most UAVs: 50-300 m/s"
            ))

        # Never exceed speed check
        if spec.never_exceed_speed_ms and spec.never_exceed_speed_ms < spec.max_speed_ms:
            errors.append(ValidationError(
                field="never_exceed_speed_ms",
                message="Vne must be >= max speed",
                suggestion="Vne is typically 10-20% above max level speed"
            ))

    def _validate_geometry(
        self,
        spec: DroneSpec,
        errors: List[ValidationError],
        warnings: List[ValidationWarning],
        notes: List[str]
    ):
        """Validate geometric parameters."""

        # Aspect ratio check
        AR = spec.aspect_ratio
        if AR < self.RANGES["aspect_ratio"][0]:
            warnings.append(ValidationWarning(
                field="aspect_ratio",
                message=f"Aspect ratio {AR:.2f} is very low",
                typical_range="2-25 (delta wing to glider)"
            ))
        elif AR > self.RANGES["aspect_ratio"][1]:
            warnings.append(ValidationWarning(
                field="aspect_ratio",
                message=f"Aspect ratio {AR:.2f} is very high (glider-like)",
                typical_range="2-25 (delta wing to glider)"
            ))

        # Wing loading check
        WL = spec.wing_loading_kgm2
        if WL < self.RANGES["wing_loading"][0]:
            warnings.append(ValidationWarning(
                field="wing_loading",
                message=f"Wing loading {WL:.1f} kg/m² is very low (like a glider)",
                typical_range="100-500 kg/m² for combat UAVs"
            ))
        elif WL > self.RANGES["wing_loading"][1]:
            errors.append(ValidationError(
                field="wing_loading",
                message=f"Wing loading {WL:.1f} kg/m² seems impossibly high",
                suggestion="Check mass and wing area values"
            ))

        # Length vs wingspan sanity
        if spec.wingspan_m > 3 * spec.length_m:
            notes.append(f"High wingspan/length ratio ({spec.wingspan_m/spec.length_m:.1f}) suggests glider-like design")
        elif spec.length_m > 2 * spec.wingspan_m:
            notes.append(f"Low wingspan/length ratio ({spec.wingspan_m/spec.length_m:.1f}) suggests missile-like design")

    def _validate_performance(
        self,
        spec: DroneSpec,
        errors: List[ValidationError],
        warnings: List[ValidationWarning],
        notes: List[str]
    ):
        """Validate performance claims."""

        # G-force limits
        if spec.max_g_force < self.RANGES["max_g"][0]:
            warnings.append(ValidationWarning(
                field="max_g_force",
                message=f"Max g-force {spec.max_g_force}g is low for combat maneuvers",
                typical_range="6-9g for combat UAVs"
            ))
        elif spec.max_g_force > self.RANGES["max_g"][1]:
            warnings.append(ValidationWarning(
                field="max_g_force",
                message=f"Max g-force {spec.max_g_force}g is extremely high",
                typical_range="6-9g for combat UAVs (missiles go higher)"
            ))

        # Thrust-to-weight ratio
        TW = spec.thrust_to_weight
        if TW < self.RANGES["thrust_to_weight"][0]:
            warnings.append(ValidationWarning(
                field="thrust_to_weight",
                message=f"Thrust/weight ratio {TW:.2f} is low for combat",
                typical_range="0.4-1.0 for combat UAVs"
            ))
        elif TW > self.RANGES["thrust_to_weight"][1]:
            warnings.append(ValidationWarning(
                field="thrust_to_weight",
                message=f"Thrust/weight ratio {TW:.2f} is very high (rocket-like)",
                typical_range="0.4-1.0 for combat UAVs"
            ))

        # Altitude check
        if spec.max_altitude_m > 25000:
            warnings.append(ValidationWarning(
                field="max_altitude_m",
                message=f"Service ceiling {spec.max_altitude_m}m is in the stratosphere",
                typical_range="3000-18000m for most UAVs"
            ))

    def _validate_propulsion(
        self,
        spec: DroneSpec,
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ):
        """Validate propulsion parameters."""

        # Electric drones shouldn't claim afterburner
        if spec.propulsion_type == PropulsionType.ELECTRIC:
            if spec.afterburner:
                errors.append(ValidationError(
                    field="afterburner",
                    message="Electric propulsion cannot have afterburner",
                    suggestion="Set afterburner=False"
                ))

            # Electric range warning
            if spec.fuel_capacity_kg > spec.mass_kg * 0.5:
                warnings.append(ValidationWarning(
                    field="fuel_capacity_kg",
                    message="Battery mass is >50% of total mass",
                    typical_range="20-40% for electric UAVs"
                ))

        # Jet/turbofan specific
        if spec.propulsion_type in [PropulsionType.JET, PropulsionType.TURBOFAN]:
            # Minimum thrust for jets
            if spec.max_thrust_n < 5000:
                warnings.append(ValidationWarning(
                    field="max_thrust_n",
                    message=f"Thrust {spec.max_thrust_n}N is low for jet propulsion",
                    typical_range=">10kN for jet UAVs"
                ))

    def _validate_cross_consistency(
        self,
        spec: DroneSpec,
        errors: List[ValidationError],
        warnings: List[ValidationWarning],
        notes: List[str]
    ):
        """Cross-check parameters for consistency."""

        # Check if stall speed is consistent with wing loading
        # V_stall = sqrt(2 * W / (rho * S * CL_max))
        # Rearranging: CL_max = 2 * W / (rho * S * V_stall^2)
        rho = 1.225  # Sea level
        weight = spec.mass_kg * 9.81
        implied_CL_max = 2 * weight / (rho * spec.wing_area_m2 * spec.stall_speed_ms**2)

        if implied_CL_max < self.RANGES["CL_max"][0]:
            warnings.append(ValidationWarning(
                field="stall_speed_ms",
                message=f"Stall speed implies CL_max={implied_CL_max:.2f} which is low",
                typical_range="CL_max typically 1.2-1.6"
            ))
        elif implied_CL_max > self.RANGES["CL_max"][1]:
            warnings.append(ValidationWarning(
                field="stall_speed_ms",
                message=f"Stall speed implies CL_max={implied_CL_max:.2f} which is very high",
                typical_range="CL_max typically 1.2-1.6 (even with flaps)"
            ))

        # Check max speed vs thrust
        # Simplified: at max speed, thrust = drag
        # D = 0.5 * rho * V^2 * S * CD_0 + W^2 / (0.5 * rho * V^2 * S * pi * AR * e)
        CD_0_est = 0.02
        e_est = 0.8
        AR = spec.aspect_ratio
        V_max = spec.max_speed_ms
        S = spec.wing_area_m2

        drag_at_max = (0.5 * rho * V_max**2 * S * CD_0_est +
                       weight**2 / (0.5 * rho * V_max**2 * S * 3.14159 * AR * e_est))

        if spec.max_thrust_n < drag_at_max * 0.8:
            warnings.append(ValidationWarning(
                field="max_speed_ms",
                message=f"Max speed may not be achievable with given thrust (est. drag {drag_at_max:.0f}N)",
                typical_range="Thrust should exceed drag at max speed"
            ))

        # Notes about the design
        if spec.carrier_capable and not spec.catapult_launch:
            notes.append("Carrier capable but no catapult launch - requires STOL capability")

        if spec.stealth_features and spec.aspect_ratio > 10:
            notes.append("High aspect ratio may compromise stealth characteristics")


def validate_spec(spec: DroneSpec) -> ValidationResult:
    """Convenience function to validate a spec."""
    validator = SpecValidator()
    return validator.validate(spec)

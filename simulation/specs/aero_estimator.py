"""
Aerodynamics Estimator

Estimates aerodynamic coefficients and stability derivatives from
basic drone geometry. Uses empirical correlations and semi-empirical
methods suitable for preliminary design.

References:
- Raymer, "Aircraft Design: A Conceptual Approach"
- Roskam, "Airplane Design" series
- DATCOM methods
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from .drone_spec import DroneSpec, PropulsionType, ControlSurfaceType


@dataclass
class EstimatedAero:
    """Estimated aerodynamic parameters."""

    # Lift
    CL_0: float                    # Zero-alpha lift coefficient
    CL_alpha: float                # Lift curve slope [/rad]
    CL_max: float                  # Maximum lift coefficient
    CL_min: float                  # Minimum lift coefficient (negative alpha)

    # Drag
    CD_0: float                    # Zero-lift drag (parasite drag)
    CD_i_factor: float             # Induced drag factor (1/pi*AR*e)
    oswald_efficiency: float       # Oswald span efficiency factor

    # Stability derivatives
    Cm_0: float                    # Zero-lift pitching moment
    Cm_alpha: float                # Pitch stiffness [/rad]
    Cn_beta: float                 # Yaw stability (weathercock) [/rad]
    Cl_beta: float                 # Roll due to sideslip (dihedral) [/rad]

    # Damping derivatives
    Cm_q: float                    # Pitch damping
    Cn_r: float                    # Yaw damping
    Cl_p: float                    # Roll damping

    # Control derivatives (estimated)
    CL_de: float                   # Elevator lift effectiveness
    Cm_de: float                   # Elevator pitch effectiveness
    Cl_da: float                   # Aileron roll effectiveness
    Cn_dr: float                   # Rudder yaw effectiveness

    # Stall characteristics
    alpha_stall_rad: float         # Stall angle of attack
    alpha_zero_lift_rad: float     # Zero-lift angle of attack

    # Inertia estimates
    Ixx: float                     # Roll inertia [kg*m^2]
    Iyy: float                     # Pitch inertia [kg*m^2]
    Izz: float                     # Yaw inertia [kg*m^2]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "CL_0": self.CL_0,
            "CL_alpha": self.CL_alpha,
            "CL_max": self.CL_max,
            "CL_min": self.CL_min,
            "CD_0": self.CD_0,
            "oswald_efficiency": self.oswald_efficiency,
            "Cm_0": self.Cm_0,
            "Cm_alpha": self.Cm_alpha,
            "Cm_q": self.Cm_q,
            "Cn_beta": self.Cn_beta,
            "Cn_r": self.Cn_r,
            "Cl_beta": self.Cl_beta,
            "Cl_p": self.Cl_p,
            "CL_de": self.CL_de,
            "Cm_de": self.Cm_de,
            "Cl_da": self.Cl_da,
            "Cn_dr": self.Cn_dr,
            "alpha_stall": self.alpha_stall_rad,
            "Ixx": self.Ixx,
            "Iyy": self.Iyy,
            "Izz": self.Izz,
        }


class AeroEstimator:
    """
    Estimate aerodynamic coefficients from basic drone geometry.

    Uses empirical correlations suitable for:
    - Fixed-wing combat UAVs
    - Reconnaissance drones
    - Loitering munitions
    - VTOL transitioning aircraft (in fixed-wing mode)

    Note: These are estimates for simulation purposes. For actual
    aircraft design, use CFD, wind tunnel testing, or flight test data.
    """

    # Typical values for different aircraft classes
    TYPICAL_VALUES = {
        "combat_uav": {
            "CD_0": 0.020,
            "CL_max": 1.4,
            "oswald": 0.80,
        },
        "reconnaissance": {
            "CD_0": 0.015,
            "CL_max": 1.6,
            "oswald": 0.85,
        },
        "loitering_munition": {
            "CD_0": 0.025,
            "CL_max": 1.2,
            "oswald": 0.75,
        },
        "stealth": {
            "CD_0": 0.022,
            "CL_max": 1.3,
            "oswald": 0.70,
        },
    }

    def estimate_from_spec(self, spec: DroneSpec) -> EstimatedAero:
        """
        Estimate all aerodynamic parameters from a drone specification.

        Args:
            spec: Customer drone specification

        Returns:
            EstimatedAero with all estimated parameters
        """
        # Use provided values or estimate
        AR = spec.aspect_ratio

        # Lift curve slope (Helmbold equation for finite wings)
        CL_alpha = self._estimate_lift_curve_slope(spec)

        # Zero-lift coefficients
        CL_0 = spec.CL_0 if spec.CL_0 is not None else 0.1

        # Maximum lift
        CL_max = spec.CL_max if spec.CL_max is not None else self._estimate_cl_max(spec)
        CL_min = -0.8 * CL_max  # Approximate inverted limit

        # Drag
        CD_0 = spec.CD_0 if spec.CD_0 is not None else self._estimate_cd0(spec)
        oswald = spec.oswald_efficiency if spec.oswald_efficiency is not None else self._estimate_oswald(spec)
        CD_i_factor = 1.0 / (np.pi * AR * oswald) if AR > 0 else 0.05

        # Stability derivatives
        Cm_alpha = spec.Cm_alpha if spec.Cm_alpha is not None else self._estimate_cm_alpha(spec)
        Cn_beta = spec.Cn_beta if spec.Cn_beta is not None else self._estimate_cn_beta(spec)
        Cl_beta = spec.Cl_beta if spec.Cl_beta is not None else self._estimate_cl_beta(spec)

        # Damping derivatives
        Cm_q = self._estimate_cm_q(spec)
        Cn_r = self._estimate_cn_r(spec)
        Cl_p = self._estimate_cl_p(spec)

        # Control derivatives
        CL_de, Cm_de = self._estimate_elevator_effectiveness(spec)
        Cl_da = self._estimate_aileron_effectiveness(spec)
        Cn_dr = self._estimate_rudder_effectiveness(spec)

        # Stall characteristics
        alpha_stall = self._estimate_stall_angle(spec)
        alpha_zero_lift = -CL_0 / CL_alpha if CL_alpha > 0 else 0.0

        # Inertia
        Ixx, Iyy, Izz = self._estimate_inertia(spec)

        return EstimatedAero(
            CL_0=CL_0,
            CL_alpha=CL_alpha,
            CL_max=CL_max,
            CL_min=CL_min,
            CD_0=CD_0,
            CD_i_factor=CD_i_factor,
            oswald_efficiency=oswald,
            Cm_0=0.0,
            Cm_alpha=Cm_alpha,
            Cn_beta=Cn_beta,
            Cl_beta=Cl_beta,
            Cm_q=Cm_q,
            Cn_r=Cn_r,
            Cl_p=Cl_p,
            CL_de=CL_de,
            Cm_de=Cm_de,
            Cl_da=Cl_da,
            Cn_dr=Cn_dr,
            alpha_stall_rad=alpha_stall,
            alpha_zero_lift_rad=alpha_zero_lift,
            Ixx=Ixx,
            Iyy=Iyy,
            Izz=Izz,
        )

    def _estimate_lift_curve_slope(self, spec: DroneSpec) -> float:
        """
        Estimate lift curve slope using Helmbold equation.

        For finite wings: CL_alpha = 2*pi*AR / (2 + sqrt(4 + AR^2))
        This accounts for 3D effects and tip losses.
        """
        if spec.CL_alpha is not None:
            return spec.CL_alpha

        AR = spec.aspect_ratio
        if AR <= 0:
            return 5.0  # Default

        # Helmbold equation (good for moderate AR)
        CL_alpha = 2 * np.pi * AR / (2 + np.sqrt(4 + AR**2))

        # Adjust for sweep (if available from aspect ratio)
        # Higher AR typically means less sweep for combat UAVs
        if AR > 8:
            CL_alpha *= 0.95  # Slight reduction for high AR

        return float(CL_alpha)

    def _estimate_cl_max(self, spec: DroneSpec) -> float:
        """
        Estimate maximum lift coefficient.

        Based on wing loading and stall speed relationship:
        CL_max = 2 * W / (rho * S * V_stall^2)
        """
        if spec.stall_speed_ms > 0 and spec.wing_area_m2 > 0:
            rho = 1.225  # Sea level density
            weight = spec.mass_kg * 9.81
            V_stall = spec.stall_speed_ms

            CL_max = 2 * weight / (rho * spec.wing_area_m2 * V_stall**2)
            # Clamp to reasonable range
            CL_max = np.clip(CL_max, 0.8, 2.0)
            return float(CL_max)

        # Default based on aircraft type
        if spec.stealth_features:
            return 1.3
        elif spec.thrust_to_weight > 0.8:
            return 1.5  # High performance
        else:
            return 1.4  # Typical UAV

    def _estimate_cd0(self, spec: DroneSpec) -> float:
        """
        Estimate zero-lift drag coefficient.

        Uses wetted area correlation method (Raymer).
        CD_0 = Cf * S_wet / S_ref * FF
        """
        # Estimate wetted area ratio (S_wet / S_ref)
        # Typical values: 4-6 for conventional, 2-3 for flying wing
        if spec.control_surface_type in [ControlSurfaceType.ELEVON, ControlSurfaceType.DELTA]:
            wetted_ratio = 2.5  # Flying wing
        else:
            wetted_ratio = 5.0  # Conventional

        # Skin friction coefficient (turbulent, Re ~ 10^7)
        Cf = 0.0025

        # Form factor (accounts for thickness, interference)
        FF = 1.2

        CD_0_base = Cf * wetted_ratio * FF

        # Adjustments
        if spec.stealth_features:
            CD_0_base *= 1.15  # Stealth shaping adds drag

        if spec.has_speedbrake:
            CD_0_base *= 1.02  # Small penalty for actuators

        # Landing gear (assume retractable for combat UAVs)
        # No additional drag

        return float(np.clip(CD_0_base, 0.015, 0.040))

    def _estimate_oswald(self, spec: DroneSpec) -> float:
        """
        Estimate Oswald span efficiency factor.

        e = 1.78 * (1 - 0.045 * AR^0.68) - 0.64  (Raymer correlation)
        """
        AR = spec.aspect_ratio

        if AR <= 0:
            return 0.75

        # Raymer correlation
        e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64

        # Adjustments for configuration
        if spec.control_surface_type == ControlSurfaceType.ELEVON:
            e *= 0.95  # Flying wing typically lower
        if spec.stealth_features:
            e *= 0.92  # Stealth shaping reduces efficiency

        return float(np.clip(e, 0.60, 0.90))

    def _estimate_cm_alpha(self, spec: DroneSpec) -> float:
        """
        Estimate pitch stiffness (stability derivative).

        Negative = statically stable
        Typical range: -0.3 to -1.5 per radian
        """
        # More negative = more stable
        # Combat aircraft are often marginally stable or unstable

        if spec.control_surface_type in [ControlSurfaceType.CANARD]:
            # Canard configurations can be less stable
            return -0.3
        elif spec.control_surface_type in [ControlSurfaceType.ELEVON, ControlSurfaceType.DELTA]:
            # Flying wings need careful balance
            return -0.4
        else:
            # Conventional tail
            # Estimate based on tail volume coefficient
            return -0.6

    def _estimate_cn_beta(self, spec: DroneSpec) -> float:
        """
        Estimate yaw stability (weathercock stability).

        Positive = stable (nose into wind)
        Typical range: 0.05 to 0.20 per radian
        """
        # Larger vertical tail = more stability
        # Estimate from geometry

        # Flying wings have less directional stability
        if spec.control_surface_type in [ControlSurfaceType.ELEVON, ControlSurfaceType.DELTA]:
            return 0.05

        # Conventional
        return 0.10

    def _estimate_cl_beta(self, spec: DroneSpec) -> float:
        """
        Estimate roll due to sideslip (dihedral effect).

        Negative = stable (rolls away from sideslip)
        Typical range: -0.02 to -0.10 per radian
        """
        # Swept wings have inherent dihedral effect
        AR = spec.aspect_ratio

        if AR > 8:
            # High AR, likely unswept
            return -0.03
        elif AR > 5:
            # Moderate sweep
            return -0.05
        else:
            # Low AR, high sweep (delta)
            return -0.08

    def _estimate_cm_q(self, spec: DroneSpec) -> float:
        """Estimate pitch damping."""
        # Always negative (resists pitch rate)
        return -15.0

    def _estimate_cn_r(self, spec: DroneSpec) -> float:
        """Estimate yaw damping."""
        # Always negative (resists yaw rate)
        return -0.15

    def _estimate_cl_p(self, spec: DroneSpec) -> float:
        """Estimate roll damping."""
        # Always negative (resists roll rate)
        # Higher AR = more damping
        AR = spec.aspect_ratio
        return -0.4 - 0.02 * min(AR, 10)

    def _estimate_elevator_effectiveness(self, spec: DroneSpec) -> Tuple[float, float]:
        """Estimate elevator lift and moment effectiveness."""
        if spec.control_surface_type in [ControlSurfaceType.ELEVON, ControlSurfaceType.DELTA]:
            # Elevons on flying wing
            CL_de = 0.3
            Cm_de = -0.8
        else:
            # Conventional tail
            CL_de = 0.4
            Cm_de = -1.2
        return CL_de, Cm_de

    def _estimate_aileron_effectiveness(self, spec: DroneSpec) -> float:
        """Estimate aileron roll effectiveness."""
        AR = spec.aspect_ratio

        # Higher AR = more effective ailerons
        Cl_da = 0.10 + 0.01 * min(AR, 10)
        return float(Cl_da)

    def _estimate_rudder_effectiveness(self, spec: DroneSpec) -> float:
        """Estimate rudder yaw effectiveness."""
        if spec.control_surface_type in [ControlSurfaceType.ELEVON]:
            # No rudder on pure flying wing
            return -0.03  # Differential drag only
        else:
            return -0.08

    def _estimate_stall_angle(self, spec: DroneSpec) -> float:
        """
        Estimate stall angle of attack.

        Based on CL_max and lift curve slope.
        """
        CL_max = spec.CL_max if spec.CL_max else self._estimate_cl_max(spec)
        CL_alpha = spec.CL_alpha if spec.CL_alpha else self._estimate_lift_curve_slope(spec)
        CL_0 = spec.CL_0 if spec.CL_0 else 0.1

        if CL_alpha > 0:
            alpha_stall = (CL_max - CL_0) / CL_alpha
        else:
            alpha_stall = np.radians(15)

        # Clamp to reasonable range (10-20 degrees)
        return float(np.clip(alpha_stall, np.radians(10), np.radians(20)))

    def _estimate_inertia(self, spec: DroneSpec) -> Tuple[float, float, float]:
        """
        Estimate moments of inertia from geometry.

        Uses simplified cylinder + wing model.
        """
        # Use provided values if available
        if all([spec.Ixx_kgm2, spec.Iyy_kgm2, spec.Izz_kgm2]):
            return spec.Ixx_kgm2, spec.Iyy_kgm2, spec.Izz_kgm2

        m = spec.mass_kg
        b = spec.wingspan_m
        L = spec.length_m

        # Fuselage as cylinder
        r_fuse = spec.fuselage_diameter_m / 2 if spec.fuselage_diameter_m else L / 15
        m_fuse = 0.4 * m  # 40% in fuselage
        m_wing = 0.3 * m  # 30% in wings
        m_other = 0.3 * m  # 30% distributed

        # Fuselage inertia (cylinder about axes)
        I_fuse_xx = 0.5 * m_fuse * r_fuse**2  # Roll
        I_fuse_yy = (1/12) * m_fuse * L**2    # Pitch
        I_fuse_zz = (1/12) * m_fuse * L**2    # Yaw

        # Wing inertia (rod about center)
        I_wing_xx = (1/12) * m_wing * (b * 0.1)**2  # Wing chord contribution
        I_wing_yy = (1/12) * m_wing * (b * 0.2)**2  # Wing span * thickness
        I_wing_zz = (1/12) * m_wing * b**2          # Wing span

        # Total (add distributed mass at average radius)
        r_avg = (b + L) / 6
        Ixx = spec.Ixx_kgm2 if spec.Ixx_kgm2 else I_fuse_xx + I_wing_xx + m_other * r_avg**2 * 0.2
        Iyy = spec.Iyy_kgm2 if spec.Iyy_kgm2 else I_fuse_yy + I_wing_yy + m_other * r_avg**2 * 0.8
        Izz = spec.Izz_kgm2 if spec.Izz_kgm2 else I_fuse_zz + I_wing_zz + m_other * r_avg**2

        # Scale by mass (empirical)
        scale = m / 500  # Normalize to 500kg reference
        Ixx *= scale
        Iyy *= scale
        Izz *= scale

        return float(Ixx), float(Iyy), float(Izz)


def estimate_aero(spec: DroneSpec) -> EstimatedAero:
    """Convenience function to estimate aerodynamics."""
    estimator = AeroEstimator()
    return estimator.estimate_from_spec(spec)

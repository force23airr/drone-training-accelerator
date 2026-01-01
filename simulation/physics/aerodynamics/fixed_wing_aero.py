"""
Fixed-Wing Aircraft Aerodynamic Model.

Implements a full 6-DOF aerodynamic model with:
- Lift, drag, and side force computation
- Stability derivatives for all axes
- Control surface effectiveness
- Stall modeling with smooth transition
- Compressibility effects (Prandtl-Glauert correction)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from .aerodynamic_model import (
    AerodynamicModel,
    AeroState,
    AeroCoefficients,
    ControlSurfaceState,
    AircraftGeometry,
)
from .atmosphere_model import ISAAtmosphere


@dataclass
class StabilityDerivatives:
    """
    Aircraft stability and control derivatives.

    Naming convention: C{force}_{variable}
    - First letter: affected coefficient (L=lift, D=drag, Y=side, l=roll, m=pitch, n=yaw)
    - Second letter: derivative variable (alpha, beta, p, q, r, da, de, dr)

    All derivatives are per radian unless otherwise noted.
    """

    # Lift derivatives
    CL_0: float = 0.2           # Zero-alpha lift coefficient
    CL_alpha: float = 5.5       # Lift curve slope [/rad]
    CL_q: float = 5.0           # Pitch rate effect on lift
    CL_alpha_dot: float = 1.0   # Alpha rate effect on lift
    CL_de: float = 0.4          # Elevator effectiveness on lift

    # Drag derivatives
    CD_0: float = 0.02          # Zero-lift drag (parasite)
    CD_alpha: float = 0.0       # Linear drag variation (usually 0)
    CD_q: float = 0.0           # Pitch rate effect on drag
    CD_de: float = 0.0          # Elevator effect on drag
    CD_flaps: float = 0.02      # Flap drag increment (per unit deflection)
    CD_speedbrake: float = 0.05  # Speedbrake drag increment

    # Side force derivatives
    CY_beta: float = -0.5       # Sideslip effect on side force
    CY_p: float = 0.0           # Roll rate effect on side force
    CY_r: float = 0.3           # Yaw rate effect on side force
    CY_da: float = 0.0          # Aileron effect on side force
    CY_dr: float = 0.2          # Rudder effectiveness on side force

    # Rolling moment derivatives
    Cl_beta: float = -0.05      # Dihedral effect
    Cl_p: float = -0.5          # Roll damping
    Cl_r: float = 0.1           # Roll due to yaw rate
    Cl_da: float = 0.15         # Aileron effectiveness
    Cl_dr: float = 0.01         # Rudder effect on roll

    # Pitching moment derivatives
    Cm_0: float = 0.0           # Zero-alpha pitching moment
    Cm_alpha: float = -0.5      # Pitch stiffness (stability)
    Cm_q: float = -15.0         # Pitch damping
    Cm_alpha_dot: float = -5.0  # Alpha rate effect
    Cm_de: float = -1.2         # Elevator effectiveness

    # Yawing moment derivatives
    Cn_beta: float = 0.1        # Weathercock stability
    Cn_p: float = -0.03         # Adverse yaw from roll
    Cn_r: float = -0.15         # Yaw damping
    Cn_da: float = 0.01         # Aileron effect on yaw (adverse)
    Cn_dr: float = -0.08        # Rudder effectiveness

    # Flap effects
    CL_flaps: float = 0.4       # Flap lift increment
    Cm_flaps: float = -0.1      # Flap pitching moment

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'StabilityDerivatives':
        """Create from dictionary, using defaults for missing values."""
        defaults = cls()
        for key, value in data.items():
            if hasattr(defaults, key):
                setattr(defaults, key, value)
        return defaults


@dataclass
class StallModel:
    """
    Stall characteristics for smooth stall modeling.
    """

    alpha_stall: float = 0.26        # Stall angle [rad] (~15 deg)
    alpha_stall_neg: float = -0.17   # Negative stall angle [rad]
    CL_max: float = 1.4              # Maximum lift coefficient
    CL_min: float = -1.0             # Minimum lift coefficient
    stall_sharpness: float = 10.0    # Transition sharpness (higher = sharper)

    # Post-stall behavior
    post_stall_lift_slope: float = -2.0   # dCL/dalpha after stall [/rad]
    deep_stall_alpha: float = 0.52        # Deep stall angle (~30 deg)


class FixedWingAerodynamics(AerodynamicModel):
    """
    Complete 6-DOF fixed-wing aerodynamic model.

    Features:
    - Full stability derivative model
    - Smooth stall transition using sigmoid blending
    - Compressibility corrections (Prandtl-Glauert)
    - Ground effect modeling (optional)
    - Control surface saturation

    The model computes forces in stability axes then transforms to body axes.
    """

    def __init__(
        self,
        geometry: AircraftGeometry,
        derivatives: StabilityDerivatives = None,
        stall_model: StallModel = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize fixed-wing aerodynamic model.

        Args:
            geometry: Aircraft geometry parameters
            derivatives: Stability and control derivatives
            stall_model: Stall characteristics
            config: Additional configuration options
        """
        super().__init__(geometry, config)

        self.derivatives = derivatives or StabilityDerivatives()
        self.stall_model = stall_model or StallModel()
        self.atmosphere = ISAAtmosphere()

        # Configuration options
        self.enable_compressibility = config.get('enable_compressibility', True) if config else True
        self.enable_stall = config.get('enable_stall', True) if config else True
        self.oswald_efficiency = config.get('oswald_efficiency', 0.85) if config else 0.85

        # Cached calculations
        self._prev_alpha = 0.0
        self._alpha_dot = 0.0

    def compute_coefficients(
        self,
        aero_state: AeroState,
        control_surfaces: ControlSurfaceState,
    ) -> AeroCoefficients:
        """
        Compute all aerodynamic coefficients.

        Args:
            aero_state: Current aerodynamic state
            control_surfaces: Control surface deflections

        Returns:
            Complete set of aerodynamic coefficients
        """
        # Extract state
        alpha = aero_state.alpha
        beta = aero_state.beta
        p, q, r = aero_state.angular_velocity
        V = max(aero_state.airspeed, 1.0)  # Avoid division by zero

        # Non-dimensional rates
        b = self.geometry.wingspan
        c = self.geometry.mean_chord
        p_hat = p * b / (2 * V)      # Non-dimensional roll rate
        q_hat = q * c / (2 * V)      # Non-dimensional pitch rate
        r_hat = r * b / (2 * V)      # Non-dimensional yaw rate

        # Estimate alpha_dot (for unsteady effects)
        alpha_dot_hat = self._alpha_dot * c / (2 * V)

        # Get stability derivatives
        d = self.derivatives
        s = self.stall_model

        # --- Lift Coefficient ---
        CL_basic = (d.CL_0 +
                    d.CL_alpha * alpha +
                    d.CL_q * q_hat +
                    d.CL_alpha_dot * alpha_dot_hat +
                    d.CL_de * control_surfaces.elevator +
                    d.CL_flaps * control_surfaces.flaps)

        # Apply stall model
        if self.enable_stall:
            CL = self._apply_stall_model(CL_basic, alpha)
        else:
            CL = CL_basic

        # --- Drag Coefficient ---
        # Induced drag using lift coefficient
        AR = self.geometry.aspect_ratio
        if AR > 0:
            CD_induced = CL ** 2 / (np.pi * AR * self.oswald_efficiency)
        else:
            CD_induced = 0.0

        CD = (d.CD_0 +
              CD_induced +
              d.CD_q * abs(q_hat) +
              d.CD_de * abs(control_surfaces.elevator) +
              d.CD_flaps * control_surfaces.flaps +
              d.CD_speedbrake * control_surfaces.speedbrake)

        # Apply compressibility correction
        if self.enable_compressibility and aero_state.mach > 0.3:
            CD = self._apply_compressibility(CD, aero_state.mach)

        # --- Side Force Coefficient ---
        CY = (d.CY_beta * beta +
              d.CY_p * p_hat +
              d.CY_r * r_hat +
              d.CY_da * control_surfaces.aileron +
              d.CY_dr * control_surfaces.rudder)

        # --- Rolling Moment Coefficient ---
        Cl = (d.Cl_beta * beta +
              d.Cl_p * p_hat +
              d.Cl_r * r_hat +
              d.Cl_da * control_surfaces.aileron +
              d.Cl_dr * control_surfaces.rudder)

        # --- Pitching Moment Coefficient ---
        Cm = (d.Cm_0 +
              d.Cm_alpha * alpha +
              d.Cm_q * q_hat +
              d.Cm_alpha_dot * alpha_dot_hat +
              d.Cm_de * control_surfaces.elevator +
              d.Cm_flaps * control_surfaces.flaps)

        # --- Yawing Moment Coefficient ---
        Cn = (d.Cn_beta * beta +
              d.Cn_p * p_hat +
              d.Cn_r * r_hat +
              d.Cn_da * control_surfaces.aileron +
              d.Cn_dr * control_surfaces.rudder)

        # Convert from stability axes (CL, CD) to body axes (CX, CZ)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        CX = -CD * cos_a + CL * sin_a
        CZ = -CD * sin_a - CL * cos_a

        return AeroCoefficients(
            CX=CX, CY=CY, CZ=CZ,
            Cl=Cl, Cm=Cm, Cn=Cn,
            CL=CL, CD=CD
        )

    def _apply_stall_model(self, CL_linear: float, alpha: float) -> float:
        """
        Apply stall model using sigmoid blending for smooth transition.

        The model blends between linear and post-stall lift curves:
        - Below stall: Linear lift curve CL = CL_0 + CL_alpha * alpha
        - At stall: Smooth transition using sigmoid
        - Post-stall: Reduced lift with eventual deep stall

        Args:
            CL_linear: Linear-theory lift coefficient
            alpha: Angle of attack [rad]

        Returns:
            Lift coefficient with stall effects
        """
        s = self.stall_model

        # Positive stall
        if alpha > 0:
            # Blend factor: 0 below stall, 1 above
            blend = self._sigmoid(alpha - s.alpha_stall, s.stall_sharpness)

            # Post-stall lift (simplified flat plate model)
            alpha_beyond_stall = alpha - s.alpha_stall
            CL_post_stall = s.CL_max + s.post_stall_lift_slope * alpha_beyond_stall

            # Deep stall reduction
            if alpha > s.deep_stall_alpha:
                deep_stall_factor = np.cos(2 * (alpha - s.deep_stall_alpha))
                CL_post_stall *= max(0.3, deep_stall_factor)

            CL = (1 - blend) * CL_linear + blend * CL_post_stall
            return np.clip(CL, -2.0, s.CL_max)

        # Negative stall
        else:
            blend = self._sigmoid(-(alpha - s.alpha_stall_neg), s.stall_sharpness)

            alpha_beyond_stall = alpha - s.alpha_stall_neg
            CL_post_stall = s.CL_min + s.post_stall_lift_slope * (-alpha_beyond_stall)

            CL = (1 - blend) * CL_linear + blend * CL_post_stall
            return np.clip(CL, s.CL_min, 2.0)

    def _apply_compressibility(self, CD: float, mach: float) -> float:
        """
        Apply compressibility correction to drag coefficient.

        Uses Prandtl-Glauert correction for subsonic flow and
        wave drag model for transonic/supersonic.

        Args:
            CD: Incompressible drag coefficient
            mach: Mach number

        Returns:
            Compressibility-corrected drag coefficient
        """
        if mach < 0.7:
            # Prandtl-Glauert correction (subsonic)
            beta_pg = np.sqrt(1 - mach ** 2)
            return CD / beta_pg

        elif mach < 1.0:
            # Transonic regime - wave drag buildup
            # Simple empirical model
            wave_drag = 0.1 * (mach - 0.7) ** 2
            return CD + wave_drag

        elif mach < 1.2:
            # Transonic/low supersonic
            wave_drag = 0.1 + 0.2 * (mach - 1.0)
            return CD + wave_drag

        else:
            # Supersonic (simplified)
            wave_drag = 0.12 + 0.1 * (mach - 1.2)
            return CD + wave_drag

    @staticmethod
    def _sigmoid(x: float, k: float) -> float:
        """
        Sigmoid function for smooth transitions.

        Args:
            x: Input value
            k: Sharpness parameter

        Returns:
            Value between 0 and 1
        """
        return 1.0 / (1.0 + np.exp(-k * x))

    def update_alpha_rate(self, alpha: float, dt: float):
        """
        Update alpha rate estimation for unsteady aerodynamics.

        Call this once per timestep to track alpha_dot.

        Args:
            alpha: Current angle of attack [rad]
            dt: Timestep [s]
        """
        if dt > 0:
            self._alpha_dot = (alpha - self._prev_alpha) / dt
        self._prev_alpha = alpha

    def get_stall_margin(self, alpha: float) -> float:
        """
        Get margin to stall in radians.

        Positive value means below stall, negative means stalled.

        Args:
            alpha: Current angle of attack [rad]

        Returns:
            Stall margin [rad]
        """
        if alpha >= 0:
            return self.stall_model.alpha_stall - alpha
        else:
            return alpha - self.stall_model.alpha_stall_neg

    def get_trimmed_elevator(
        self,
        target_CL: float,
        alpha: float,
        q: float = 0.0
    ) -> float:
        """
        Compute elevator deflection required for trim at given lift.

        Args:
            target_CL: Desired lift coefficient
            alpha: Current angle of attack [rad]
            q: Pitch rate [rad/s] (usually 0 for trim)

        Returns:
            Required elevator deflection [rad]
        """
        d = self.derivatives

        # CL required from elevator
        CL_from_alpha = d.CL_0 + d.CL_alpha * alpha
        CL_needed = target_CL - CL_from_alpha

        if abs(d.CL_de) > 1e-6:
            de_for_lift = CL_needed / d.CL_de
        else:
            de_for_lift = 0.0

        # Also need to trim pitching moment to zero
        # Cm = Cm_0 + Cm_alpha * alpha + Cm_de * de = 0
        Cm_from_alpha = d.Cm_0 + d.Cm_alpha * alpha
        if abs(d.Cm_de) > 1e-6:
            de_for_trim = -Cm_from_alpha / d.Cm_de
        else:
            de_for_trim = 0.0

        # Average (simplified trim solution)
        return 0.5 * (de_for_lift + de_for_trim)


def create_jet_uav_aero_model(config: Dict[str, Any]) -> FixedWingAerodynamics:
    """
    Factory function to create aerodynamic model from platform config.

    Args:
        config: Platform configuration dictionary with physics_params

    Returns:
        Configured FixedWingAerodynamics instance
    """
    physics = config.get('physics_params', {})

    # Create geometry
    geometry = AircraftGeometry(
        wingspan=physics.get('wingspan', 10.0),
        wing_area=physics.get('wing_area', 20.0),
        mean_chord=physics.get('mean_chord', 2.0),
        aspect_ratio=physics.get('aspect_ratio', None),
    )

    # Create stability derivatives
    deriv_data = {
        'CL_0': physics.get('CL_0', 0.2),
        'CL_alpha': physics.get('CL_alpha', 5.5),
        'CD_0': physics.get('CD_0', 0.02),
        'Cm_alpha': physics.get('Cm_alpha', -0.5),
        'CL_de': physics.get('CL_de', 0.4),
        'Cm_de': physics.get('Cm_de', -1.2),
        'Cl_da': physics.get('Cl_da', 0.15),
        'Cn_dr': physics.get('Cn_dr', -0.08),
    }
    derivatives = StabilityDerivatives.from_dict(deriv_data)

    # Create stall model
    stall_model = StallModel(
        alpha_stall=physics.get('stall_alpha', np.radians(15)),
        CL_max=physics.get('CL_max', 1.4),
    )

    # Model configuration
    model_config = {
        'enable_compressibility': physics.get('compressibility_correction', True),
        'enable_stall': True,
        'oswald_efficiency': physics.get('oswald_efficiency', 0.85),
    }

    return FixedWingAerodynamics(
        geometry=geometry,
        derivatives=derivatives,
        stall_model=stall_model,
        config=model_config
    )

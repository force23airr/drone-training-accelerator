"""
Aerodynamic Model Base Classes.

Defines the abstract interface for aerodynamic force and moment computation,
along with common data structures for aerodynamic state and control surfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class AeroState:
    """
    Complete aerodynamic state of an aircraft.

    All velocities are in body frame unless otherwise noted.
    """

    # Position and orientation
    position: np.ndarray          # [x, y, z] in world frame [m]
    orientation: np.ndarray       # Quaternion [x, y, z, w] or Euler [roll, pitch, yaw]

    # Linear velocities (body frame)
    velocity_body: np.ndarray     # [u, v, w] body-frame velocity [m/s]

    # Angular velocities (body frame)
    angular_velocity: np.ndarray  # [p, q, r] body rates [rad/s]

    # Derived quantities (computed)
    altitude: float = 0.0         # Altitude [m]
    airspeed: float = 0.0         # True airspeed [m/s]
    alpha: float = 0.0            # Angle of attack [rad]
    beta: float = 0.0             # Sideslip angle [rad]
    mach: float = 0.0             # Mach number
    dynamic_pressure: float = 0.0  # Dynamic pressure [Pa]

    def __post_init__(self):
        """Compute derived quantities from primary state."""
        self.altitude = self.position[2] if self.position is not None else 0.0
        self._compute_aero_angles()

    def _compute_aero_angles(self):
        """Compute angle of attack, sideslip, and airspeed."""
        if self.velocity_body is None:
            return

        u, v, w = self.velocity_body

        # True airspeed
        self.airspeed = np.sqrt(u**2 + v**2 + w**2)

        if self.airspeed > 1e-6:
            # Angle of attack: alpha = atan2(w, u)
            self.alpha = np.arctan2(w, u)

            # Sideslip angle: beta = asin(v / V)
            # Clamp to avoid numerical issues
            self.beta = np.arcsin(np.clip(v / self.airspeed, -1.0, 1.0))
        else:
            self.alpha = 0.0
            self.beta = 0.0

    @classmethod
    def from_world_velocity(
        cls,
        position: np.ndarray,
        velocity_world: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray
    ) -> 'AeroState':
        """
        Create AeroState from world-frame velocity.

        Args:
            position: Position [x, y, z] in world frame
            velocity_world: Velocity [vx, vy, vz] in world frame
            orientation: Quaternion [x, y, z, w]
            angular_velocity: Body rates [p, q, r]

        Returns:
            AeroState with body-frame velocity computed
        """
        # Convert world velocity to body frame
        velocity_body = cls._world_to_body(velocity_world, orientation)

        return cls(
            position=position,
            orientation=orientation,
            velocity_body=velocity_body,
            angular_velocity=angular_velocity
        )

    @staticmethod
    def _world_to_body(v_world: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Transform vector from world frame to body frame using quaternion."""
        # Quaternion conjugate for inverse rotation
        qx, qy, qz, qw = quat
        q_conj = np.array([-qx, -qy, -qz, qw])

        # Rotate vector: v_body = q* * v * q
        return AeroState._rotate_vector(v_world, q_conj)

    @staticmethod
    def _rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q."""
        qx, qy, qz, qw = q

        # Quaternion rotation formula
        t = 2.0 * np.cross(np.array([qx, qy, qz]), v)
        return v + qw * t + np.cross(np.array([qx, qy, qz]), t)


@dataclass
class ControlSurfaceState:
    """
    Control surface deflections.

    Sign conventions (standard aircraft):
    - Aileron: Positive = right wing trailing edge down (roll right)
    - Elevator: Positive = trailing edge up (pitch down moment, nose up)
    - Rudder: Positive = trailing edge left (yaw right)
    - Flaps: 0 = retracted, 1 = fully extended
    - Speedbrake: 0 = retracted, 1 = fully extended
    """

    aileron: float = 0.0       # Aileron deflection [rad]
    elevator: float = 0.0      # Elevator deflection [rad]
    rudder: float = 0.0        # Rudder deflection [rad]
    flaps: float = 0.0         # Flap deflection [0-1 normalized]
    speedbrake: float = 0.0    # Speedbrake deflection [0-1 normalized]

    # Optional: Differential control surfaces for flying wings
    elevon_left: Optional[float] = None   # Combined elevator/aileron
    elevon_right: Optional[float] = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [aileron, elevator, rudder, flaps, speedbrake]."""
        return np.array([
            self.aileron,
            self.elevator,
            self.rudder,
            self.flaps,
            self.speedbrake
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControlSurfaceState':
        """Create from numpy array."""
        return cls(
            aileron=arr[0] if len(arr) > 0 else 0.0,
            elevator=arr[1] if len(arr) > 1 else 0.0,
            rudder=arr[2] if len(arr) > 2 else 0.0,
            flaps=arr[3] if len(arr) > 3 else 0.0,
            speedbrake=arr[4] if len(arr) > 4 else 0.0,
        )


@dataclass
class AeroCoefficients:
    """
    Dimensionless aerodynamic coefficients.

    Force coefficients (body axes):
    - CX: Axial force (positive forward)
    - CY: Side force (positive right)
    - CZ: Normal force (positive down)

    Moment coefficients (body axes):
    - Cl: Rolling moment (positive right wing down)
    - Cm: Pitching moment (positive nose up)
    - Cn: Yawing moment (positive nose right)

    Alternate representation (stability axes):
    - CL: Lift coefficient (perpendicular to velocity)
    - CD: Drag coefficient (parallel to velocity, opposing)
    """

    # Force coefficients (body frame)
    CX: float = 0.0    # Axial force coefficient
    CY: float = 0.0    # Side force coefficient
    CZ: float = 0.0    # Normal force coefficient

    # Moment coefficients (body frame)
    Cl: float = 0.0    # Rolling moment coefficient
    Cm: float = 0.0    # Pitching moment coefficient
    Cn: float = 0.0    # Yawing moment coefficient

    # Stability axis coefficients (alternative representation)
    CL: float = 0.0    # Lift coefficient
    CD: float = 0.0    # Drag coefficient

    def to_body_forces(
        self,
        dynamic_pressure: float,
        reference_area: float
    ) -> np.ndarray:
        """
        Convert to body-frame forces.

        Args:
            dynamic_pressure: q = 0.5 * rho * V^2 [Pa]
            reference_area: Wing reference area [m^2]

        Returns:
            Forces [X, Y, Z] in body frame [N]
        """
        qS = dynamic_pressure * reference_area
        return np.array([
            qS * self.CX,
            qS * self.CY,
            qS * self.CZ
        ])

    def to_body_moments(
        self,
        dynamic_pressure: float,
        reference_area: float,
        wingspan: float,
        chord: float
    ) -> np.ndarray:
        """
        Convert to body-frame moments.

        Args:
            dynamic_pressure: q = 0.5 * rho * V^2 [Pa]
            reference_area: Wing reference area [m^2]
            wingspan: Wing span [m]
            chord: Mean aerodynamic chord [m]

        Returns:
            Moments [L, M, N] in body frame [Nm]
        """
        qS = dynamic_pressure * reference_area
        return np.array([
            qS * wingspan * self.Cl,    # Roll
            qS * chord * self.Cm,       # Pitch
            qS * wingspan * self.Cn     # Yaw
        ])


@dataclass
class AircraftGeometry:
    """
    Aircraft geometric parameters for aerodynamic calculations.
    """

    wingspan: float              # Wing span [m]
    wing_area: float             # Reference wing area [m^2]
    mean_chord: float            # Mean aerodynamic chord [m]
    aspect_ratio: float = None   # Aspect ratio = b^2 / S

    # Center of gravity location (from nose, normalized by chord)
    cg_location: float = 0.25    # Typical: 25% MAC

    # Tail geometry (for stability calculations)
    horizontal_tail_area: float = 0.0
    vertical_tail_area: float = 0.0
    tail_arm: float = 0.0        # Distance from CG to tail AC

    def __post_init__(self):
        """Compute derived quantities."""
        if self.aspect_ratio is None and self.wingspan > 0 and self.wing_area > 0:
            self.aspect_ratio = self.wingspan ** 2 / self.wing_area


class AerodynamicModel(ABC):
    """
    Abstract base class for aerodynamic models.

    Subclasses must implement coefficient computation for specific aircraft types.
    """

    def __init__(self, geometry: AircraftGeometry, config: Dict[str, Any] = None):
        """
        Initialize aerodynamic model.

        Args:
            geometry: Aircraft geometric parameters
            config: Model-specific configuration
        """
        self.geometry = geometry
        self.config = config or {}

    @abstractmethod
    def compute_coefficients(
        self,
        aero_state: AeroState,
        control_surfaces: ControlSurfaceState,
    ) -> AeroCoefficients:
        """
        Compute aerodynamic coefficients.

        Args:
            aero_state: Current aerodynamic state
            control_surfaces: Control surface deflections

        Returns:
            Dimensionless aerodynamic coefficients
        """
        pass

    def compute_forces_moments(
        self,
        aero_state: AeroState,
        control_surfaces: ControlSurfaceState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic forces and moments.

        Args:
            aero_state: Current aerodynamic state
            control_surfaces: Control surface deflections

        Returns:
            Tuple of (forces [X, Y, Z], moments [L, M, N]) in body frame
        """
        # Get coefficients
        coeffs = self.compute_coefficients(aero_state, control_surfaces)

        # Convert to dimensional forces and moments
        forces = coeffs.to_body_forces(
            aero_state.dynamic_pressure,
            self.geometry.wing_area
        )

        moments = coeffs.to_body_moments(
            aero_state.dynamic_pressure,
            self.geometry.wing_area,
            self.geometry.wingspan,
            self.geometry.mean_chord
        )

        return forces, moments

    @staticmethod
    def stability_to_body_forces(
        CL: float,
        CD: float,
        CY: float,
        alpha: float,
        dynamic_pressure: float,
        reference_area: float
    ) -> np.ndarray:
        """
        Convert stability-axis coefficients to body-frame forces.

        Args:
            CL: Lift coefficient
            CD: Drag coefficient
            CY: Side force coefficient
            alpha: Angle of attack [rad]
            dynamic_pressure: Dynamic pressure [Pa]
            reference_area: Reference area [m^2]

        Returns:
            Body-frame forces [X, Y, Z] [N]
        """
        qS = dynamic_pressure * reference_area

        # Rotation from stability to body axes
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        # Lift and drag to X and Z
        # X = -D*cos(alpha) + L*sin(alpha)
        # Z = -D*sin(alpha) - L*cos(alpha)
        X = qS * (-CD * cos_a + CL * sin_a)
        Y = qS * CY
        Z = qS * (-CD * sin_a - CL * cos_a)

        return np.array([X, Y, Z])

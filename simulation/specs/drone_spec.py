"""
Drone Specification Schema

Customer-friendly dataclass for defining custom drone specifications.
Designed for startups, universities, and manufacturers to easily
input their drone's physical and performance characteristics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class PropulsionType(Enum):
    """Propulsion system types."""
    JET = "jet"
    TURBOPROP = "turboprop"
    TURBOFAN = "turbofan"
    ELECTRIC = "electric"
    PISTON = "piston"
    HYBRID = "hybrid"


class ControlSurfaceType(Enum):
    """Control surface configurations."""
    CONVENTIONAL = "conventional"  # Aileron, elevator, rudder
    ELEVON = "elevon"              # Flying wing (pitch + roll combined)
    V_TAIL = "v_tail"              # V-tail ruddervators
    DELTA = "delta"                # Delta wing
    CANARD = "canard"              # Canard configuration
    TAILERONS = "tailerons"        # All-moving horizontal tail


class DroneCategory(Enum):
    """Drone category/class."""
    FIXED_WING = "fixed_wing"
    QUADCOPTER = "quadcopter"
    HEXACOPTER = "hexacopter"
    OCTOCOPTER = "octocopter"
    VTOL = "vtol"
    COAXIAL = "coaxial"


@dataclass
class WeaponSpec:
    """Weapon/payload specification."""
    weapon_type: str = "gun"       # gun, missile_ir, missile_radar, bomb
    quantity: int = 1
    ammo: int = 500
    range_m: float = 300.0
    damage: float = 10.0
    cooldown_s: float = 0.1
    mass_kg: float = 50.0


@dataclass
class DroneSpec:
    """
    Customer drone specification.

    This is the primary interface for customers to define their drone.
    Only basic, easily-measurable parameters are required. Advanced
    aerodynamic coefficients are auto-calculated from geometry.

    Required fields are marked with (REQUIRED).
    Optional fields will be estimated if not provided.
    """

    # =========================================================================
    # IDENTITY
    # =========================================================================
    name: str                                    # (REQUIRED) "Acme X-100 Combat Drone"
    manufacturer: str = "Unknown"                # "Acme Aerospace Inc."
    version: str = "1.0.0"                       # Spec version
    category: DroneCategory = DroneCategory.FIXED_WING

    # =========================================================================
    # PHYSICAL DIMENSIONS (REQUIRED)
    # =========================================================================
    mass_kg: float = 0.0                         # (REQUIRED) Total takeoff mass
    wingspan_m: float = 0.0                      # (REQUIRED) Wingtip to wingtip
    length_m: float = 0.0                        # (REQUIRED) Nose to tail
    wing_area_m2: float = 0.0                    # (REQUIRED) Wing planform area

    # Optional dimensions
    height_m: Optional[float] = None             # Fuselage height
    fuselage_diameter_m: Optional[float] = None  # Fuselage diameter/width
    mean_chord_m: Optional[float] = None         # Mean aerodynamic chord

    # =========================================================================
    # PERFORMANCE ENVELOPE (REQUIRED)
    # =========================================================================
    max_speed_ms: float = 0.0                    # (REQUIRED) Maximum level speed
    cruise_speed_ms: float = 0.0                 # (REQUIRED) Efficient cruise
    stall_speed_ms: float = 0.0                  # (REQUIRED) Minimum flight speed
    max_g_force: float = 6.0                     # (REQUIRED) Structural limit (+g)
    max_altitude_m: float = 10000.0              # Service ceiling

    # Optional performance
    min_g_force: Optional[float] = None          # Negative g limit
    max_climb_rate_ms: Optional[float] = None    # Max climb rate
    max_descent_rate_ms: Optional[float] = None  # Max descent rate
    max_roll_rate_degs: Optional[float] = None   # Roll rate limit
    max_pitch_rate_degs: Optional[float] = None  # Pitch rate limit
    max_yaw_rate_degs: Optional[float] = None    # Yaw rate limit
    max_bank_angle_deg: Optional[float] = None   # Bank angle limit
    never_exceed_speed_ms: Optional[float] = None  # Vne

    # =========================================================================
    # PROPULSION (REQUIRED)
    # =========================================================================
    propulsion_type: PropulsionType = PropulsionType.JET
    max_thrust_n: float = 0.0                    # (REQUIRED) Maximum thrust
    fuel_capacity_kg: float = 100.0              # Fuel/battery capacity

    # Optional propulsion details
    num_engines: int = 1                         # Number of engines
    idle_thrust_fraction: Optional[float] = None # Thrust at idle (0-1)
    specific_fuel_consumption: Optional[float] = None  # kg/(N*s)
    thrust_vectoring: bool = False               # Has thrust vectoring
    afterburner: bool = False                    # Has afterburner
    afterburner_thrust_ratio: Optional[float] = None  # AB thrust multiplier

    # =========================================================================
    # CONTROL SURFACES
    # =========================================================================
    control_surface_type: ControlSurfaceType = ControlSurfaceType.CONVENTIONAL

    # Optional control surface details
    aileron_max_deflection_deg: Optional[float] = None
    elevator_max_deflection_deg: Optional[float] = None
    rudder_max_deflection_deg: Optional[float] = None
    flap_max_deflection_deg: Optional[float] = None
    has_speedbrake: bool = False
    has_leading_edge_slats: bool = False

    # =========================================================================
    # INERTIA (Optional - estimated from geometry)
    # =========================================================================
    Ixx_kgm2: Optional[float] = None             # Roll moment of inertia
    Iyy_kgm2: Optional[float] = None             # Pitch moment of inertia
    Izz_kgm2: Optional[float] = None             # Yaw moment of inertia

    # =========================================================================
    # AERODYNAMICS (Optional - estimated from geometry)
    # These can be provided by customers with wind tunnel data
    # =========================================================================
    CL_0: Optional[float] = None                 # Zero-alpha lift coefficient
    CL_alpha: Optional[float] = None             # Lift curve slope (/rad)
    CL_max: Optional[float] = None               # Maximum lift coefficient
    CD_0: Optional[float] = None                 # Zero-lift drag coefficient
    oswald_efficiency: Optional[float] = None    # Oswald span efficiency

    # Stability derivatives (advanced users only)
    Cm_alpha: Optional[float] = None             # Pitch stiffness
    Cn_beta: Optional[float] = None              # Yaw stability (weathercock)
    Cl_beta: Optional[float] = None              # Roll due to sideslip (dihedral)

    # =========================================================================
    # PAYLOAD & WEAPONS
    # =========================================================================
    payload_capacity_kg: float = 0.0             # Max payload mass
    hardpoints: int = 0                          # Number of weapon hardpoints
    internal_bay: bool = False                   # Has internal weapons bay
    weapons: List[WeaponSpec] = field(default_factory=list)

    # =========================================================================
    # SENSORS & AVIONICS (for future use)
    # =========================================================================
    radar_range_km: Optional[float] = None
    irst_equipped: bool = False                  # Infrared search & track
    rwr_equipped: bool = False                   # Radar warning receiver
    datalink_range_km: Optional[float] = None

    # =========================================================================
    # SPECIAL CAPABILITIES
    # =========================================================================
    carrier_capable: bool = False
    catapult_launch: bool = False
    arrested_landing: bool = False
    autonomous_capable: bool = True
    swarm_capable: bool = False
    stealth_features: bool = False
    all_weather: bool = True

    # =========================================================================
    # METADATA
    # =========================================================================
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def aspect_ratio(self) -> float:
        """Wing aspect ratio."""
        if self.wing_area_m2 > 0:
            return self.wingspan_m ** 2 / self.wing_area_m2
        return 0.0

    @property
    def wing_loading_kgm2(self) -> float:
        """Wing loading in kg/m^2."""
        if self.wing_area_m2 > 0:
            return self.mass_kg / self.wing_area_m2
        return 0.0

    @property
    def thrust_to_weight(self) -> float:
        """Thrust-to-weight ratio."""
        weight_n = self.mass_kg * 9.81
        if weight_n > 0:
            return self.max_thrust_n / weight_n
        return 0.0

    @property
    def power_loading_wkg(self) -> float:
        """Power loading (for electric drones)."""
        if self.mass_kg > 0 and self.max_thrust_n > 0:
            # Approximate power from thrust * speed
            return (self.max_thrust_n * self.cruise_speed_ms) / self.mass_kg
        return 0.0

    def validate_required(self) -> List[str]:
        """Check that all required fields are provided."""
        errors = []

        if not self.name:
            errors.append("name is required")
        if self.mass_kg <= 0:
            errors.append("mass_kg must be positive")
        if self.wingspan_m <= 0:
            errors.append("wingspan_m must be positive")
        if self.length_m <= 0:
            errors.append("length_m must be positive")
        if self.wing_area_m2 <= 0:
            errors.append("wing_area_m2 must be positive")
        if self.max_speed_ms <= 0:
            errors.append("max_speed_ms must be positive")
        if self.cruise_speed_ms <= 0:
            errors.append("cruise_speed_ms must be positive")
        if self.stall_speed_ms <= 0:
            errors.append("stall_speed_ms must be positive")
        if self.max_thrust_n <= 0:
            errors.append("max_thrust_n must be positive")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "identity": {
                "name": self.name,
                "manufacturer": self.manufacturer,
                "version": self.version,
                "category": self.category.value,
            },
            "physical": {
                "mass_kg": self.mass_kg,
                "wingspan_m": self.wingspan_m,
                "length_m": self.length_m,
                "wing_area_m2": self.wing_area_m2,
                "height_m": self.height_m,
                "fuselage_diameter_m": self.fuselage_diameter_m,
            },
            "performance": {
                "max_speed_ms": self.max_speed_ms,
                "cruise_speed_ms": self.cruise_speed_ms,
                "stall_speed_ms": self.stall_speed_ms,
                "max_g_force": self.max_g_force,
                "max_altitude_m": self.max_altitude_m,
            },
            "propulsion": {
                "type": self.propulsion_type.value,
                "max_thrust_n": self.max_thrust_n,
                "fuel_capacity_kg": self.fuel_capacity_kg,
                "num_engines": self.num_engines,
            },
            "control": {
                "surface_type": self.control_surface_type.value,
            },
            "computed": {
                "aspect_ratio": self.aspect_ratio,
                "wing_loading_kgm2": self.wing_loading_kgm2,
                "thrust_to_weight": self.thrust_to_weight,
            },
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        return f"""
Drone Specification: {self.name}
{'=' * 50}
Manufacturer: {self.manufacturer} (v{self.version})
Category: {self.category.value}

Physical:
  Mass: {self.mass_kg:.1f} kg
  Wingspan: {self.wingspan_m:.2f} m
  Length: {self.length_m:.2f} m
  Wing Area: {self.wing_area_m2:.2f} m²
  Aspect Ratio: {self.aspect_ratio:.2f}
  Wing Loading: {self.wing_loading_kgm2:.1f} kg/m²

Performance:
  Max Speed: {self.max_speed_ms:.1f} m/s ({self.max_speed_ms * 3.6:.0f} km/h)
  Cruise Speed: {self.cruise_speed_ms:.1f} m/s ({self.cruise_speed_ms * 3.6:.0f} km/h)
  Stall Speed: {self.stall_speed_ms:.1f} m/s ({self.stall_speed_ms * 3.6:.0f} km/h)
  Max G-Force: +{self.max_g_force:.1f}g
  Service Ceiling: {self.max_altitude_m:.0f} m

Propulsion:
  Type: {self.propulsion_type.value}
  Max Thrust: {self.max_thrust_n:.0f} N
  T/W Ratio: {self.thrust_to_weight:.2f}
  Fuel Capacity: {self.fuel_capacity_kg:.1f} kg

Payload:
  Capacity: {self.payload_capacity_kg:.1f} kg
  Hardpoints: {self.hardpoints}
"""

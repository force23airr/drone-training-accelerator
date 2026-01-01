"""
Comprehensive Drone Specification Schema

Captures EVERY possible parameter for ANY drone type:
- Rotorcraft (quadcopter, hexacopter, octocopter, coaxial)
- Fixed-wing (conventional, flying wing, canard)
- VTOL (tiltrotor, tailsitter, lift+cruise)
- Jet-powered UAVs
- Hybrid configurations
- Custom experimental builds

This is the single source of truth for drone characteristics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from datetime import datetime


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AirframeType(Enum):
    """Primary airframe classification."""
    # Rotorcraft
    QUADCOPTER = "quadcopter"
    QUADCOPTER_X = "quadcopter_x"      # X configuration
    QUADCOPTER_PLUS = "quadcopter_plus" # + configuration
    QUADCOPTER_H = "quadcopter_h"       # H-frame
    HEXACOPTER = "hexacopter"
    HEXACOPTER_X = "hexacopter_x"
    HEXACOPTER_PLUS = "hexacopter_plus"
    HEXACOPTER_Y = "hexacopter_y"       # Y6 coaxial
    OCTOCOPTER = "octocopter"
    OCTOCOPTER_X = "octocopter_x"
    OCTOCOPTER_PLUS = "octocopter_plus"
    OCTOCOPTER_COAX = "octocopter_coax"  # X8 coaxial
    TRICOPTER = "tricopter"
    BICOPTER = "bicopter"
    MONOCOPTER = "monocopter"
    COAXIAL = "coaxial"                  # Single axis, counter-rotating

    # Fixed-wing
    FIXED_WING_CONVENTIONAL = "fixed_wing_conventional"  # Tail with elevator/rudder
    FIXED_WING_FLYING_WING = "fixed_wing_flying_wing"    # No tail, elevons
    FIXED_WING_CANARD = "fixed_wing_canard"              # Canard configuration
    FIXED_WING_TANDEM = "fixed_wing_tandem"              # Tandem wing
    FIXED_WING_BIPLANE = "fixed_wing_biplane"
    FIXED_WING_DELTA = "fixed_wing_delta"                # Delta wing

    # VTOL Hybrids
    VTOL_TILTROTOR = "vtol_tiltrotor"          # Motors tilt (V-22 style)
    VTOL_TILTWING = "vtol_tiltwing"            # Entire wing tilts
    VTOL_TAILSITTER = "vtol_tailsitter"        # Takes off on tail
    VTOL_LIFT_CRUISE = "vtol_lift_cruise"      # Separate lift and cruise motors
    VTOL_QUADPLANE = "vtol_quadplane"          # Quad + fixed wing

    # Special
    HELICOPTER_SINGLE = "helicopter_single"     # Single main + tail rotor
    HELICOPTER_COAXIAL = "helicopter_coaxial"   # Coaxial rotors
    HELICOPTER_TANDEM = "helicopter_tandem"     # CH-47 style
    ORNITHOPTER = "ornithopter"                 # Flapping wing
    PARAMOTOR = "paramotor"                     # Paraglider + motor
    AIRSHIP = "airship"                         # Lighter than air

    # Custom
    CUSTOM = "custom"


class PropulsionType(Enum):
    """Propulsion system type."""
    ELECTRIC_BRUSHLESS = "electric_brushless"
    ELECTRIC_BRUSHED = "electric_brushed"
    INTERNAL_COMBUSTION_PISTON = "ic_piston"
    INTERNAL_COMBUSTION_ROTARY = "ic_rotary"
    TURBOPROP = "turboprop"
    TURBOFAN = "turbofan"
    TURBOJET = "turbojet"
    RAMJET = "ramjet"
    ROCKET_SOLID = "rocket_solid"
    ROCKET_LIQUID = "rocket_liquid"
    HYBRID_ELECTRIC = "hybrid_electric"
    HYDROGEN_FUEL_CELL = "hydrogen_fuel_cell"
    SOLAR_ELECTRIC = "solar_electric"
    NONE = "none"  # Glider


class MotorConfiguration(Enum):
    """Motor layout configuration."""
    SINGLE = "single"
    DUAL = "dual"
    TRI = "tri"
    QUAD_X = "quad_x"
    QUAD_PLUS = "quad_plus"
    QUAD_H = "quad_h"
    HEX_X = "hex_x"
    HEX_PLUS = "hex_plus"
    HEX_Y = "hex_y"
    OCTO_X = "octo_x"
    OCTO_PLUS = "octo_plus"
    OCTO_COAX = "octo_coax"
    CUSTOM = "custom"


class FlightControllerType(Enum):
    """Flight controller firmware/hardware."""
    PX4 = "px4"
    ARDUPILOT = "ardupilot"
    BETAFLIGHT = "betaflight"
    INAV = "inav"
    CLEANFLIGHT = "cleanflight"
    KISS = "kiss"
    DJI = "dji"
    AUTERION = "auterion"
    CUSTOM = "custom"
    NONE = "none"  # Direct motor control


class SensorType(Enum):
    """Sensor types available."""
    IMU = "imu"
    GPS = "gps"
    RTK_GPS = "rtk_gps"
    BAROMETER = "barometer"
    MAGNETOMETER = "magnetometer"
    OPTICAL_FLOW = "optical_flow"
    LIDAR_1D = "lidar_1d"           # Rangefinder
    LIDAR_2D = "lidar_2d"           # Scanning LIDAR
    LIDAR_3D = "lidar_3d"           # Full 3D pointcloud
    CAMERA_RGB = "camera_rgb"
    CAMERA_STEREO = "camera_stereo"
    CAMERA_DEPTH = "camera_depth"
    CAMERA_THERMAL = "camera_thermal"
    CAMERA_MULTISPECTRAL = "camera_multispectral"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"
    AIRSPEED = "airspeed"           # Pitot tube
    AOA_SENSOR = "aoa_sensor"       # Angle of attack
    AOS_SENSOR = "aos_sensor"       # Angle of sideslip
    ENGINE_RPM = "engine_rpm"
    FUEL_FLOW = "fuel_flow"
    EGT = "egt"                     # Exhaust gas temperature
    CURRENT_SENSOR = "current_sensor"
    VOLTAGE_SENSOR = "voltage_sensor"


# =============================================================================
# SUB-SPECIFICATIONS
# =============================================================================

@dataclass
class InertialProperties:
    """Mass and inertia properties."""
    # Mass
    mass_empty_kg: float = 0.0                  # Dry mass without battery/fuel
    mass_battery_kg: float = 0.0                # Battery mass
    mass_fuel_kg: float = 0.0                   # Fuel mass (if applicable)
    mass_payload_kg: float = 0.0                # Current payload mass
    mass_total_kg: float = 0.0                  # Total takeoff mass
    max_takeoff_mass_kg: float = 0.0            # MTOW

    # Center of gravity (relative to geometric center, meters)
    cg_x: float = 0.0                           # Forward positive
    cg_y: float = 0.0                           # Right positive
    cg_z: float = 0.0                           # Down positive

    # Moments of inertia (kg*m^2)
    ixx: float = 0.0                            # Roll inertia
    iyy: float = 0.0                            # Pitch inertia
    izz: float = 0.0                            # Yaw inertia
    ixy: float = 0.0                            # Product of inertia
    ixz: float = 0.0                            # Product of inertia
    iyz: float = 0.0                            # Product of inertia

    # CG shift with fuel burn (for long endurance)
    cg_shift_per_kg_fuel: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MotorSpecification:
    """Individual motor specification."""
    motor_id: int = 0

    # Position relative to CG (meters)
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Orientation (for tilting motors)
    tilt_angle_deg: float = 0.0                 # Forward tilt (for forward flight)
    cant_angle_deg: float = 0.0                 # Sideways cant

    # Motor properties
    kv_rating: float = 0.0                      # RPM per volt
    max_rpm: float = 0.0
    max_current_a: float = 0.0
    max_power_w: float = 0.0
    resistance_ohm: float = 0.0
    efficiency: float = 0.85                     # Motor efficiency

    # Spin direction
    spin_direction: int = 1                      # 1 = CW, -1 = CCW

    # Thrust characteristics
    thrust_coefficient_kf: float = 0.0          # Thrust = kf * rpm^2
    torque_coefficient_km: float = 0.0          # Torque = km * rpm^2

    # Response dynamics
    time_constant_s: float = 0.05               # Motor response time

    # Tilt capability (for tiltrotors)
    can_tilt: bool = False
    min_tilt_deg: float = 0.0
    max_tilt_deg: float = 90.0
    tilt_rate_deg_s: float = 45.0


@dataclass
class PropellerSpecification:
    """Propeller specification."""
    propeller_id: int = 0
    motor_id: int = 0                            # Associated motor

    diameter_inch: float = 0.0
    diameter_m: float = 0.0
    pitch_inch: float = 0.0
    num_blades: int = 2

    # Aerodynamic coefficients
    ct: float = 0.0                              # Thrust coefficient
    cp: float = 0.0                              # Power coefficient

    # Performance curves (optional, for detailed modeling)
    thrust_curve: Optional[List[Tuple[float, float]]] = None   # (RPM, thrust_N)
    power_curve: Optional[List[Tuple[float, float]]] = None    # (RPM, power_W)

    # Material properties
    material: str = "plastic"                    # plastic, carbon, wood
    mass_g: float = 0.0


@dataclass
class BatterySpecification:
    """Battery/power system specification."""
    # Type
    chemistry: str = "lipo"                      # lipo, liion, lihv, nimh, fuel

    # Capacity
    capacity_mah: float = 0.0
    capacity_wh: float = 0.0

    # Voltage
    cell_count: int = 0                          # S count
    nominal_voltage: float = 0.0                 # Per cell
    full_voltage: float = 0.0                    # Per cell
    empty_voltage: float = 0.0                   # Per cell
    pack_voltage_nominal: float = 0.0            # Total pack

    # Current
    max_continuous_discharge_c: float = 0.0     # C rating
    max_burst_discharge_c: float = 0.0
    max_continuous_current_a: float = 0.0

    # Physical
    mass_g: float = 0.0

    # Internal resistance
    internal_resistance_mohm: float = 0.0

    # Position in airframe
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0


@dataclass
class JetEngineSpecification:
    """Jet/turbine engine specification."""
    engine_id: int = 0
    engine_type: PropulsionType = PropulsionType.TURBOFAN

    # Position
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Orientation (thrust vector)
    thrust_vector_x: float = 1.0                 # Forward
    thrust_vector_y: float = 0.0
    thrust_vector_z: float = 0.0

    # Performance
    max_thrust_sea_level_n: float = 0.0
    max_thrust_afterburner_n: float = 0.0       # If equipped
    military_thrust_n: float = 0.0              # Max without afterburner
    idle_thrust_n: float = 0.0

    # Fuel consumption
    sfc_kg_per_n_hr: float = 0.0                # Specific fuel consumption
    sfc_afterburner: float = 0.0

    # Response
    spool_time_s: float = 2.0                   # Time to full throttle

    # Altitude/speed effects
    thrust_lapse_rate: float = 0.7              # Exponent for density ratio
    mach_effect_coefficient: float = 0.2        # Thrust increase with Mach

    # Bypass ratio (turbofan)
    bypass_ratio: float = 0.0                    # 0 for turbojet

    # Limits
    max_rpm: float = 0.0
    max_egt_c: float = 0.0                      # Exhaust gas temp limit


@dataclass
class WingSpecification:
    """Wing specification for fixed-wing aircraft."""
    wing_id: int = 0
    wing_type: str = "main"                      # main, canard, horizontal_stab, vertical_stab

    # Geometry
    span_m: float = 0.0
    chord_root_m: float = 0.0
    chord_tip_m: float = 0.0
    chord_mean_m: float = 0.0                    # MAC
    area_m2: float = 0.0
    aspect_ratio: float = 0.0
    taper_ratio: float = 1.0

    # Angles
    sweep_deg: float = 0.0                       # Leading edge sweep
    dihedral_deg: float = 0.0
    incidence_deg: float = 0.0                   # Wing incidence angle
    twist_deg: float = 0.0                       # Washout (tip twist)

    # Position relative to CG
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Airfoil
    airfoil_root: str = "NACA0012"
    airfoil_tip: str = "NACA0012"

    # Aerodynamic coefficients
    cl_0: float = 0.0                            # Zero-alpha lift
    cl_alpha: float = 0.1                        # Lift curve slope (per deg)
    cl_max: float = 1.5                          # Max lift coefficient
    cd_0: float = 0.02                           # Zero-lift drag
    cd_i_factor: float = 1.0                     # Induced drag factor
    cm_0: float = 0.0                            # Zero-alpha moment
    cm_alpha: float = 0.0                        # Moment curve slope

    # Stall characteristics
    stall_angle_deg: float = 15.0
    stall_type: str = "gradual"                  # gradual, sharp, deep

    # Efficiency
    oswald_efficiency: float = 0.8


@dataclass
class ControlSurfaceSpecification:
    """Control surface specification."""
    surface_id: int = 0
    surface_type: str = "aileron"                # aileron, elevator, rudder, elevon, ruddervator, flap, spoiler, airbrake

    wing_id: int = 0                             # Associated wing

    # Geometry
    span_m: float = 0.0
    chord_m: float = 0.0                         # Chord of control surface
    area_m2: float = 0.0
    hinge_position: float = 0.7                  # % chord from leading edge

    # Deflection limits
    max_deflection_up_deg: float = 25.0
    max_deflection_down_deg: float = 25.0

    # Effectiveness
    effectiveness: float = 1.0                   # Multiplier for control power

    # Control derivatives (per degree deflection)
    cl_delta: float = 0.0                        # Lift change
    cd_delta: float = 0.0                        # Drag change
    cm_delta: float = 0.0                        # Pitch moment (elevator)
    cn_delta: float = 0.0                        # Yaw moment (rudder)
    cy_delta: float = 0.0                        # Roll moment (aileron)

    # Servo
    servo_speed_deg_s: float = 200.0
    servo_torque_nm: float = 0.5


@dataclass
class AerodynamicSurfaces:
    """Complete aerodynamic surface configuration."""
    wings: List[WingSpecification] = field(default_factory=list)
    control_surfaces: List[ControlSurfaceSpecification] = field(default_factory=list)

    # Global aero properties
    reference_area_m2: float = 0.0               # Wing reference area
    reference_chord_m: float = 0.0               # MAC
    reference_span_m: float = 0.0

    # Stability derivatives (body axes)
    # Longitudinal
    cd_0: float = 0.02                           # Parasite drag
    cl_alpha: float = 0.1                        # Lift curve slope
    cm_alpha: float = -0.01                      # Pitch stability (negative = stable)
    cm_q: float = -10.0                          # Pitch damping

    # Lateral-directional
    cy_beta: float = -0.3                        # Side force due to sideslip
    cl_beta: float = -0.05                       # Roll due to sideslip (dihedral effect)
    cn_beta: float = 0.05                        # Yaw due to sideslip (weathercock)
    cl_p: float = -0.4                           # Roll damping
    cn_r: float = -0.1                           # Yaw damping
    cl_r: float = 0.05                           # Roll due to yaw rate
    cn_p: float = -0.02                          # Yaw due to roll rate


@dataclass
class SensorSpecification:
    """Individual sensor specification."""
    sensor_id: int = 0
    sensor_type: SensorType = SensorType.IMU

    # Position
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Orientation (for cameras, lidars)
    orientation_roll: float = 0.0
    orientation_pitch: float = 0.0
    orientation_yaw: float = 0.0

    # Update rate
    update_rate_hz: float = 100.0

    # Noise characteristics
    noise_density: float = 0.0
    bias_instability: float = 0.0
    random_walk: float = 0.0

    # Range (for distance sensors)
    min_range_m: float = 0.0
    max_range_m: float = 100.0

    # Field of view (for cameras, lidars)
    fov_horizontal_deg: float = 90.0
    fov_vertical_deg: float = 60.0

    # Resolution (for cameras)
    resolution_x: int = 640
    resolution_y: int = 480

    # Accuracy (for GPS)
    horizontal_accuracy_m: float = 2.5
    vertical_accuracy_m: float = 5.0


@dataclass
class SensorSuite:
    """Complete sensor configuration."""
    sensors: List[SensorSpecification] = field(default_factory=list)

    # Quick access flags
    has_gps: bool = True
    has_imu: bool = True
    has_barometer: bool = True
    has_magnetometer: bool = True
    has_optical_flow: bool = False
    has_lidar: bool = False
    has_camera: bool = False
    has_airspeed: bool = False

    # GPS configuration
    gps_type: str = "standard"                   # standard, rtk, differential
    gps_update_rate_hz: float = 10.0
    gps_accuracy_m: float = 2.5

    # IMU configuration
    imu_update_rate_hz: float = 400.0
    gyro_noise_deg_s: float = 0.01
    accel_noise_m_s2: float = 0.1


@dataclass
class PerformanceEnvelope:
    """Aircraft performance limits."""
    # Speed limits
    max_speed_m_s: float = 0.0
    cruise_speed_m_s: float = 0.0
    stall_speed_m_s: float = 0.0                 # Fixed-wing only
    never_exceed_speed_m_s: float = 0.0          # VNE
    max_mach: float = 0.0

    # Climb/descent
    max_climb_rate_m_s: float = 0.0
    max_descent_rate_m_s: float = 0.0
    service_ceiling_m: float = 0.0

    # Attitude limits
    max_roll_deg: float = 60.0
    max_pitch_deg: float = 45.0
    max_roll_rate_deg_s: float = 180.0
    max_pitch_rate_deg_s: float = 180.0
    max_yaw_rate_deg_s: float = 120.0

    # G limits
    max_positive_g: float = 3.0
    max_negative_g: float = -1.0

    # Endurance
    max_endurance_min: float = 0.0
    max_range_km: float = 0.0

    # Takeoff/landing
    takeoff_distance_m: float = 0.0              # Fixed-wing
    landing_distance_m: float = 0.0

    # Wind limits
    max_wind_m_s: float = 15.0
    max_gust_m_s: float = 20.0


@dataclass
class PayloadSpecification:
    """Payload configuration."""
    payload_id: int = 0
    name: str = ""

    # Mass and CG
    mass_kg: float = 0.0
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Type
    payload_type: str = "generic"                # generic, camera, lidar, cargo, weapon, sensor

    # Gimbal (for cameras)
    has_gimbal: bool = False
    gimbal_roll_range_deg: Tuple[float, float] = (-45, 45)
    gimbal_pitch_range_deg: Tuple[float, float] = (-90, 30)
    gimbal_yaw_range_deg: Tuple[float, float] = (-180, 180)

    # Droppable/releasable
    is_droppable: bool = False
    release_mechanism: str = "none"              # none, servo, pyro, magnetic

    # Power requirements
    power_draw_w: float = 0.0


@dataclass
class CommunicationSystem:
    """Communication and datalink specification."""
    # Control link
    control_frequency_mhz: float = 2400.0
    control_protocol: str = "mavlink"            # mavlink, sbus, crsf, elrs, custom
    control_range_km: float = 10.0
    control_latency_ms: float = 20.0

    # Telemetry
    telemetry_frequency_mhz: float = 915.0
    telemetry_bandwidth_kbps: float = 57.6

    # Video
    video_frequency_mhz: float = 5800.0
    video_bandwidth_mbps: float = 20.0
    video_latency_ms: float = 100.0

    # Data link (for military)
    has_encrypted_link: bool = False
    has_satellite_link: bool = False

    # Failsafe
    failsafe_action: str = "rtl"                 # rtl, land, loiter, continue


@dataclass
class EnvironmentalLimits:
    """Environmental operating limits."""
    # Temperature
    min_operating_temp_c: float = -20.0
    max_operating_temp_c: float = 50.0

    # Altitude
    max_altitude_m: float = 10000.0
    min_altitude_m: float = 0.0

    # Weather
    rain_resistant: bool = False
    ip_rating: str = "IP00"                      # IP rating

    # Icing
    icing_certified: bool = False


# =============================================================================
# MAIN SPECIFICATION CLASS
# =============================================================================

@dataclass
class DroneSpecification:
    """
    COMPREHENSIVE DRONE SPECIFICATION

    This class captures EVERY parameter needed to simulate ANY drone.
    Users fill in what they know, and the system estimates the rest.
    """

    # ==========================================================================
    # IDENTIFICATION
    # ==========================================================================
    name: str = "Unnamed Drone"
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    version: str = "1.0"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())

    # Classification
    airframe_type: AirframeType = AirframeType.QUADCOPTER
    propulsion_type: PropulsionType = PropulsionType.ELECTRIC_BRUSHLESS

    # ==========================================================================
    # PHYSICAL DIMENSIONS
    # ==========================================================================
    # Overall dimensions (meters)
    length_m: float = 0.0
    width_m: float = 0.0                         # Wingspan for fixed-wing
    height_m: float = 0.0

    # Arm length (for multirotors)
    arm_length_m: float = 0.0

    # Folded dimensions (if foldable)
    folded_length_m: float = 0.0
    folded_width_m: float = 0.0
    folded_height_m: float = 0.0
    is_foldable: bool = False

    # ==========================================================================
    # MASS & INERTIA
    # ==========================================================================
    inertial: InertialProperties = field(default_factory=InertialProperties)

    # Quick access to total mass
    mass_kg: float = 0.0                         # Shortcut for inertial.mass_total_kg

    # ==========================================================================
    # PROPULSION SYSTEM
    # ==========================================================================
    # Motor configuration
    motor_config: MotorConfiguration = MotorConfiguration.QUAD_X
    num_motors: int = 4
    motors: List[MotorSpecification] = field(default_factory=list)

    # Propellers
    propellers: List[PropellerSpecification] = field(default_factory=list)

    # Power system
    battery: BatterySpecification = field(default_factory=BatterySpecification)

    # Jet engines (if applicable)
    jet_engines: List[JetEngineSpecification] = field(default_factory=list)

    # Fuel system (if applicable)
    fuel_capacity_kg: float = 0.0
    fuel_type: str = "none"                      # none, jet-a, avgas, diesel

    # ==========================================================================
    # AERODYNAMICS
    # ==========================================================================
    aerodynamics: AerodynamicSurfaces = field(default_factory=AerodynamicSurfaces)

    # Simplified drag model (for multirotors)
    drag_coefficient_xy: float = 0.5             # Horizontal drag
    drag_coefficient_z: float = 1.0              # Vertical drag
    frontal_area_m2: float = 0.01

    # Ground effect
    ground_effect_height_m: float = 0.5          # Height where ground effect starts
    ground_effect_coefficient: float = 0.1

    # ==========================================================================
    # CONTROL SYSTEM
    # ==========================================================================
    flight_controller: FlightControllerType = FlightControllerType.PX4
    flight_controller_model: str = ""            # e.g., "Pixhawk 6C", "Cube Orange"

    # Control rates (deg/s)
    max_roll_rate: float = 220.0
    max_pitch_rate: float = 220.0
    max_yaw_rate: float = 200.0

    # PID tuning (optional - for accurate simulation)
    pid_roll: Tuple[float, float, float] = (6.5, 0.0, 0.0)      # P, I, D
    pid_pitch: Tuple[float, float, float] = (6.5, 0.0, 0.0)
    pid_yaw: Tuple[float, float, float] = (4.0, 0.0, 0.0)
    pid_altitude: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    # ==========================================================================
    # SENSORS
    # ==========================================================================
    sensors: SensorSuite = field(default_factory=SensorSuite)

    # ==========================================================================
    # PERFORMANCE
    # ==========================================================================
    performance: PerformanceEnvelope = field(default_factory=PerformanceEnvelope)

    # Quick access performance (for simple configs)
    max_speed_m_s: float = 20.0
    max_climb_rate_m_s: float = 5.0
    max_thrust_to_weight: float = 2.0
    hover_throttle: float = 0.5                  # Throttle to hover (0-1)

    # ==========================================================================
    # PAYLOAD
    # ==========================================================================
    payloads: List[PayloadSpecification] = field(default_factory=list)
    max_payload_kg: float = 0.0

    # ==========================================================================
    # COMMUNICATION
    # ==========================================================================
    comms: CommunicationSystem = field(default_factory=CommunicationSystem)

    # ==========================================================================
    # ENVIRONMENTAL
    # ==========================================================================
    env_limits: EnvironmentalLimits = field(default_factory=EnvironmentalLimits)

    # ==========================================================================
    # REGULATORY
    # ==========================================================================
    registration_number: str = ""
    weight_class: str = ""                       # micro, small, medium, large
    requires_certification: bool = False
    operational_category: str = ""               # open, specific, certified

    # ==========================================================================
    # METADATA
    # ==========================================================================
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # CAD/model files
    urdf_file: str = ""
    cad_file: str = ""                           # STEP, STL, etc.
    mesh_files: List[str] = field(default_factory=list)

    # Validation status
    is_validated: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    # ==========================================================================
    # METHODS
    # ==========================================================================

    def __post_init__(self):
        """Post-initialization validation and sync."""
        # Sync mass shortcut with inertial
        if self.mass_kg > 0 and self.inertial.mass_total_kg == 0:
            self.inertial.mass_total_kg = self.mass_kg
        elif self.inertial.mass_total_kg > 0:
            self.mass_kg = self.inertial.mass_total_kg

    def to_dict(self) -> Dict[str, Any]:
        """Export specification to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroneSpecification":
        """Create specification from dictionary."""
        # Handle nested dataclasses
        if 'airframe_type' in data and isinstance(data['airframe_type'], str):
            data['airframe_type'] = AirframeType(data['airframe_type'])
        if 'propulsion_type' in data and isinstance(data['propulsion_type'], str):
            data['propulsion_type'] = PropulsionType(data['propulsion_type'])
        # ... handle other nested types
        return cls(**data)

    def to_json(self) -> str:
        """Export specification to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DroneSpecification":
        """Create specification from JSON string."""
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, filepath: str) -> None:
        """Save specification to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> "DroneSpecification":
        """Load specification from file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== {self.name} ===",
            f"Type: {self.airframe_type.value}",
            f"Propulsion: {self.propulsion_type.value}",
            f"Mass: {self.mass_kg:.2f} kg",
            f"Dimensions: {self.length_m:.2f} x {self.width_m:.2f} x {self.height_m:.2f} m",
            f"Motors: {self.num_motors} ({self.motor_config.value})",
            f"Max Speed: {self.max_speed_m_s:.1f} m/s",
            f"Max Climb: {self.max_climb_rate_m_s:.1f} m/s",
            f"Flight Controller: {self.flight_controller.value}",
        ]

        if self.validation_errors:
            lines.append(f"ERRORS: {len(self.validation_errors)}")
        if self.validation_warnings:
            lines.append(f"Warnings: {len(self.validation_warnings)}")

        return "\n".join(lines)


# =============================================================================
# PRESET FACTORY FUNCTIONS
# =============================================================================

def create_quadcopter_spec(
    name: str,
    mass_kg: float,
    arm_length_m: float,
    motor_kv: float = 920,
    propeller_inch: float = 10,
    battery_cells: int = 4,
) -> DroneSpecification:
    """Quick factory for quadcopter specification."""
    spec = DroneSpecification(
        name=name,
        airframe_type=AirframeType.QUADCOPTER_X,
        propulsion_type=PropulsionType.ELECTRIC_BRUSHLESS,
        mass_kg=mass_kg,
        arm_length_m=arm_length_m,
        num_motors=4,
        motor_config=MotorConfiguration.QUAD_X,
    )

    # Create 4 motors in X configuration
    positions = [
        (arm_length_m * 0.707, arm_length_m * 0.707, 0),    # Front-right
        (-arm_length_m * 0.707, arm_length_m * 0.707, 0),   # Back-right
        (-arm_length_m * 0.707, -arm_length_m * 0.707, 0),  # Back-left
        (arm_length_m * 0.707, -arm_length_m * 0.707, 0),   # Front-left
    ]
    spin_dirs = [1, -1, 1, -1]  # CW, CCW, CW, CCW

    for i, (pos, spin) in enumerate(zip(positions, spin_dirs)):
        motor = MotorSpecification(
            motor_id=i,
            position_x=pos[0],
            position_y=pos[1],
            position_z=pos[2],
            kv_rating=motor_kv,
            spin_direction=spin,
        )
        spec.motors.append(motor)

        prop = PropellerSpecification(
            propeller_id=i,
            motor_id=i,
            diameter_inch=propeller_inch,
            diameter_m=propeller_inch * 0.0254,
        )
        spec.propellers.append(prop)

    # Battery
    spec.battery = BatterySpecification(
        chemistry="lipo",
        cell_count=battery_cells,
        nominal_voltage=3.7,
    )

    return spec


def create_fixed_wing_spec(
    name: str,
    mass_kg: float,
    wingspan_m: float,
    wing_area_m2: float,
) -> DroneSpecification:
    """Quick factory for fixed-wing specification."""
    spec = DroneSpecification(
        name=name,
        airframe_type=AirframeType.FIXED_WING_CONVENTIONAL,
        propulsion_type=PropulsionType.ELECTRIC_BRUSHLESS,
        mass_kg=mass_kg,
        width_m=wingspan_m,
        num_motors=1,
    )

    # Main wing
    main_wing = WingSpecification(
        wing_id=0,
        wing_type="main",
        span_m=wingspan_m,
        area_m2=wing_area_m2,
        aspect_ratio=wingspan_m**2 / wing_area_m2,
    )
    spec.aerodynamics.wings.append(main_wing)
    spec.aerodynamics.reference_area_m2 = wing_area_m2
    spec.aerodynamics.reference_span_m = wingspan_m

    return spec

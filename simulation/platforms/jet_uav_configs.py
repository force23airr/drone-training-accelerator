"""
Jet-Powered UAV Platform Configurations.

Defines platform configurations for military jet UAVs:
- Combat UCAVs (X-47B style)
- Loitering munitions (Switchblade, Harop)
- Reconnaissance UAVs (MQ-9, RQ-170)
- Loyal wingman drones (XQ-58)

Each platform includes:
- Physical parameters (mass, dimensions)
- Aerodynamic coefficients
- Propulsion specifications
- Performance envelope
"""

import numpy as np
from .platform_configs import PlatformConfig, register_platform


# =============================================================================
# COMBAT UAV (UCAV) - X-47B STYLE
# =============================================================================

register_platform(
    "x47b_ucav",
    PlatformConfig(
        name="X-47B UCAV",
        platform_type="jet_uav",
        num_motors=1,
        mass=20200.0,           # kg (max takeoff weight)
        arm_length=0.0,         # N/A for fixed-wing
        max_thrust_per_motor=80000.0,  # N (F100-class equivalent)
        max_rpm=0,              # N/A for jet
        observation_dim=24,     # Extended for jet UAV state
        max_episode_steps=5000,
        physics_params={
            # Platform category
            "platform_category": "fixed_wing_jet",
            "control_surface_type": "elevon",  # Flying wing

            # Geometry
            "wingspan": 18.9,           # m
            "wing_area": 70.0,          # m^2
            "aspect_ratio": 5.1,        # Low AR flying wing
            "mean_chord": 3.7,          # m

            # Aerodynamics (flying wing characteristics)
            "CL_0": 0.15,
            "CL_alpha": 4.2,            # Lower than conventional
            "CL_max": 1.2,
            "CD_0": 0.018,              # Low drag, stealthy design
            "oswald_efficiency": 0.75,
            "stall_alpha": np.radians(12),

            # Stability (flying wing - marginally stable)
            "Cm_alpha": -0.2,
            "Cn_beta": 0.08,
            "Cl_beta": -0.03,
            "naturally_stable": False,

            # Control derivatives
            "CL_de": 0.35,
            "Cm_de": -1.0,
            "Cl_da": 0.12,
            "Cn_dr": -0.05,

            # Propulsion
            "engine_type": "turbofan",
            "bypass_ratio": 0.36,
            "max_thrust_sl": 80000,     # N
            "afterburner_thrust_sl": 0,  # No afterburner
            "sfc": 0.00007,             # kg/(N*s)
            "fuel_capacity": 8200,      # kg

            # Performance
            "max_speed": 1000,          # km/h (high subsonic)
            "cruise_speed": 850,        # km/h
            "stall_speed": 180,         # km/h
            "service_ceiling": 12000,   # m
            "max_mach": 0.9,
            "combat_radius": 3900,      # km

            # Inertia
            "Ixx": 80000,               # kg*m^2
            "Iyy": 120000,
            "Izz": 180000,

            # Stealth characteristics
            "rcs": 0.001,               # m^2 (very low)
            "ir_signature": "low",

            # Payload
            "payload_capacity": 2000,   # kg
            "internal_bay_capacity": 2000,
            "hardpoints": 2,            # Internal bays

            # Carrier operations
            "carrier_capable": True,
            "has_tailhook": True,
            "catapult_compatible": True,
        }
    )
)


# =============================================================================
# LOITERING MUNITIONS
# =============================================================================

register_platform(
    "switchblade_600",
    PlatformConfig(
        name="Switchblade 600",
        platform_type="jet_uav",
        num_motors=1,
        mass=50.0,              # kg
        arm_length=0.0,
        max_thrust_per_motor=500.0,  # N (electric pusher)
        max_rpm=0,
        observation_dim=20,
        max_episode_steps=3000,
        physics_params={
            "platform_category": "fixed_wing_jet",
            "control_surface_type": "conventional",

            # Geometry (small UAV)
            "wingspan": 1.3,            # m
            "wing_area": 0.5,           # m^2
            "aspect_ratio": 3.4,
            "mean_chord": 0.38,         # m

            # Aerodynamics
            "CL_0": 0.3,
            "CL_alpha": 4.5,
            "CL_max": 1.3,
            "CD_0": 0.04,
            "oswald_efficiency": 0.7,
            "stall_alpha": np.radians(18),

            # Stability
            "Cm_alpha": -0.6,
            "Cn_beta": 0.12,
            "Cl_beta": -0.04,

            # Control
            "CL_de": 0.5,
            "Cm_de": -1.5,
            "Cl_da": 0.18,
            "Cn_dr": -0.1,

            # Propulsion (electric)
            "engine_type": "electric",
            "max_thrust_sl": 500,       # N
            "battery_capacity_wh": 2000,
            "sfc": 0.0,                 # Electric - use battery model

            # Performance
            "max_speed": 185,           # km/h
            "cruise_speed": 110,        # km/h
            "stall_speed": 65,          # km/h
            "service_ceiling": 4500,    # m
            "max_endurance": 40,        # minutes

            # Inertia
            "Ixx": 2.0,                 # kg*m^2
            "Iyy": 3.5,
            "Izz": 4.5,

            # Munition characteristics
            "warhead_mass": 15,         # kg
            "warhead_type": "anti_armor",
            "terminal_guidance": "eo_ir",
            "man_in_loop": True,
            "abort_capable": True,

            # Launch
            "tube_launched": True,
            "launch_velocity": 20,      # m/s
        }
    )
)

register_platform(
    "harop_loitering",
    PlatformConfig(
        name="IAI Harop",
        platform_type="jet_uav",
        num_motors=1,
        mass=135.0,             # kg
        arm_length=0.0,
        max_thrust_per_motor=1200.0,  # N
        max_rpm=0,
        observation_dim=20,
        max_episode_steps=6000,
        physics_params={
            "platform_category": "fixed_wing_jet",
            "control_surface_type": "conventional",

            # Geometry
            "wingspan": 3.0,            # m
            "wing_area": 1.8,           # m^2
            "aspect_ratio": 5.0,
            "mean_chord": 0.6,          # m

            # Aerodynamics
            "CL_0": 0.25,
            "CL_alpha": 4.8,
            "CL_max": 1.4,
            "CD_0": 0.035,
            "oswald_efficiency": 0.75,
            "stall_alpha": np.radians(16),

            # Stability
            "Cm_alpha": -0.55,
            "Cn_beta": 0.1,
            "Cl_beta": -0.05,

            # Control
            "CL_de": 0.45,
            "Cm_de": -1.3,
            "Cl_da": 0.16,
            "Cn_dr": -0.09,

            # Propulsion (small turbine)
            "engine_type": "turbine_small",
            "max_thrust_sl": 1200,      # N
            "sfc": 0.00015,             # Higher for small turbine
            "fuel_capacity": 30,        # kg

            # Performance
            "max_speed": 400,           # km/h
            "cruise_speed": 280,        # km/h
            "stall_speed": 100,         # km/h
            "service_ceiling": 9000,    # m
            "max_endurance": 6,         # hours

            # Inertia
            "Ixx": 15,                  # kg*m^2
            "Iyy": 25,
            "Izz": 35,

            # Munition characteristics (SEAD focused)
            "warhead_mass": 23,         # kg
            "warhead_type": "blast_frag",
            "seeker_type": "anti_radiation",
            "passive_rf_seeker": True,
            "home_on_jam": True,

            # Operational
            "loiter_time_hours": 6,
            "detection_range_km": 100,  # Radar detection
        }
    )
)


# =============================================================================
# RECONNAISSANCE UAVs (for later phases)
# =============================================================================

register_platform(
    "mq9_reaper",
    PlatformConfig(
        name="MQ-9 Reaper",
        platform_type="jet_uav",
        num_motors=1,
        mass=4760.0,            # kg (max takeoff)
        arm_length=0.0,
        max_thrust_per_motor=9100.0,  # N (TPE331-10)
        max_rpm=0,
        observation_dim=22,
        max_episode_steps=10000,
        physics_params={
            "platform_category": "fixed_wing_jet",
            "control_surface_type": "conventional",

            # Geometry
            "wingspan": 20.1,           # m
            "wing_area": 38.0,          # m^2
            "aspect_ratio": 10.6,
            "mean_chord": 1.9,          # m

            # Aerodynamics (high AR for endurance)
            "CL_0": 0.2,
            "CL_alpha": 5.5,
            "CL_max": 1.4,
            "CD_0": 0.025,
            "oswald_efficiency": 0.85,
            "stall_alpha": np.radians(15),

            # Stability
            "Cm_alpha": -0.5,
            "Cn_beta": 0.1,
            "Cl_beta": -0.05,

            # Control
            "CL_de": 0.4,
            "Cm_de": -1.2,
            "Cl_da": 0.15,
            "Cn_dr": -0.08,

            # Propulsion (turboprop)
            "engine_type": "turboprop",
            "max_thrust_sl": 9100,      # N
            "sfc": 0.00006,
            "fuel_capacity": 1800,      # kg

            # Performance
            "max_speed": 480,           # km/h
            "cruise_speed": 370,        # km/h
            "stall_speed": 120,         # km/h
            "service_ceiling": 15000,   # m
            "max_endurance": 27,        # hours

            # Inertia
            "Ixx": 25000,               # kg*m^2
            "Iyy": 45000,
            "Izz": 55000,

            # Payload
            "payload_capacity": 1700,   # kg
            "hardpoints": 6,
            "sensor_payload": ["eo_ir", "sar", "sigint"],
        }
    )
)

register_platform(
    "rq170_sentinel",
    PlatformConfig(
        name="RQ-170 Sentinel",
        platform_type="jet_uav",
        num_motors=1,
        mass=8500.0,            # kg (estimated)
        arm_length=0.0,
        max_thrust_per_motor=30000.0,  # N
        max_rpm=0,
        observation_dim=22,
        max_episode_steps=8000,
        physics_params={
            "platform_category": "fixed_wing_jet",
            "control_surface_type": "elevon",  # Flying wing

            # Geometry
            "wingspan": 26.0,           # m (estimated)
            "wing_area": 55.0,          # m^2
            "aspect_ratio": 12.3,
            "mean_chord": 2.1,          # m

            # Aerodynamics (high altitude optimized)
            "CL_0": 0.25,
            "CL_alpha": 5.8,
            "CL_max": 1.5,
            "CD_0": 0.015,              # Very clean design
            "oswald_efficiency": 0.88,
            "stall_alpha": np.radians(16),

            # Stability
            "Cm_alpha": -0.3,
            "Cn_beta": 0.06,
            "Cl_beta": -0.03,
            "naturally_stable": False,

            # Control
            "CL_de": 0.38,
            "Cm_de": -1.1,
            "Cl_da": 0.13,
            "Cn_dr": -0.04,

            # Propulsion
            "engine_type": "turbofan",
            "max_thrust_sl": 30000,     # N
            "sfc": 0.00005,
            "fuel_capacity": 3000,      # kg

            # Performance
            "max_speed": 700,           # km/h
            "cruise_speed": 550,        # km/h
            "stall_speed": 140,         # km/h
            "service_ceiling": 15000,   # m
            "max_endurance": 10,        # hours

            # Inertia
            "Ixx": 35000,               # kg*m^2
            "Iyy": 55000,
            "Izz": 75000,

            # Stealth
            "rcs": 0.0001,              # m^2 (minimal)
            "ir_signature": "minimal",

            # Sensors
            "sensor_payload": ["eo_ir", "sar", "sigint"],
        }
    )
)


# =============================================================================
# LOYAL WINGMAN / ATTRITABLE UCAV
# =============================================================================

register_platform(
    "xq58_valkyrie",
    PlatformConfig(
        name="XQ-58A Valkyrie",
        platform_type="jet_uav",
        num_motors=1,
        mass=2720.0,            # kg
        arm_length=0.0,
        max_thrust_per_motor=18000.0,  # N
        max_rpm=0,
        observation_dim=24,
        max_episode_steps=5000,
        physics_params={
            "platform_category": "fixed_wing_jet",
            "control_surface_type": "conventional",

            # Geometry
            "wingspan": 8.2,            # m
            "wing_area": 18.0,          # m^2
            "aspect_ratio": 3.7,
            "mean_chord": 2.2,          # m

            # Aerodynamics (high speed optimized)
            "CL_0": 0.1,
            "CL_alpha": 3.8,
            "CL_max": 1.0,
            "CD_0": 0.022,
            "oswald_efficiency": 0.72,
            "stall_alpha": np.radians(14),

            # Transonic effects
            "drag_divergence_mach": 0.85,
            "compressibility_correction": True,

            # Stability
            "Cm_alpha": -0.45,
            "Cn_beta": 0.09,
            "Cl_beta": -0.04,

            # Control
            "CL_de": 0.4,
            "Cm_de": -1.15,
            "Cl_da": 0.14,
            "Cn_dr": -0.07,

            # Propulsion
            "engine_type": "turbofan",
            "bypass_ratio": 0.4,
            "max_thrust_sl": 18000,     # N
            "sfc": 0.00009,
            "fuel_capacity": 800,       # kg

            # Performance
            "max_speed": 1050,          # km/h
            "max_mach": 0.95,
            "cruise_speed": 900,        # km/h
            "stall_speed": 200,         # km/h
            "service_ceiling": 13700,   # m
            "combat_radius": 3900,      # km

            # Inertia
            "Ixx": 5000,                # kg*m^2
            "Iyy": 12000,
            "Izz": 15000,

            # Stealth
            "rcs": 0.1,                 # m^2 (reduced)
            "stealth_level": "reduced_rcs",

            # Payload
            "internal_weapons_bay": True,
            "payload_capacity": 270,    # kg
            "hardpoints": 2,            # Internal

            # Attritable design
            "unit_cost_factor": "low",
            "expendable": True,

            # Ground launch
            "rocket_assisted_takeoff": True,
            "conventional_runway": False,
        }
    )
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_jet_uav_platforms() -> list:
    """Get list of all registered jet UAV platform IDs."""
    from .platform_configs import get_platforms_by_type
    return get_platforms_by_type("jet_uav")


def get_combat_platforms() -> list:
    """Get platforms suitable for combat missions."""
    return ["x47b_ucav", "xq58_valkyrie"]


def get_loitering_munition_platforms() -> list:
    """Get loitering munition platforms."""
    return ["switchblade_600", "harop_loitering"]


def get_reconnaissance_platforms() -> list:
    """Get reconnaissance UAV platforms."""
    return ["mq9_reaper", "rq170_sentinel"]


def get_carrier_capable_platforms() -> list:
    """Get platforms capable of carrier operations."""
    return ["x47b_ucav"]

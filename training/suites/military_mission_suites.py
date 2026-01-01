"""
Military Training Mission Suites.

Defines mission configurations for jet-powered military UAV training including:
- Fixed-wing fundamentals (takeoff, cruise, landing)
- Strike missions (air-to-ground with threat avoidance)
- Carrier operations (catapult launch, arrested landing)
- Loitering munition operations (loiter, acquire, engage)

Designed for:
- X-47B UCAV (carrier-capable strike)
- Switchblade 600 (loitering munition)
- Harop (anti-radiation loitering munition)
- XQ-58 Valkyrie (loyal wingman)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

from .mission_suites import MissionConfig, register_mission


# =============================================================================
# FIXED-WING FUNDAMENTALS (Beginner)
# =============================================================================

register_mission(
    "fixed_wing_takeoff",
    MissionConfig(
        name="Fixed-Wing Takeoff",
        description="Execute runway takeoff and initial climb-out",
        difficulty="beginner",
        objectives=[
            "Maintain runway centerline during roll",
            "Achieve rotation speed and lift off",
            "Establish stable climb angle",
            "Reach safe altitude (500m)",
        ],
        success_criteria={
            "min_altitude": 500.0,
            "max_lateral_deviation": 20.0,
            "min_climb_rate": 5.0,
        },
        reward_weights={
            "alive": 1.0,
            "altitude_gain": 0.5,
            "centerline": 0.3,
            "airspeed": 0.2,
            "stability": 0.2,
        },
        curriculum_stages=[
            {"wind_speed": 0, "crosswind": 0},
            {"wind_speed": 5, "crosswind": 2},
            {"wind_speed": 10, "crosswind": 5},
        ],
        environment_params={
            "platform": "x47b_ucav",
            "start_on_runway": True,
        }
    )
)

register_mission(
    "cruise_flight",
    MissionConfig(
        name="Cruise Flight",
        description="Maintain level flight at specified altitude and airspeed",
        difficulty="beginner",
        objectives=[
            "Maintain target altitude within 50m",
            "Hold cruise airspeed within 20 m/s",
            "Maintain heading within 5 degrees",
            "Sustain for 60+ seconds",
        ],
        success_criteria={
            "max_altitude_error": 50.0,
            "max_airspeed_error": 20.0,
            "min_cruise_time": 60.0,
        },
        reward_weights={
            "alive": 1.0,
            "altitude_hold": 0.8,
            "airspeed_hold": 0.6,
            "heading_hold": 0.4,
            "fuel_efficiency": 0.2,
        },
        curriculum_stages=[
            {"altitude": 1000, "airspeed": 150, "turbulence": 0},
            {"altitude": 3000, "airspeed": 200, "turbulence": 0.2},
            {"altitude": 5000, "airspeed": 250, "turbulence": 0.5},
        ],
        environment_params={
            "platform": "x47b_ucav",
        }
    )
)

register_mission(
    "runway_landing",
    MissionConfig(
        name="Runway Landing",
        description="Execute precision approach and touchdown",
        difficulty="beginner",
        objectives=[
            "Maintain glideslope (3 degrees)",
            "Align with runway centerline",
            "Control descent rate for smooth touchdown",
            "Stop within runway length",
        ],
        success_criteria={
            "max_glideslope_error": 30.0,
            "max_centerline_error": 10.0,
            "max_touchdown_vertical_speed": 3.0,
            "landing_success": 1,
        },
        reward_weights={
            "alive": 1.0,
            "glideslope": 1.0,
            "centerline": 0.8,
            "touchdown_quality": 2.0,
            "airspeed_control": 0.5,
        },
        curriculum_stages=[
            {"approach_distance": 3000, "crosswind": 0, "visibility": "good"},
            {"approach_distance": 5000, "crosswind": 5, "visibility": "good"},
            {"approach_distance": 8000, "crosswind": 10, "visibility": "reduced"},
        ],
        environment_params={
            "platform": "x47b_ucav",
        }
    )
)

# =============================================================================
# ISR MISSIONS (Intermediate)
# =============================================================================

register_mission(
    "orbit_surveillance",
    MissionConfig(
        name="Orbit Surveillance",
        description="Establish and maintain surveillance orbit pattern",
        difficulty="intermediate",
        objectives=[
            "Fly to designated surveillance area",
            "Establish orbit pattern (racetrack/figure-8)",
            "Maintain sensor coverage of target area",
            "Manage fuel for extended loiter",
        ],
        success_criteria={
            "orbit_time": 300.0,
            "max_orbit_error": 500.0,
            "target_coverage": 0.8,
        },
        reward_weights={
            "alive": 0.5,
            "orbit_accuracy": 1.0,
            "sensor_coverage": 1.5,
            "fuel_management": 0.5,
            "altitude_stability": 0.3,
        },
        environment_params={
            "platform": "mq9_reaper",
            "orbit_radius": 3000,
            "orbit_altitude": 5000,
        }
    )
)

register_mission(
    "route_reconnaissance",
    MissionConfig(
        name="Route Reconnaissance",
        description="Survey multiple points along a designated route",
        difficulty="intermediate",
        objectives=[
            "Navigate sequential survey waypoints",
            "Maintain altitude for sensor operation",
            "Complete route within fuel budget",
            "Collect imagery of all target points",
        ],
        success_criteria={
            "waypoints_completed": 8,
            "fuel_remaining": 0.2,
            "coverage_score": 0.9,
        },
        reward_weights={
            "alive": 0.5,
            "waypoint_accuracy": 1.0,
            "route_efficiency": 0.8,
            "coverage": 1.2,
            "fuel_conservation": 0.4,
        },
        environment_params={
            "platform": "rq170_sentinel",
            "num_waypoints": 10,
            "route_length": 50000,
        }
    )
)

# =============================================================================
# STRIKE MISSIONS (Advanced)
# =============================================================================

register_mission(
    "strike_ingress",
    MissionConfig(
        name="Strike Ingress",
        description="Navigate to target area while avoiding detection",
        difficulty="advanced",
        objectives=[
            "Fly low-altitude ingress route",
            "Minimize time in radar coverage",
            "Use terrain masking when available",
            "Reach target area undetected",
        ],
        success_criteria={
            "detection_time": 30.0,  # Max time detected
            "survived": 1,
            "target_area_reached": 1,
        },
        reward_weights={
            "alive": 2.0,
            "detection_penalty": -1.0,
            "terrain_masking": 0.5,
            "progress_to_target": 0.8,
            "fuel_efficiency": 0.2,
        },
        curriculum_stages=[
            {"threat_density": "low", "terrain": "flat"},
            {"threat_density": "medium", "terrain": "hilly"},
            {"threat_density": "high", "terrain": "mountainous"},
        ],
        environment_params={
            "platform": "x47b_ucav",
            "mission_type": "ingress",
        }
    )
)

register_mission(
    "precision_strike",
    MissionConfig(
        name="Precision Strike",
        description="Engage ground targets with precision weapons",
        difficulty="advanced",
        objectives=[
            "Identify and designate ground targets",
            "Position for optimal weapon delivery",
            "Execute weapon release within envelope",
            "Assess target damage (BDA)",
        ],
        success_criteria={
            "targets_destroyed": 2,
            "weapons_hit_rate": 0.8,
            "survived": 1,
        },
        reward_weights={
            "alive": 1.5,
            "target_destruction": 5.0,
            "weapon_accuracy": 2.0,
            "threat_avoidance": 1.0,
            "mission_completion": 3.0,
        },
        curriculum_stages=[
            {"num_targets": 1, "threat_density": "low", "target_type": "static"},
            {"num_targets": 2, "threat_density": "medium", "target_type": "static"},
            {"num_targets": 3, "threat_density": "high", "target_type": "mobile"},
        ],
        environment_params={
            "platform": "x47b_ucav",
            "weapon_loadout": ["JDAM", "SDB"],
        }
    )
)

register_mission(
    "sead_mission",
    MissionConfig(
        name="SEAD Mission",
        description="Suppress Enemy Air Defenses - neutralize radar threats",
        difficulty="advanced",
        objectives=[
            "Detect and locate enemy radar emissions",
            "Prioritize threat radar systems",
            "Engage and destroy SAM radars",
            "Enable follow-on strike package",
        ],
        success_criteria={
            "radars_destroyed": 2,
            "survived": 1,
            "corridor_cleared": 1,
        },
        reward_weights={
            "alive": 2.0,
            "radar_detection": 1.0,
            "radar_destruction": 5.0,
            "time_in_threat": -0.5,
            "mission_success": 10.0,
        },
        environment_params={
            "platform": "x47b_ucav",
            "threat_types": ["SAM_LONG", "SAM_MEDIUM"],
            "arm_weapon": "AGM-88",
        }
    )
)

# =============================================================================
# CARRIER OPERATIONS (Expert)
# =============================================================================

register_mission(
    "catapult_launch",
    MissionConfig(
        name="Catapult Launch",
        description="Execute EMALS catapult launch from carrier deck",
        difficulty="intermediate",
        objectives=[
            "Configure for catapult launch",
            "Handle catapult acceleration",
            "Establish positive rate of climb",
            "Execute departure procedure",
        ],
        success_criteria={
            "positive_climb_established": 1,
            "departure_altitude": 200.0,
            "max_bank_angle": 30.0,
        },
        reward_weights={
            "alive": 2.0,
            "climb_rate": 1.0,
            "airspeed_control": 0.8,
            "heading_control": 0.6,
            "stability": 0.5,
        },
        environment_params={
            "platform": "x47b_ucav",
            "start_phase": "catapult",
            "sea_state": 3,
        }
    )
)

register_mission(
    "carrier_approach",
    MissionConfig(
        name="Carrier Approach",
        description="Execute precision approach to aircraft carrier",
        difficulty="advanced",
        objectives=[
            "Establish on glideslope (3.5 degrees)",
            "Maintain lineup with angled deck",
            "Control on-speed AOA",
            "Respond to LSO calls",
        ],
        success_criteria={
            "max_glideslope_error": 10.0,
            "max_lineup_error": 5.0,
            "approach_stable": 1,
        },
        reward_weights={
            "alive": 1.5,
            "glideslope_accuracy": 2.0,
            "lineup_accuracy": 2.0,
            "aoa_control": 1.5,
            "rate_of_descent": 1.0,
        },
        curriculum_stages=[
            {"approach_case": "case_i", "sea_state": 2, "deck_motion": False},
            {"approach_case": "case_i", "sea_state": 3, "deck_motion": True},
            {"approach_case": "case_ii", "sea_state": 4, "deck_motion": True},
        ],
        environment_params={
            "platform": "x47b_ucav",
            "start_phase": "approach",
        }
    )
)

register_mission(
    "carrier_landing",
    MissionConfig(
        name="Carrier Arrested Landing",
        description="Complete arrested landing on moving carrier deck",
        difficulty="expert",
        objectives=[
            "Fly precision approach in groove",
            "Deploy tailhook at appropriate time",
            "Catch wire (target: 3-wire)",
            "Handle deck motion compensation",
        ],
        success_criteria={
            "arrested_landing": 1,
            "wire_caught": 3,  # Target wire
            "max_bolters": 2,
        },
        reward_weights={
            "alive": 2.0,
            "glideslope": 2.5,
            "lineup": 2.5,
            "wire_catch": 10.0,
            "optimal_wire": 5.0,
            "deck_motion_handling": 1.5,
            "bolter_penalty": -5.0,
        },
        curriculum_stages=[
            {"sea_state": 2, "wind_over_deck": 25, "visibility": "good"},
            {"sea_state": 3, "wind_over_deck": 30, "visibility": "good"},
            {"sea_state": 4, "wind_over_deck": 35, "visibility": "reduced"},
            {"sea_state": 5, "wind_over_deck": 40, "visibility": "night"},
        ],
        environment_params={
            "platform": "x47b_ucav",
            "approach_case": "case_i",
            "enable_deck_motion": True,
        }
    )
)

register_mission(
    "carrier_bolter_recovery",
    MissionConfig(
        name="Carrier Bolter Recovery",
        description="Execute go-around after missed wire",
        difficulty="advanced",
        objectives=[
            "Recognize bolter situation",
            "Apply max power and climb",
            "Fly departure pattern",
            "Set up for another approach",
        ],
        success_criteria={
            "safe_climbout": 1,
            "pattern_altitude": 600.0,
            "re_approach_established": 1,
        },
        reward_weights={
            "alive": 2.0,
            "reaction_time": 1.0,
            "climb_rate": 1.5,
            "pattern_accuracy": 1.0,
            "airspeed_control": 0.8,
        },
        environment_params={
            "platform": "x47b_ucav",
            "start_condition": "bolter",
        }
    )
)

# =============================================================================
# LOITERING MUNITION OPERATIONS (Advanced/Expert)
# =============================================================================

register_mission(
    "loiter_establishment",
    MissionConfig(
        name="Loiter Pattern Establishment",
        description="Transit to area and establish loiter pattern",
        difficulty="intermediate",
        objectives=[
            "Navigate to designated loiter area",
            "Establish efficient loiter pattern",
            "Maintain altitude for sensor coverage",
            "Conserve battery/fuel for max endurance",
        ],
        success_criteria={
            "loiter_established": 1,
            "loiter_time": 300.0,
            "max_drift": 500.0,
        },
        reward_weights={
            "alive": 1.0,
            "transit_efficiency": 0.5,
            "loiter_accuracy": 1.5,
            "energy_conservation": 0.8,
            "sensor_coverage": 1.0,
        },
        environment_params={
            "platform": "switchblade_600",
            "loiter_radius": 1500,
        }
    )
)

register_mission(
    "target_acquisition",
    MissionConfig(
        name="Target Acquisition",
        description="Detect, identify, and track ground targets",
        difficulty="advanced",
        objectives=[
            "Search designated area",
            "Detect target signatures",
            "Classify and identify targets",
            "Establish tracking lock",
        ],
        success_criteria={
            "targets_detected": 2,
            "track_time": 30.0,
            "false_positives": 0,
        },
        reward_weights={
            "alive": 1.0,
            "detection": 2.0,
            "classification_accuracy": 1.5,
            "track_stability": 1.0,
            "search_efficiency": 0.5,
        },
        environment_params={
            "platform": "switchblade_600",
            "target_types": ["vehicle", "infantry"],
        }
    )
)

register_mission(
    "terminal_attack",
    MissionConfig(
        name="Loitering Munition Terminal Attack",
        description="Execute terminal dive attack on confirmed target",
        difficulty="expert",
        objectives=[
            "Confirm target designation",
            "Compute optimal attack geometry",
            "Execute terminal dive",
            "Maintain guidance to impact",
        ],
        success_criteria={
            "target_hit": 1,
            "cep": 5.0,  # meters
            "abort_if_needed": 1,
        },
        reward_weights={
            "impact_accuracy": 10.0,
            "dive_angle": 1.0,
            "guidance_stability": 1.5,
            "target_type_bonus": 2.0,
        },
        curriculum_stages=[
            {"target_motion": "static", "countermeasures": False},
            {"target_motion": "linear", "countermeasures": False},
            {"target_motion": "evasive", "countermeasures": True},
        ],
        environment_params={
            "platform": "switchblade_600",
            "munition_type": "switchblade",
        }
    )
)

register_mission(
    "anti_radiation_attack",
    MissionConfig(
        name="Anti-Radiation Attack (Harop)",
        description="Home on radar emissions for SEAD operations",
        difficulty="expert",
        objectives=[
            "Detect radar emissions",
            "Locate emitter position",
            "Execute anti-radiation attack profile",
            "Handle intermittent emissions",
        ],
        success_criteria={
            "emitter_destroyed": 1,
            "tracking_maintained": 1,
        },
        reward_weights={
            "alive": 1.0,
            "emission_detection": 1.5,
            "tracking_accuracy": 2.0,
            "attack_success": 10.0,
            "intermittent_handling": 1.0,
        },
        environment_params={
            "platform": "harop_loitering",
            "munition_type": "harop",
            "emitter_behavior": "intermittent",
        }
    )
)

register_mission(
    "abort_wave_off",
    MissionConfig(
        name="Attack Abort and Wave-Off",
        description="Safely abort attack and return to loiter",
        difficulty="advanced",
        objectives=[
            "Recognize abort criteria",
            "Execute safe pull-off maneuver",
            "Return to loiter pattern",
            "Re-acquire target if appropriate",
        ],
        success_criteria={
            "safe_abort": 1,
            "loiter_resumed": 1,
            "altitude_maintained": 1,
        },
        reward_weights={
            "alive": 2.0,
            "abort_timing": 1.5,
            "recovery_altitude": 1.0,
            "loiter_re_establishment": 1.0,
            "collateral_avoidance": 5.0,
        },
        environment_params={
            "platform": "switchblade_600",
            "abort_scenario": "civilian_presence",
        }
    )
)

# =============================================================================
# COORDINATED MISSIONS (Expert)
# =============================================================================

register_mission(
    "loyal_wingman",
    MissionConfig(
        name="Loyal Wingman Operations",
        description="Operate as autonomous wingman to manned aircraft",
        difficulty="expert",
        objectives=[
            "Maintain formation with lead aircraft",
            "Execute directed maneuvers",
            "Provide sensor/weapon coverage",
            "Handle communication loss gracefully",
        ],
        success_criteria={
            "formation_accuracy": 0.9,
            "command_compliance": 0.95,
            "mission_support": 1,
        },
        reward_weights={
            "alive": 1.5,
            "formation_keeping": 2.0,
            "command_response": 1.5,
            "sensor_coordination": 1.0,
            "autonomous_decision": 0.8,
        },
        environment_params={
            "platform": "xq58_valkyrie",
            "lead_aircraft": "f35",
            "formation_type": "combat_spread",
        }
    )
)

register_mission(
    "swarm_area_denial",
    MissionConfig(
        name="Swarm Area Denial",
        description="Coordinate loitering munition swarm for area coverage",
        difficulty="expert",
        objectives=[
            "Distribute swarm for maximum coverage",
            "Coordinate target assignments",
            "Sequence attacks for effectiveness",
            "Maintain swarm cohesion",
        ],
        success_criteria={
            "area_coverage": 0.85,
            "targets_engaged": 5,
            "coordination_score": 0.9,
        },
        reward_weights={
            "coverage": 2.0,
            "coordination": 2.5,
            "target_engagement": 3.0,
            "resource_efficiency": 1.0,
            "swarm_survival": 1.5,
        },
        environment_params={
            "platform": "switchblade_600",
            "swarm_size": 6,
            "num_targets": 8,
        }
    )
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_military_missions() -> List[str]:
    """Get all military mission IDs."""
    return [
        # Fundamentals
        "fixed_wing_takeoff",
        "cruise_flight",
        "runway_landing",
        # ISR
        "orbit_surveillance",
        "route_reconnaissance",
        # Strike
        "strike_ingress",
        "precision_strike",
        "sead_mission",
        # Carrier
        "catapult_launch",
        "carrier_approach",
        "carrier_landing",
        "carrier_bolter_recovery",
        # Loitering
        "loiter_establishment",
        "target_acquisition",
        "terminal_attack",
        "anti_radiation_attack",
        "abort_wave_off",
        # Coordinated
        "loyal_wingman",
        "swarm_area_denial",
    ]


def get_missions_by_platform(platform: str) -> List[str]:
    """Get missions suitable for a specific platform."""
    platform_missions = {
        "x47b_ucav": [
            "fixed_wing_takeoff", "cruise_flight", "runway_landing",
            "strike_ingress", "precision_strike", "sead_mission",
            "catapult_launch", "carrier_approach", "carrier_landing",
            "carrier_bolter_recovery",
        ],
        "switchblade_600": [
            "loiter_establishment", "target_acquisition",
            "terminal_attack", "abort_wave_off", "swarm_area_denial",
        ],
        "harop_loitering": [
            "loiter_establishment", "target_acquisition",
            "anti_radiation_attack", "abort_wave_off",
        ],
        "mq9_reaper": [
            "cruise_flight", "runway_landing",
            "orbit_surveillance", "route_reconnaissance",
        ],
        "rq170_sentinel": [
            "cruise_flight", "route_reconnaissance",
        ],
        "xq58_valkyrie": [
            "fixed_wing_takeoff", "cruise_flight",
            "strike_ingress", "loyal_wingman",
        ],
    }
    return platform_missions.get(platform, [])


def get_curriculum_sequence(mission_category: str) -> List[str]:
    """Get recommended training sequence for a mission category."""
    sequences = {
        "carrier_qualification": [
            "fixed_wing_takeoff",
            "cruise_flight",
            "runway_landing",
            "catapult_launch",
            "carrier_approach",
            "carrier_bolter_recovery",
            "carrier_landing",
        ],
        "strike_qualification": [
            "fixed_wing_takeoff",
            "cruise_flight",
            "runway_landing",
            "strike_ingress",
            "precision_strike",
            "sead_mission",
        ],
        "loitering_munition_qualification": [
            "loiter_establishment",
            "target_acquisition",
            "abort_wave_off",
            "terminal_attack",
        ],
    }
    return sequences.get(mission_category, [])

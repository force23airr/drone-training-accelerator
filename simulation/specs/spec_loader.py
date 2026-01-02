"""
Drone Specification Loader

Loads customer drone specifications from YAML files and converts
them to simulator-compatible configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import fields

from .drone_spec import (
    DroneSpec,
    WeaponSpec,
    PropulsionType,
    ControlSurfaceType,
    DroneCategory,
)
from .aero_estimator import AeroEstimator, EstimatedAero
from .validator import SpecValidator, ValidationResult


class DroneSpecLoader:
    """
    Load and convert drone specifications from YAML files.

    Handles:
    - YAML parsing with validation
    - Conversion to DroneSpec dataclass
    - Auto-calculation of missing aerodynamic parameters
    - Conversion to simulator PlatformConfig and DogfightConfig
    """

    def __init__(self):
        self.estimator = AeroEstimator()
        self.validator = SpecValidator()

    def load_from_yaml(self, path: Union[str, Path]) -> DroneSpec:
        """
        Load a drone specification from a YAML file.

        Args:
            path: Path to the YAML spec file

        Returns:
            DroneSpec instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return self._parse_yaml_data(data)

    def load_from_dict(self, data: Dict[str, Any]) -> DroneSpec:
        """
        Load a drone specification from a dictionary.

        Args:
            data: Dictionary with spec data (same format as YAML)

        Returns:
            DroneSpec instance
        """
        return self._parse_yaml_data(data)

    def _parse_yaml_data(self, data: Dict[str, Any]) -> DroneSpec:
        """Parse YAML/dict data into DroneSpec."""

        # Flatten nested structure
        flat = {}

        # Identity section
        if 'identity' in data:
            identity = data['identity']
            flat['name'] = identity.get('name', 'Unknown')
            flat['manufacturer'] = identity.get('manufacturer', 'Unknown')
            flat['version'] = identity.get('version', '1.0.0')
            if 'category' in identity:
                flat['category'] = DroneCategory(identity['category'])

        # Physical section
        if 'physical' in data:
            phys = data['physical']
            flat['mass_kg'] = phys.get('mass_kg', 0)
            flat['wingspan_m'] = phys.get('wingspan_m', 0)
            flat['length_m'] = phys.get('length_m', 0)
            flat['wing_area_m2'] = phys.get('wing_area_m2', 0)
            flat['height_m'] = phys.get('height_m')
            flat['fuselage_diameter_m'] = phys.get('fuselage_diameter_m')
            flat['mean_chord_m'] = phys.get('mean_chord_m')

        # Performance section
        if 'performance' in data:
            perf = data['performance']
            flat['max_speed_ms'] = perf.get('max_speed_ms', 0)
            flat['cruise_speed_ms'] = perf.get('cruise_speed_ms', 0)
            flat['stall_speed_ms'] = perf.get('stall_speed_ms', 0)
            flat['max_g_force'] = perf.get('max_g_force', 6.0)
            flat['max_altitude_m'] = perf.get('max_altitude_m', 10000)
            flat['min_g_force'] = perf.get('min_g_force')
            flat['max_climb_rate_ms'] = perf.get('max_climb_rate_ms')
            flat['max_roll_rate_degs'] = perf.get('max_roll_rate_degs')
            flat['max_pitch_rate_degs'] = perf.get('max_pitch_rate_degs')
            flat['max_yaw_rate_degs'] = perf.get('max_yaw_rate_degs')
            flat['max_bank_angle_deg'] = perf.get('max_bank_angle_deg')

        # Propulsion section
        if 'propulsion' in data:
            prop = data['propulsion']
            if 'type' in prop:
                flat['propulsion_type'] = PropulsionType(prop['type'])
            flat['max_thrust_n'] = prop.get('max_thrust_n', 0)
            flat['fuel_capacity_kg'] = prop.get('fuel_capacity_kg', 100)
            flat['num_engines'] = prop.get('num_engines', 1)
            flat['idle_thrust_fraction'] = prop.get('idle_thrust_fraction')
            flat['specific_fuel_consumption'] = prop.get('sfc')
            flat['thrust_vectoring'] = prop.get('thrust_vectoring', False)
            flat['afterburner'] = prop.get('afterburner', False)
            flat['afterburner_thrust_ratio'] = prop.get('afterburner_thrust_ratio')

        # Control section
        if 'control' in data:
            ctrl = data['control']
            if 'surface_type' in ctrl:
                flat['control_surface_type'] = ControlSurfaceType(ctrl['surface_type'])
            flat['aileron_max_deflection_deg'] = ctrl.get('aileron_max_deg')
            flat['elevator_max_deflection_deg'] = ctrl.get('elevator_max_deg')
            flat['rudder_max_deflection_deg'] = ctrl.get('rudder_max_deg')
            flat['has_speedbrake'] = ctrl.get('has_speedbrake', False)
            flat['has_leading_edge_slats'] = ctrl.get('has_slats', False)

        # Inertia section
        if 'inertia' in data:
            inertia = data['inertia']
            flat['Ixx_kgm2'] = inertia.get('Ixx')
            flat['Iyy_kgm2'] = inertia.get('Iyy')
            flat['Izz_kgm2'] = inertia.get('Izz')

        # Aerodynamics section (optional, for advanced users)
        if 'aerodynamics' in data:
            aero = data['aerodynamics']
            flat['CL_0'] = aero.get('CL_0')
            flat['CL_alpha'] = aero.get('CL_alpha')
            flat['CL_max'] = aero.get('CL_max')
            flat['CD_0'] = aero.get('CD_0')
            flat['oswald_efficiency'] = aero.get('oswald_efficiency')
            flat['Cm_alpha'] = aero.get('Cm_alpha')
            flat['Cn_beta'] = aero.get('Cn_beta')
            flat['Cl_beta'] = aero.get('Cl_beta')

        # Payload section
        if 'payload' in data:
            pay = data['payload']
            flat['payload_capacity_kg'] = pay.get('capacity_kg', 0)
            flat['hardpoints'] = pay.get('hardpoints', 0)
            flat['internal_bay'] = pay.get('internal_bay', False)

            # Parse weapons
            if 'weapons' in pay:
                weapons = []
                for w in pay['weapons']:
                    weapons.append(WeaponSpec(
                        weapon_type=w.get('type', 'gun'),
                        quantity=w.get('quantity', 1),
                        ammo=w.get('ammo', 500),
                        range_m=w.get('range_m', 300),
                        damage=w.get('damage', 10),
                        cooldown_s=w.get('cooldown_s', 0.1),
                        mass_kg=w.get('mass_kg', 50),
                    ))
                flat['weapons'] = weapons

        # Sensors section
        if 'sensors' in data:
            sens = data['sensors']
            flat['radar_range_km'] = sens.get('radar_range_km')
            flat['irst_equipped'] = sens.get('irst', False)
            flat['rwr_equipped'] = sens.get('rwr', False)
            flat['datalink_range_km'] = sens.get('datalink_range_km')

        # Capabilities section
        if 'capabilities' in data:
            cap = data['capabilities']
            flat['carrier_capable'] = cap.get('carrier_capable', False)
            flat['catapult_launch'] = cap.get('catapult_launch', False)
            flat['arrested_landing'] = cap.get('arrested_landing', False)
            flat['autonomous_capable'] = cap.get('autonomous', True)
            flat['swarm_capable'] = cap.get('swarm', False)
            flat['stealth_features'] = cap.get('stealth', False)
            flat['all_weather'] = cap.get('all_weather', True)

        # Metadata
        if 'notes' in data:
            flat['notes'] = data['notes']
        if 'tags' in data:
            flat['tags'] = data['tags']
        if 'custom' in data:
            flat['custom_params'] = data['custom']

        # Filter out None values and create DroneSpec
        # Only include fields that exist in DroneSpec
        valid_fields = {f.name for f in fields(DroneSpec)}
        filtered = {k: v for k, v in flat.items() if k in valid_fields and v is not None}

        return DroneSpec(**filtered)

    def validate(self, spec: DroneSpec) -> ValidationResult:
        """
        Validate a drone specification.

        Args:
            spec: DroneSpec to validate

        Returns:
            ValidationResult with errors and warnings
        """
        return self.validator.validate(spec)

    def estimate_aero(self, spec: DroneSpec) -> EstimatedAero:
        """
        Estimate aerodynamic parameters for a spec.

        Args:
            spec: DroneSpec to estimate for

        Returns:
            EstimatedAero with all estimated parameters
        """
        return self.estimator.estimate_from_spec(spec)

    def to_platform_config(self, spec: DroneSpec) -> Dict[str, Any]:
        """
        Convert DroneSpec to internal PlatformConfig format.

        This creates the physics_params dictionary used by the simulator.

        Args:
            spec: Validated DroneSpec

        Returns:
            Dictionary compatible with PlatformConfig
        """
        # Estimate aerodynamics
        aero = self.estimate_aero(spec)

        # Build physics_params
        physics_params = {
            # Geometry
            "mass": spec.mass_kg,
            "wingspan": spec.wingspan_m,
            "length": spec.length_m,
            "wing_area": spec.wing_area_m2,
            "aspect_ratio": spec.aspect_ratio,
            "mean_chord": spec.mean_chord_m or (spec.wing_area_m2 / spec.wingspan_m),

            # Performance limits
            "max_speed": spec.max_speed_ms,
            "cruise_speed": spec.cruise_speed_ms,
            "stall_speed": spec.stall_speed_ms,
            "max_g_force": spec.max_g_force,
            "service_ceiling": spec.max_altitude_m,

            # Propulsion
            "engine_type": spec.propulsion_type.value,
            "max_thrust_sl": spec.max_thrust_n,
            "fuel_capacity": spec.fuel_capacity_kg,
            "num_engines": spec.num_engines,
            "has_afterburner": spec.afterburner,

            # Aerodynamics (estimated or provided)
            "CL_0": aero.CL_0,
            "CL_alpha": aero.CL_alpha,
            "CL_max": aero.CL_max,
            "CD_0": aero.CD_0,
            "oswald_efficiency": aero.oswald_efficiency,

            # Stability derivatives
            "Cm_alpha": aero.Cm_alpha,
            "Cm_q": aero.Cm_q,
            "Cn_beta": aero.Cn_beta,
            "Cn_r": aero.Cn_r,
            "Cl_beta": aero.Cl_beta,
            "Cl_p": aero.Cl_p,

            # Control derivatives
            "CL_de": aero.CL_de,
            "Cm_de": aero.Cm_de,
            "Cl_da": aero.Cl_da,
            "Cn_dr": aero.Cn_dr,

            # Stall
            "stall_alpha": aero.alpha_stall_rad,

            # Inertia
            "Ixx": aero.Ixx,
            "Iyy": aero.Iyy,
            "Izz": aero.Izz,

            # Control surface type
            "control_type": spec.control_surface_type.value,

            # Payload
            "payload_capacity": spec.payload_capacity_kg,
            "hardpoints": spec.hardpoints,

            # Custom
            **spec.custom_params,
        }

        return {
            "name": spec.name,
            "platform_type": spec.category.value,
            "num_motors": spec.num_engines,
            "mass": spec.mass_kg,
            "arm_length": spec.wingspan_m / 2,
            "max_thrust_per_motor": spec.max_thrust_n / max(spec.num_engines, 1),
            "max_rpm": 0,  # Not applicable for jets
            "observation_dim": 21,
            "max_episode_steps": 10000,
            "physics_params": physics_params,
        }

    def to_dogfight_config(self, spec: DroneSpec, opponent_spec: Optional[DroneSpec] = None) -> Dict[str, Any]:
        """
        Create DogfightConfig from DroneSpec.

        Args:
            spec: Player's drone spec
            opponent_spec: Optional opponent spec (uses same as player if None)

        Returns:
            Dictionary compatible with DogfightConfig
        """
        # Estimate aero for physics
        aero = self.estimate_aero(spec)

        # Valid weapon types in the simulator
        VALID_WEAPON_TYPES = {"gun", "missile_ir", "missile_radar", "laser"}

        # Map custom weapon types to valid ones
        WEAPON_TYPE_MAP = {
            "warhead": "missile_ir",  # Treat warhead as high-damage missile
            "bomb": "missile_ir",
            "rocket": "missile_ir",
        }

        # Build weapons config from spec
        weapons_config = []
        for w in spec.weapons:
            weapon_type = w.weapon_type
            # Map to valid type if needed
            if weapon_type not in VALID_WEAPON_TYPES:
                weapon_type = WEAPON_TYPE_MAP.get(weapon_type, "gun")
            weapons_config.append({
                "type": weapon_type,
                "ammo": w.ammo,
                "range": w.range_m,
                "damage": w.damage,
                "cooldown": w.cooldown_s,
            })

        # Default weapons if none specified
        if not weapons_config:
            weapons_config = [
                {"type": "gun", "ammo": 500, "range": 300, "damage": 10, "cooldown": 0.1},
                {"type": "missile_ir", "ammo": 4, "range": 2000, "damage": 100, "cooldown": 2.0},
            ]

        return {
            # Arena sized for drone performance - smaller for better engagement
            "arena_size": max(600, min(spec.max_speed_ms * 8, 1500)),
            "arena_height_min": 100,
            "arena_height_max": min(spec.max_altitude_m, 1500),

            # Teams
            "num_red": 1,
            "num_blue": 1,

            # Performance limits from spec
            "min_speed": spec.stall_speed_ms * 1.1,  # Safety margin
            "max_speed": spec.max_speed_ms,
            "max_g_force": spec.max_g_force,

            # Weapons
            "weapons_config": weapons_config,

            # Custom drone config (for physics)
            "red_drone_config": {
                "name": spec.name,
                "mass": spec.mass_kg,
                "wing_area": spec.wing_area_m2,
                "thrust_max": spec.max_thrust_n,
                "CL_alpha": aero.CL_alpha,
                "CD_0": aero.CD_0,
                "oswald": aero.oswald_efficiency,
                "aspect_ratio": spec.aspect_ratio,
                "Ixx": aero.Ixx,
                "Iyy": aero.Iyy,
                "Izz": aero.Izz,
            },

            # Match settings
            "respawn_enabled": True,
            "kills_to_win": 3,
            "max_match_time": 60,
        }

    def save_to_yaml(self, spec: DroneSpec, path: Union[str, Path]):
        """
        Save a DroneSpec to a YAML file.

        Args:
            spec: DroneSpec to save
            path: Output path
        """
        data = spec.to_dict()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_spec(path: Union[str, Path]) -> DroneSpec:
    """Convenience function to load a spec from YAML."""
    loader = DroneSpecLoader()
    return loader.load_from_yaml(path)


def validate_spec_file(path: Union[str, Path]) -> ValidationResult:
    """Convenience function to validate a spec file."""
    loader = DroneSpecLoader()
    spec = loader.load_from_yaml(path)
    return loader.validate(spec)

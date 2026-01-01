#!/usr/bin/env python3
"""
Interactive Drone Onboarding CLI

Walk users through the process of adding their custom drone
to the training platform.
"""

import argparse
from pathlib import Path
from typing import Optional

from onboarding.drone_specification import (
    DroneSpecification,
    AirframeType,
    PropulsionType,
    MotorConfiguration,
    FlightControllerType,
)
from onboarding.simulation_generator import onboard_drone, OnboardingResult
from onboarding.cad_importer import (
    import_from_urdf,
    import_from_step,
    import_from_stl,
    import_from_json,
)


def interactive_onboarding() -> OnboardingResult:
    """
    Interactive CLI wizard for drone onboarding.

    Walks user through entering drone specifications.
    """
    print("=" * 60)
    print("DRONE ONBOARDING WIZARD")
    print("=" * 60)
    print("\nWelcome! Let's add your drone to the training platform.")
    print("You can provide as much or as little info as you have -")
    print("we'll estimate the rest.\n")

    spec = DroneSpecification()

    # Basic info
    spec.name = input("Drone name: ").strip() or "My Custom Drone"
    spec.manufacturer = input("Manufacturer (optional): ").strip()
    spec.model = input("Model (optional): ").strip()

    # Airframe type
    print("\nAirframe type:")
    print("  1. Quadcopter")
    print("  2. Hexacopter")
    print("  3. Octocopter")
    print("  4. Fixed-wing")
    print("  5. VTOL (tiltrotor/quadplane)")
    print("  6. Other/Custom")

    airframe_choice = input("Select [1-6] (default: 1): ").strip() or "1"
    airframe_map = {
        "1": AirframeType.QUADCOPTER_X,
        "2": AirframeType.HEXACOPTER_X,
        "3": AirframeType.OCTOCOPTER_X,
        "4": AirframeType.FIXED_WING_CONVENTIONAL,
        "5": AirframeType.VTOL_QUADPLANE,
        "6": AirframeType.CUSTOM,
    }
    spec.airframe_type = airframe_map.get(airframe_choice, AirframeType.QUADCOPTER_X)

    # Set num_motors based on airframe
    motor_defaults = {
        AirframeType.QUADCOPTER_X: 4,
        AirframeType.HEXACOPTER_X: 6,
        AirframeType.OCTOCOPTER_X: 8,
        AirframeType.FIXED_WING_CONVENTIONAL: 1,
        AirframeType.VTOL_QUADPLANE: 5,
    }
    spec.num_motors = motor_defaults.get(spec.airframe_type, 4)

    # Mass
    print("\n--- Physical Properties ---")
    mass_input = input("Total mass (kg) [required]: ").strip()
    try:
        spec.mass_kg = float(mass_input)
    except ValueError:
        print("Invalid mass. Using default 1.0 kg")
        spec.mass_kg = 1.0

    # Arm length (for multirotors)
    if "fixed_wing" not in spec.airframe_type.value:
        arm_input = input("Arm length (meters) [optional - will estimate]: ").strip()
        if arm_input:
            try:
                spec.arm_length_m = float(arm_input)
            except ValueError:
                pass

    # Motor info
    print("\n--- Propulsion ---")
    kv_input = input("Motor KV rating [optional]: ").strip()
    if kv_input:
        try:
            kv = float(kv_input)
            # Will be used when creating motor specs
        except ValueError:
            pass

    prop_input = input("Propeller size (inches) [optional]: ").strip()

    # Battery
    print("\n--- Battery ---")
    cell_input = input("Battery cell count (e.g., 4 for 4S) [optional]: ").strip()
    if cell_input:
        try:
            spec.battery.cell_count = int(cell_input)
        except ValueError:
            pass

    capacity_input = input("Battery capacity (mAh) [optional]: ").strip()
    if capacity_input:
        try:
            spec.battery.capacity_mah = float(capacity_input)
        except ValueError:
            pass

    # Flight controller
    print("\n--- Flight Controller ---")
    print("  1. PX4")
    print("  2. ArduPilot")
    print("  3. Betaflight")
    print("  4. DJI")
    print("  5. Other/Custom")

    fc_choice = input("Select [1-5] (default: 1): ").strip() or "1"
    fc_map = {
        "1": FlightControllerType.PX4,
        "2": FlightControllerType.ARDUPILOT,
        "3": FlightControllerType.BETAFLIGHT,
        "4": FlightControllerType.DJI,
        "5": FlightControllerType.CUSTOM,
    }
    spec.flight_controller = fc_map.get(fc_choice, FlightControllerType.PX4)

    # Performance (optional)
    print("\n--- Performance (optional) ---")
    speed_input = input("Max speed (m/s) [optional]: ").strip()
    if speed_input:
        try:
            spec.max_speed_m_s = float(speed_input)
        except ValueError:
            pass

    climb_input = input("Max climb rate (m/s) [optional]: ").strip()
    if climb_input:
        try:
            spec.max_climb_rate_m_s = float(climb_input)
        except ValueError:
            pass

    tw_input = input("Thrust-to-weight ratio [optional]: ").strip()
    if tw_input:
        try:
            spec.max_thrust_to_weight = float(tw_input)
        except ValueError:
            pass

    # Sensors
    print("\n--- Sensors ---")
    spec.sensors.has_gps = input("Has GPS? (y/n, default: y): ").strip().lower() != 'n'
    spec.sensors.has_optical_flow = input("Has optical flow? (y/n, default: n): ").strip().lower() == 'y'
    spec.sensors.has_lidar = input("Has LiDAR? (y/n, default: n): ").strip().lower() == 'y'
    spec.sensors.has_camera = input("Has camera? (y/n, default: n): ").strip().lower() == 'y'

    # CAD files
    print("\n--- CAD/Model Files (optional) ---")
    urdf_input = input("URDF file path [optional]: ").strip()
    if urdf_input and Path(urdf_input).exists():
        spec.urdf_file = urdf_input

    cad_input = input("CAD file path (STEP/STL) [optional]: ").strip()
    if cad_input and Path(cad_input).exists():
        spec.cad_file = cad_input

    # Summary
    print("\n" + "=" * 60)
    print("SPECIFICATION SUMMARY")
    print("=" * 60)
    print(spec.summary())

    # Confirm
    confirm = input("\nProceed with onboarding? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Onboarding cancelled.")
        return OnboardingResult()

    # Onboard
    print("\nOnboarding drone...")
    result = onboard_drone(spec)

    print("\n" + result.summary())

    return result


def cli_main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Onboard custom drones to the training platform"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Interactive wizard
    wizard_parser = subparsers.add_parser('wizard', help='Interactive onboarding wizard')

    # Import from file
    import_parser = subparsers.add_parser('import', help='Import from file')
    import_parser.add_argument('file', type=str, help='File to import (JSON, URDF, STEP, STL)')
    import_parser.add_argument('--output', '-o', type=str, help='Output directory')

    # Quick onboard
    quick_parser = subparsers.add_parser('quick', help='Quick onboard with minimal info')
    quick_parser.add_argument('name', type=str, help='Drone name')
    quick_parser.add_argument('mass', type=float, help='Mass in kg')
    quick_parser.add_argument('--arm-length', type=float, help='Arm length in meters')
    quick_parser.add_argument('--type', choices=['quad', 'hex', 'octo', 'fixed_wing'],
                              default='quad', help='Airframe type')

    # Validate spec
    validate_parser = subparsers.add_parser('validate', help='Validate a spec file')
    validate_parser.add_argument('file', type=str, help='JSON spec file to validate')

    args = parser.parse_args()

    if args.command == 'wizard':
        result = interactive_onboarding()
        return 0 if result.success else 1

    elif args.command == 'import':
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return 1

        suffix = filepath.suffix.lower()
        if suffix == '.json':
            spec = import_from_json(str(filepath))
        elif suffix == '.urdf':
            spec = import_from_urdf(str(filepath))
        elif suffix in ['.step', '.stp']:
            spec = import_from_step(str(filepath))
        elif suffix == '.stl':
            spec = import_from_stl(str(filepath))
        else:
            print(f"Unsupported file format: {suffix}")
            return 1

        result = onboard_drone(spec, output_dir=args.output)
        print(result.summary())
        return 0 if result.success else 1

    elif args.command == 'quick':
        from onboarding.simulation_generator import quick_onboard_quadcopter

        spec = DroneSpecification(
            name=args.name,
            mass_kg=args.mass,
        )

        if args.arm_length:
            spec.arm_length_m = args.arm_length

        type_map = {
            'quad': AirframeType.QUADCOPTER_X,
            'hex': AirframeType.HEXACOPTER_X,
            'octo': AirframeType.OCTOCOPTER_X,
            'fixed_wing': AirframeType.FIXED_WING_CONVENTIONAL,
        }
        spec.airframe_type = type_map.get(args.type, AirframeType.QUADCOPTER_X)

        result = onboard_drone(spec)
        print(result.summary())
        return 0 if result.success else 1

    elif args.command == 'validate':
        spec = import_from_json(args.file)
        from onboarding.spec_validator import validate_specification
        result = validate_specification(spec)
        print(result.summary())
        return 0 if result.is_valid else 1

    else:
        # Default to wizard
        result = interactive_onboarding()
        return 0 if result.success else 1


if __name__ == "__main__":
    exit(cli_main())

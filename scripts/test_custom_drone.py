#!/usr/bin/env python3
"""
Test Custom Drone Specifications

This script allows customers (startups, universities, manufacturers)
to test their custom drone specifications in the dogfight simulator.

Usage:
    # Validate a spec file
    python scripts/test_custom_drone.py --spec my_drone.yaml --validate

    # Run combat test
    python scripts/test_custom_drone.py --spec my_drone.yaml --episodes 10

    # Run with visualization
    python scripts/test_custom_drone.py --spec my_drone.yaml --visualize

    # Compare against a specific opponent
    python scripts/test_custom_drone.py --spec my_drone.yaml --opponent x47b

    # Generate performance report
    python scripts/test_custom_drone.py --spec my_drone.yaml --report
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.specs import (
    DroneSpec,
    DroneSpecLoader,
    SpecValidator,
    ValidationResult,
)


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 70)
    print("  DRONE COMBAT SIMULATOR - Custom Drone Testing")
    print("  For startups, universities, and manufacturers")
    print("=" * 70)
    print()


def print_spec_summary(spec: DroneSpec):
    """Print drone specification summary."""
    print(spec.summary())


def print_validation_result(result: ValidationResult):
    """Print validation result with formatting."""
    print("\n" + "=" * 50)
    print("VALIDATION RESULT")
    print("=" * 50)
    print(str(result))


def print_aero_estimates(loader: DroneSpecLoader, spec: DroneSpec):
    """Print estimated aerodynamic parameters."""
    aero = loader.estimate_aero(spec)

    print("\n" + "=" * 50)
    print("ESTIMATED AERODYNAMICS")
    print("=" * 50)
    print(f"""
Lift:
  CL_0 (zero-alpha):     {aero.CL_0:.3f}
  CL_alpha (slope):      {aero.CL_alpha:.3f} /rad
  CL_max:                {aero.CL_max:.3f}
  Stall angle:           {np.degrees(aero.alpha_stall_rad):.1f}°

Drag:
  CD_0 (parasite):       {aero.CD_0:.4f}
  Oswald efficiency:     {aero.oswald_efficiency:.3f}

Stability Derivatives:
  Cm_alpha (pitch):      {aero.Cm_alpha:.3f} /rad {"(stable)" if aero.Cm_alpha < 0 else "(unstable)"}
  Cn_beta (yaw):         {aero.Cn_beta:.3f} /rad {"(stable)" if aero.Cn_beta > 0 else "(unstable)"}
  Cl_beta (dihedral):    {aero.Cl_beta:.3f} /rad

Damping:
  Cm_q (pitch damp):     {aero.Cm_q:.2f}
  Cn_r (yaw damp):       {aero.Cn_r:.3f}
  Cl_p (roll damp):      {aero.Cl_p:.3f}

Control Effectiveness:
  Cm_de (elevator):      {aero.Cm_de:.3f}
  Cl_da (aileron):       {aero.Cl_da:.3f}
  Cn_dr (rudder):        {aero.Cn_dr:.3f}

Inertia (estimated):
  Ixx (roll):            {aero.Ixx:.1f} kg·m²
  Iyy (pitch):           {aero.Iyy:.1f} kg·m²
  Izz (yaw):             {aero.Izz:.1f} kg·m²
""")


def run_combat_test(
    spec: DroneSpec,
    loader: DroneSpecLoader,
    episodes: int = 10,
    opponent: str = "random",
    visualize: bool = False,
    verbose: bool = True,
):
    """
    Run combat simulation with custom drone.

    Args:
        spec: Customer drone specification
        loader: DroneSpecLoader instance
        episodes: Number of episodes to run
        opponent: Opponent type or "random"
        visualize: Whether to show visualization
        verbose: Print episode details
    """
    try:
        from simulation.environments.combat import DogfightEnv, DogfightConfig
    except ImportError as e:
        print(f"Error importing combat environment: {e}")
        print("Make sure you have all dependencies installed.")
        return None

    print("\n" + "=" * 50)
    print("COMBAT SIMULATION")
    print("=" * 50)
    print(f"Drone: {spec.name}")
    print(f"Episodes: {episodes}")
    print(f"Opponent: {opponent}")
    print()

    # Create dogfight config from spec
    config_dict = loader.to_dogfight_config(spec)

    # Create config object
    config = DogfightConfig(
        num_red=config_dict["num_red"],
        num_blue=config_dict["num_blue"],
        arena_size=config_dict["arena_size"],
        arena_height_min=config_dict["arena_height_min"],
        arena_height_max=config_dict["arena_height_max"],
        min_speed=config_dict["min_speed"],
        max_speed=config_dict["max_speed"],
        max_g_force=config_dict["max_g_force"],
        weapons_config=config_dict["weapons_config"],
        respawn_enabled=config_dict["respawn_enabled"],
        kills_to_win=config_dict["kills_to_win"],
        max_match_time=config_dict["max_match_time"],
    )

    # Create environment
    env = DogfightEnv(config=config)

    # Apply custom drone physics if the env supports it
    if hasattr(env, 'set_drone_config'):
        env.set_drone_config('red', config_dict.get('red_drone_config', {}))

    # Statistics tracking
    stats = {
        "episodes": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_kills": 0,
        "total_deaths": 0,
        "total_damage_dealt": 0,
        "total_damage_taken": 0,
        "episode_lengths": [],
        "episode_rewards": [],
    }

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_damage_dealt = 0
            episode_damage_taken = 0

            if verbose:
                print(f"\n--- Episode {ep + 1}/{episodes} ---")

            while True:
                # Simple pursuit AI for testing
                action = get_pursuit_action(env)

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Track combat events
                for event in info.get('combat_events', []):
                    if event.get('type') == 'hit':
                        if event.get('attacker') == env.agent_drone_id:
                            episode_damage_dealt += event.get('damage', 0)
                        else:
                            episode_damage_taken += event.get('damage', 0)

                # Visualization delay
                if visualize:
                    time.sleep(1/60)

                if terminated or truncated:
                    break

            # Record stats
            stats["episodes"] += 1
            stats["episode_lengths"].append(episode_steps)
            stats["episode_rewards"].append(episode_reward)
            stats["total_damage_dealt"] += episode_damage_dealt
            stats["total_damage_taken"] += episode_damage_taken

            # Determine outcome
            red_kills = env.red_kills
            blue_kills = env.blue_kills
            stats["total_kills"] += red_kills
            stats["total_deaths"] += blue_kills

            if red_kills > blue_kills:
                stats["wins"] += 1
                outcome = "WIN"
            elif blue_kills > red_kills:
                stats["losses"] += 1
                outcome = "LOSS"
            else:
                stats["draws"] += 1
                outcome = "DRAW"

            if verbose:
                print(f"  Result: {outcome} (Red {red_kills} - {blue_kills} Blue)")
                print(f"  Steps: {episode_steps}, Reward: {episode_reward:.1f}")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    finally:
        env.close()

    return stats


def get_pursuit_action(env):
    """Simple pursuit AI for testing."""
    try:
        agent = env.drones[env.agent_drone_id]
        enemies = [d for d in env.drones.values()
                   if d.team != agent.team and d.is_alive]

        if not enemies:
            return np.array([0, 0, 0.5, 0, 0, 0])

        nearest = min(enemies, key=lambda e: np.linalg.norm(e.position - agent.position))
        to_enemy = nearest.position - agent.position
        distance = np.linalg.norm(to_enemy)
        to_enemy_norm = to_enemy / (distance + 1e-6)

        yaw = agent.orientation[2]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])

        cross = np.cross(heading, to_enemy_norm)
        roll_cmd = np.clip(cross[2] * 3, -1, 1)

        alt_diff = nearest.position[2] - agent.position[2]
        pitch_cmd = np.clip(alt_diff / 200, -1, 1)

        alignment = np.dot(heading, to_enemy_norm)
        fire = 1.0 if distance < 400 and alignment > 0.8 else 0.0

        return np.array([roll_cmd, pitch_cmd, 0.75, 0, fire, 0])

    except Exception:
        return np.array([0, 0, 0.5, 0, 0, 0])


def print_performance_report(spec: DroneSpec, stats: dict):
    """Print performance report."""
    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)
    print(f"""
Drone: {spec.name}
Manufacturer: {spec.manufacturer}

Combat Statistics:
  Episodes:        {stats['episodes']}
  Wins:            {stats['wins']} ({100*stats['wins']/max(stats['episodes'],1):.1f}%)
  Losses:          {stats['losses']} ({100*stats['losses']/max(stats['episodes'],1):.1f}%)
  Draws:           {stats['draws']} ({100*stats['draws']/max(stats['episodes'],1):.1f}%)

  Total Kills:     {stats['total_kills']}
  Total Deaths:    {stats['total_deaths']}
  K/D Ratio:       {stats['total_kills']/max(stats['total_deaths'],1):.2f}

  Damage Dealt:    {stats['total_damage_dealt']:.0f}
  Damage Taken:    {stats['total_damage_taken']:.0f}
  Damage Ratio:    {stats['total_damage_dealt']/max(stats['total_damage_taken'],1):.2f}

Episode Statistics:
  Avg Length:      {np.mean(stats['episode_lengths']):.0f} steps
  Avg Reward:      {np.mean(stats['episode_rewards']):.1f}
  Max Reward:      {np.max(stats['episode_rewards']):.1f}
  Min Reward:      {np.min(stats['episode_rewards']):.1f}
""")

    # Performance assessment
    win_rate = stats['wins'] / max(stats['episodes'], 1)
    kd_ratio = stats['total_kills'] / max(stats['total_deaths'], 1)

    print("Assessment:")
    if win_rate >= 0.7 and kd_ratio >= 2.0:
        print("  ★★★★★ EXCELLENT - Your drone dominates in combat!")
    elif win_rate >= 0.5 and kd_ratio >= 1.0:
        print("  ★★★★☆ GOOD - Your drone is competitive in combat.")
    elif win_rate >= 0.3:
        print("  ★★★☆☆ AVERAGE - Your drone needs some improvements.")
    elif win_rate >= 0.1:
        print("  ★★☆☆☆ BELOW AVERAGE - Consider adjusting specs.")
    else:
        print("  ★☆☆☆☆ POOR - Major design changes recommended.")

    print()


def list_templates():
    """List available templates."""
    template_dir = Path(__file__).parent.parent / "simulation" / "specs" / "templates"

    print("\nAvailable Templates:")
    print("-" * 40)

    if template_dir.exists():
        for template in sorted(template_dir.glob("*.yaml")):
            print(f"  - {template.name}")
    else:
        print("  (No templates found)")

    print()
    print(f"Templates are located in: {template_dir}")
    print("Copy a template and fill in your drone's specifications.")


def main():
    parser = argparse.ArgumentParser(
        description='Test custom drone specifications in the combat simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_custom_drone.py --spec my_drone.yaml --validate
  python scripts/test_custom_drone.py --spec my_drone.yaml --episodes 20
  python scripts/test_custom_drone.py --spec my_drone.yaml --report
  python scripts/test_custom_drone.py --list-templates
        """
    )

    parser.add_argument('--spec', type=str, help='Path to drone spec YAML file')
    parser.add_argument('--validate', action='store_true', help='Only validate the spec (no simulation)')
    parser.add_argument('--aero', action='store_true', help='Show estimated aerodynamic parameters')
    parser.add_argument('--episodes', type=int, default=10, help='Number of combat episodes')
    parser.add_argument('--opponent', type=str, default='random', help='Opponent type')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    parser.add_argument('--list-templates', action='store_true', help='List available templates')

    args = parser.parse_args()

    print_banner()

    # List templates
    if args.list_templates:
        list_templates()
        return

    # Require spec file for other operations
    if not args.spec:
        parser.print_help()
        print("\nError: --spec is required (or use --list-templates)")
        return

    # Check if spec file exists
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_path}")
        print("\nUse --list-templates to see available templates.")
        return

    # Load spec
    loader = DroneSpecLoader()

    print(f"Loading specification: {spec_path}")
    try:
        spec = loader.load_from_yaml(spec_path)
    except Exception as e:
        print(f"Error loading spec: {e}")
        return

    # Print summary
    print_spec_summary(spec)

    # Validate
    result = loader.validate(spec)
    print_validation_result(result)

    if not result.is_valid:
        print("\nSpec validation failed. Please fix the errors above.")
        return

    # Show aero estimates
    if args.aero or args.validate:
        print_aero_estimates(loader, spec)

    # Stop here if only validating
    if args.validate:
        print("\nValidation complete. Your spec is ready for simulation!")
        return

    # Run combat test
    stats = run_combat_test(
        spec=spec,
        loader=loader,
        episodes=args.episodes,
        opponent=args.opponent,
        visualize=args.visualize,
        verbose=not args.quiet,
    )

    # Generate report
    if stats and (args.report or args.episodes > 1):
        print_performance_report(spec, stats)


if __name__ == '__main__':
    main()

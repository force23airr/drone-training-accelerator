"""Test carrier landing environment - X-47B approach and arrested landing."""
import numpy as np
import time
from simulation.platforms.platform_configs import get_platform_config
from simulation.environments.combat import CarrierOpsEnv

# Load X-47B config (carrier-capable)
config = get_platform_config('x47b_ucav')

print("=" * 60)
print("X-47B CARRIER LANDING SIMULATION")
print("=" * 60)
print()
print("Mission: Execute precision approach and arrested landing")
print("Target: Catch the 3-wire (wire markers are RED)")
print("Challenge: Deck is moving with the waves!")
print()
print("Visual Guide:")
print("  - GRAY structure = Aircraft Carrier (Nimitz-class)")
print("  - YELLOW stripe = Landing centerline")
print("  - RED bars = Arresting wires (1-4)")
print("  - BLUE = Ocean")
print("  - Your X-47B approaches from behind the carrier")
print()

# Create carrier ops environment
env = CarrierOpsEnv(
    config,
    render_mode='human',
    approach_case='case_i',  # Day VFR conditions
    sea_state=2,             # Light seas (less motion for clarity)
    enable_deck_motion=True
)

# Start on approach (not catapult)
obs, info = env.reset(options={'start_phase': 'approach'})

print(f"Starting phase: {info['phase']}")
print(f"Hook deployed: {info['hook_deployed']}")
print(f"Carrier position: {info['carrier_position']}")
print()
print("Flying approach - watch the glideslope and lineup!")
print("Press Ctrl+C to stop")
print()

try:
    # PID-like control state
    gs_integral = 0.0
    lineup_integral = 0.0
    last_gs_error = 0.0
    last_lineup_error = 0.0

    for i in range(3000):
        # Better approach control with PID-like gains
        gs_error = info.get('glideslope_error', 0)
        lineup_error = info.get('lineup_error', 0)

        # Accumulate integral (with anti-windup)
        gs_integral = np.clip(gs_integral + gs_error * 0.02, -50, 50)
        lineup_integral = np.clip(lineup_integral + lineup_error * 0.02, -50, 50)

        # Derivative
        gs_deriv = (gs_error - last_gs_error) / 0.02
        lineup_deriv = (lineup_error - last_lineup_error) / 0.02

        # PID control for pitch (glideslope)
        # Positive gs_error = too high, need to pitch down (negative pitch)
        # Negative gs_error = too low, need to pitch up (positive pitch)
        Kp_pitch = 0.002
        Ki_pitch = 0.0001
        Kd_pitch = 0.001
        pitch_cmd = -(Kp_pitch * gs_error + Ki_pitch * gs_integral + Kd_pitch * gs_deriv)
        pitch_cmd = np.clip(pitch_cmd, -0.3, 0.3)

        # PID control for roll (lineup)
        Kp_roll = 0.003
        Ki_roll = 0.0001
        Kd_roll = 0.001
        roll_cmd = -(Kp_roll * lineup_error + Ki_roll * lineup_integral + Kd_roll * lineup_deriv)
        roll_cmd = np.clip(roll_cmd, -0.3, 0.3)

        # Throttle: maintain approach speed (~70 m/s)
        airspeed = np.linalg.norm(env._velocity)
        target_speed = 70.0
        speed_error = target_speed - airspeed
        throttle = 0.5 + 0.01 * speed_error
        throttle = np.clip(throttle, 0.3, 0.8)

        last_gs_error = gs_error
        last_lineup_error = lineup_error

        action = np.array([roll_cmd, pitch_cmd, 0, throttle, 1.0], dtype=np.float32)
        # action[4] = 1.0 means hook deployed

        try:
            obs, reward, done, trunc, info = env.step(action)
        except Exception as e:
            print(f"Error: {e}")
            break

        time.sleep(0.02)

        if i % 30 == 0:
            phase = info.get('phase', 'UNKNOWN')
            gs = info.get('glideslope_error', 0)
            lu = info.get('lineup_error', 0)
            wire = info.get('wire_caught', 0)
            bolters = info.get('bolter_count', 0)

            # Calculate distance to carrier
            carrier_pos = info.get('carrier_position', np.array([10000, 0, 0]))
            dist_to_carrier = np.linalg.norm(env._position[:2] - carrier_pos[:2])

            alt = env._position[2]
            spd = np.linalg.norm(env._velocity)
            status = f"Phase: {phase:10s} | Dist: {dist_to_carrier:4.0f}m | Alt: {alt:5.1f}m | Spd: {spd:4.1f}m/s | GS: {gs:+5.1f}m"
            if wire > 0:
                status += f" | WIRE {wire}!"
            if bolters > 0:
                status += f" | Bolters: {bolters}"
            print(status)

        if done:
            wire = info.get('wire_caught', 0)
            if wire > 0:
                print()
                print("=" * 40)
                print(f"TRAPPED! Caught wire {wire}!")
                if wire == 3:
                    print("PERFECT - 3-wire! (OK pass)")
                elif wire == 2 or wire == 4:
                    print("Good catch!")
                else:
                    print("Wire 1 - too close to the ramp!")
                print("=" * 40)
            else:
                print("Mission ended - crash or bolter")
            break

except KeyboardInterrupt:
    print('\nStopped by user')

env.close()
print('Done!')

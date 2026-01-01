"""Visual demo of carrier - fly around the ship to see it."""
import numpy as np
import time
from simulation.platforms.platform_configs import get_platform_config
from simulation.environments.combat import CarrierOpsEnv

config = get_platform_config('x47b_ucav')

print("=" * 60)
print("CARRIER FLYBY DEMO")
print("=" * 60)
print()
print("Watch the X-47B fly around the aircraft carrier!")
print()

env = CarrierOpsEnv(
    config,
    render_mode='human',
    approach_case='case_i',
    sea_state=2,
    enable_deck_motion=True
)

obs, info = env.reset(options={'start_phase': 'approach'})

# Override position to start right next to carrier for visual demo
carrier_pos = env._carrier.position.copy()
env._position = carrier_pos + np.array([-500.0, 200.0, 80.0])  # Beside carrier
env._velocity = np.array([50.0, 0.0, 0.0])  # Flying toward carrier
env._attitude = np.array([0.0, 0.0, 0.0])

print(f"Carrier at: {carrier_pos}")
print(f"Aircraft at: {env._position}")
print()
print("Flying around the carrier - watch the ship!")
print("Press Ctrl+C to stop")
print()

try:
    for i in range(2000):
        # Fly in a circle around the carrier
        t = i * 0.02

        # Circular path around carrier
        radius = 400.0
        angular_speed = 0.3  # rad/s
        angle = t * angular_speed

        # Target position on circle around carrier
        target_x = carrier_pos[0] + radius * np.cos(angle)
        target_y = carrier_pos[1] + radius * np.sin(angle)
        target_z = 60.0  # Fly at 60m

        # Simple position control
        pos_error = np.array([target_x, target_y, target_z]) - env._position

        # Desired heading (tangent to circle, pointing forward)
        desired_heading = angle + np.pi/2
        heading_error = desired_heading - env._attitude[2]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Control commands
        roll_cmd = np.clip(0.3 * heading_error, -0.5, 0.5)
        pitch_cmd = np.clip(-0.01 * pos_error[2], -0.3, 0.3)
        throttle = 0.6

        action = np.array([roll_cmd, pitch_cmd, 0, throttle, 0], dtype=np.float32)

        obs, reward, done, trunc, info = env.step(action)
        time.sleep(0.02)

        if i % 50 == 0:
            dist = np.linalg.norm(env._position[:2] - carrier_pos[:2])
            alt = env._position[2]
            print(f"Step {i:4d} | Dist to carrier: {dist:5.0f}m | Alt: {alt:5.1f}m | Circling...")

        if done:
            print("Episode ended, restarting...")
            obs, info = env.reset(options={'start_phase': 'approach'})
            env._position = carrier_pos + np.array([-500.0, 200.0, 80.0])
            env._velocity = np.array([50.0, 0.0, 0.0])

except KeyboardInterrupt:
    print('\nStopped by user')

env.close()
print('Done!')

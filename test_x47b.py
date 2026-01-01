import numpy as np
import time
from simulation.platforms.platform_configs import get_platform_config
from simulation.environments.combat import BaseFixedWingEnv

config = get_platform_config('x47b_ucav')
print("Creating environment...")
env = BaseFixedWingEnv(config, render_mode='human')
print("Resetting...")
obs, info = env.reset()

print('X-47B UCAV Loaded!')
print(f'Observation shape: {obs.shape}')
print(f'Initial position: {env._position}')
print(f'Initial velocity: {env._velocity}')
print(f'Initial attitude: {env._attitude}')
print()
print('Running simulation - Press Ctrl+C to stop')
print()

try:
    for i in range(5000):
        # Vary controls
        t = i * 0.02
        roll_cmd = 0.3 * np.sin(t * 0.5)
        pitch_cmd = -0.1 + 0.05 * np.sin(t * 0.3)
        throttle = 0.8

        action = np.array([roll_cmd, pitch_cmd, 0, throttle, 0], dtype=np.float32)

        try:
            obs, reward, done, trunc, info = env.step(action)
        except Exception as e:
            print(f"ERROR in step: {e}")
            import traceback
            traceback.print_exc()
            break

        time.sleep(0.02)

        if i % 25 == 0:
            pos = env._position
            vel = env._velocity
            att = env._attitude
            print(f'Step {i:4d} | Pos: ({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:6.1f}) | Vel: {np.linalg.norm(vel):5.1f}m/s | Roll: {np.degrees(att[0]):5.1f}°')

        if done:
            print('Episode ended, resetting...')
            obs, info = env.reset()

except KeyboardInterrupt:
    print('\nStopped by user')

env.close()
print('Done!')

"""
Test PX4 SITL Integration.

This script demonstrates connecting your simulation to PX4 SITL.

Prerequisites:
1. Install PX4 Autopilot:
   git clone https://github.com/PX4/PX4-Autopilot.git --recursive
   cd PX4-Autopilot
   make px4_sitl none_iris

2. Or use Docker:
   docker run --rm -it --network host px4io/px4-dev-simulation-focal:latest \
       bash -c "cd /home/user/PX4-Autopilot && make px4_sitl none_iris"

3. Run this script in another terminal:
   python test_px4_sitl.py

The script will:
- Connect to PX4 via MAVLink
- Send simulated sensor data
- Receive actuator commands from PX4
- Display the control loop in action
"""

import time
import numpy as np
from simulation.integration import (
    PX4SITLBridge,
    PX4SITLConfig,
    SensorData,
    GPSData,
    MAVType,
)

print("=" * 60)
print("PX4 SITL INTEGRATION TEST")
print("=" * 60)
print()
print("Make sure PX4 SITL is running:")
print("  cd PX4-Autopilot && make px4_sitl none_iris")
print()
print("Or with Docker:")
print("  docker run --rm -it --network host px4io/px4-dev-simulation-focal")
print()

# Configure for quadcopter
config = PX4SITLConfig(
    vehicle_type=MAVType.QUADROTOR,
    sitl_port_in=14560,
    sitl_port_out=14560,
)

# Create bridge
bridge = PX4SITLBridge(config)

# Connect
if not bridge.connect():
    print("Failed to connect!")
    exit(1)

print("Waiting for PX4...")
print("(If nothing happens, make sure PX4 SITL is running)")
print()

# Simulated state
position = np.array([0.0, 0.0, 0.0])  # Start on ground
velocity = np.zeros(3)
attitude = np.zeros(3)  # roll, pitch, yaw

# Home position (Zurich - PX4 default)
HOME_LAT = 47.397742
HOME_LON = 8.545594
HOME_ALT = 488.0

try:
    for i in range(10000):
        # Create sensor data
        sensor_data = SensorData(
            accel=np.array([0.0, 0.0, -9.81]),  # Gravity
            gyro=np.zeros(3),
            mag=np.array([0.2, 0.0, 0.4]),
            abs_pressure=101325.0,
            pressure_alt=position[2],
            temperature=25.0,
        )

        # GPS data
        gps_data = GPSData(
            lat=(HOME_LAT + position[1] / 111000) * 1e7,
            lon=(HOME_LON + position[0] / 111000) * 1e7,
            alt=(HOME_ALT + position[2]) * 1000,
            vel_n=velocity[1] * 100,
            vel_e=velocity[0] * 100,
            vel_d=-velocity[2] * 100,
        )

        # Send to PX4
        bridge.send_sensors(sensor_data, gps_data, dt=0.004)

        # Get actuator commands from PX4
        actuators = bridge.get_actuator_controls()

        # Simple physics response (just for demo)
        # Real integration would use your full physics engine
        motor_avg = np.mean(actuators.controls[:4])
        if motor_avg > 0.5:
            velocity[2] += 0.01  # Climb
        else:
            velocity[2] -= 0.01  # Descend
        velocity[2] = np.clip(velocity[2], -5, 5)
        position += velocity * 0.004

        # Keep on ground
        if position[2] < 0:
            position[2] = 0
            velocity[2] = 0

        # Status update
        if i % 250 == 0:  # Every second at 250Hz
            print(f"Step {i:5d} | "
                  f"PX4: {'Connected' if bridge.px4_connected else 'Waiting...'} | "
                  f"Alt: {position[2]:5.1f}m | "
                  f"Motors: [{actuators.controls[0]:.2f}, {actuators.controls[1]:.2f}, "
                  f"{actuators.controls[2]:.2f}, {actuators.controls[3]:.2f}]")

        time.sleep(0.004)  # 250 Hz

except KeyboardInterrupt:
    print("\nStopped by user")

bridge.disconnect()
print("Done!")
print()
print("=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("""
1. With PX4 connected, open QGroundControl to see telemetry
2. Arm and fly using QGC or MAVLink commands
3. Integrate with your trained RL policy:

   from simulation.integration import PX4SITLEnv
   from simulation.environments.combat import BaseFixedWingEnv

   env = BaseFixedWingEnv(config)
   px4_env = PX4SITLEnv(env)  # Wrap with PX4

   obs = px4_env.reset()
   while True:
       action = your_policy(obs)  # Your trained policy
       obs, reward, done, info = px4_env.step(action)
       # PX4 receives sensor data, provides actuator commands
""")

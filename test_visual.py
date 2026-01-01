"""Simple test to verify visual movement works."""
import numpy as np
import time
import pybullet as p
import pybullet_data

# Connect to PyBullet with GUI
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load ground
p.loadURDF("plane.urdf")

# Create a simple aircraft shape
fuselage = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 0.5, 0.3])
fuselage_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 0.5, 0.3], rgbaColor=[0.3, 0.3, 0.4, 1])

wing = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 9, 0.1])
wing_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 9, 0.1], rgbaColor=[0.4, 0.4, 0.5, 1])

aircraft = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=fuselage,
    baseVisualShapeIndex=fuselage_vis,
    basePosition=[0, 0, 50],
    linkMasses=[0],
    linkCollisionShapeIndices=[wing],
    linkVisualShapeIndices=[wing_vis],
    linkPositions=[[0, 0, 0]],
    linkOrientations=[[0, 0, 0, 1]],
    linkInertialFramePositions=[[0, 0, 0]],
    linkInertialFrameOrientations=[[0, 0, 0, 1]],
    linkParentIndices=[0],
    linkJointTypes=[p.JOINT_FIXED],
    linkJointAxis=[[0, 0, 1]]
)

print("Aircraft created! Watch it fly...")
print("Press Ctrl+C to stop")

# Simulate flight
pos = np.array([0.0, 0.0, 50.0])
vel = np.array([50.0, 0.0, 0.0])  # 50 m/s forward
yaw = 0.0

try:
    for i in range(3000):
        # Simple circular flight path
        t = i * 0.02
        turn_rate = 0.3  # rad/s
        yaw += turn_rate * 0.02

        # Update velocity direction
        vel = np.array([50 * np.cos(yaw), 50 * np.sin(yaw), 2 * np.sin(t)])

        # Update position
        pos += vel * 0.02

        # Keep altitude reasonable
        pos[2] = max(20, min(200, pos[2]))

        # Update visual
        quat = p.getQuaternionFromEuler([0.2 * np.sin(t*2), 0, yaw])
        p.resetBasePositionAndOrientation(aircraft, pos.tolist(), quat)

        # Camera follows
        p.resetDebugVisualizerCamera(
            cameraDistance=100,
            cameraYaw=np.degrees(yaw) + 180,
            cameraPitch=-20,
            cameraTargetPosition=pos.tolist()
        )

        time.sleep(0.02)

        if i % 50 == 0:
            print(f"Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) | Yaw: {np.degrees(yaw):.0f}°")

except KeyboardInterrupt:
    print("\nStopped")

p.disconnect()
print("Done!")

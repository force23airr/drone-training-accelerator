"""
PX4 SITL Environment Wrapper.

Wraps any Gymnasium drone environment to communicate with PX4 SITL.
This enables testing trained policies with the real PX4 autopilot.

Usage:
    from simulation.environments import BaseDroneEnv
    from simulation.integration import PX4SITLEnv

    # Wrap your environment
    base_env = BaseDroneEnv(...)
    env = PX4SITLEnv(base_env)

    # Now step() sends sensor data to PX4 and receives actuator commands
    obs, info = env.reset()
    while True:
        obs, reward, done, trunc, info = env.step(action)
        # action can be ignored - PX4 controls the vehicle
"""

import time
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .px4_sitl_bridge import (
    PX4SITLBridge,
    PX4SITLConfig,
    SensorData,
    GPSData,
    ActuatorControls,
    MAVType,
    euler_to_quaternion,
    enu_to_ned,
)


@dataclass
class PX4SITLEnvConfig:
    """Configuration for PX4 SITL environment wrapper."""
    # PX4 connection
    px4_config: PX4SITLConfig = None

    # Home position (PX4 default is Zurich)
    home_lat: float = 47.397742
    home_lon: float = 8.545594
    home_alt: float = 488.0  # meters MSL

    # Control mode
    use_px4_control: bool = True  # If True, ignore action and use PX4 commands
    passthrough_action: bool = False  # If True, pass action directly to PX4

    # Timing
    real_time: bool = True  # Run at real-time speed
    timeout_sec: float = 30.0  # Connection timeout

    def __post_init__(self):
        if self.px4_config is None:
            self.px4_config = PX4SITLConfig()


class PX4SITLEnv(gym.Wrapper):
    """
    Gymnasium wrapper that connects any drone environment to PX4 SITL.

    This wrapper:
    1. Extracts state from the inner environment
    2. Converts to sensor messages (IMU, GPS, baro)
    3. Sends to PX4 via MAVLink
    4. Receives actuator commands from PX4
    5. Applies commands to inner environment (or returns them)

    The inner environment's physics runs as normal, but actuator commands
    come from the real PX4 flight stack instead of an RL policy.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[PX4SITLEnvConfig] = None,
    ):
        super().__init__(env)

        self.config = config or PX4SITLEnvConfig()
        self.bridge = PX4SITLBridge(self.config.px4_config)

        # State tracking
        self._connected = False
        self._last_actuators = ActuatorControls()
        self._step_count = 0
        self._real_time_start = 0.0
        self._sim_time = 0.0

        # Sensor computation
        self._last_velocity = np.zeros(3)
        self._last_time = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and establish PX4 connection."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Connect to PX4 if not connected
        if not self._connected:
            print("Connecting to PX4 SITL...")
            if self.bridge.connect():
                self._connected = True
                self._wait_for_px4()
            else:
                print("Warning: Could not connect to PX4 SITL")
                print("Running in standalone mode (no PX4)")

        self._step_count = 0
        self._real_time_start = time.time()
        self._sim_time = 0.0
        self._last_velocity = np.zeros(3)
        self._last_time = time.time()

        # Send initial state to PX4
        if self._connected:
            self._send_state_to_px4()

        info['px4_connected'] = self._connected

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with PX4 in the loop.

        If use_px4_control is True, the action is ignored and
        actuator commands from PX4 are used instead.
        """
        self._step_count += 1
        dt = 1.0 / getattr(self.env, 'control_hz', 50.0)
        self._sim_time += dt

        # Get actuator commands from PX4
        if self._connected and self.config.use_px4_control:
            self._last_actuators = self.bridge.get_actuator_controls()
            action = self._actuators_to_action(self._last_actuators)

        # Step the inner environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Send updated state to PX4
        if self._connected:
            self._send_state_to_px4()

        # Real-time pacing
        if self.config.real_time:
            self._pace_real_time()

        # Add PX4 info
        info['px4_connected'] = self._connected
        info['px4_actuators'] = self._last_actuators.controls.copy()

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up PX4 connection."""
        if self._connected:
            self.bridge.disconnect()
            self._connected = False
        super().close()

    def _wait_for_px4(self, timeout: float = None):
        """Wait for PX4 to connect."""
        timeout = timeout or self.config.timeout_sec
        start = time.time()

        print("Waiting for PX4 heartbeat...")
        while time.time() - start < timeout:
            if self.bridge.px4_connected:
                print("PX4 connected!")
                return True
            # Send heartbeats while waiting
            self.bridge.send_heartbeat()
            time.sleep(0.1)

        print(f"Timeout waiting for PX4 (waited {timeout}s)")
        return False

    def _send_state_to_px4(self):
        """Extract state from environment and send to PX4."""
        # Get state from inner environment
        position = self._get_env_position()
        velocity = self._get_env_velocity()
        attitude = self._get_env_attitude()
        angular_vel = self._get_env_angular_velocity()

        # Compute acceleration
        current_time = time.time()
        dt = max(current_time - self._last_time, 0.001)
        accel = (velocity - self._last_velocity) / dt
        self._last_velocity = velocity.copy()
        self._last_time = current_time

        # Convert to NED frame (PX4 uses NED)
        position_ned = enu_to_ned(position)
        velocity_ned = enu_to_ned(velocity)
        accel_ned = enu_to_ned(accel)

        # Add gravity to accelerometer (measures specific force)
        accel_body = self._world_to_body(accel_ned + np.array([0, 0, 9.81]), attitude)

        # Angular velocity in body frame
        gyro = angular_vel  # Assuming already in body frame

        # Compute barometric altitude
        pressure_alt = position[2]  # Assuming Z is up
        abs_pressure = 101325.0 * (1 - 2.25577e-5 * pressure_alt) ** 5.25588

        # Create sensor data
        sensor_data = SensorData(
            accel=accel_body,
            gyro=gyro,
            mag=np.array([0.2, 0.0, 0.4]),  # Simplified mag
            abs_pressure=abs_pressure,
            diff_pressure=0.5 * 1.225 * np.linalg.norm(velocity) ** 2,  # Dynamic pressure
            pressure_alt=pressure_alt,
            temperature=25.0,
        )

        # Convert local position to GPS
        lat, lon = self._local_to_gps(position[0], position[1])
        alt = self.config.home_alt + position[2]

        gps_data = GPSData(
            lat=lat * 1e7,
            lon=lon * 1e7,
            alt=alt * 1000,
            vel_n=velocity_ned[0] * 100,
            vel_e=velocity_ned[1] * 100,
            vel_d=velocity_ned[2] * 100,
        )

        # Send to PX4
        dt = 1.0 / getattr(self.env, 'control_hz', 50.0)
        self.bridge.send_sensors(sensor_data, gps_data, dt=dt)

        # Also send full state
        quat = euler_to_quaternion(attitude[0], attitude[1], attitude[2])
        self.bridge.send_hil_state_quaternion(
            attitude_q=quat,
            angular_vel=angular_vel,
            position=position_ned,
            velocity=velocity_ned,
            accel=accel_ned,
            lat=lat,
            lon=lon,
            alt=alt,
        )

    def _get_env_position(self) -> np.ndarray:
        """Get position from inner environment."""
        if hasattr(self.env, '_position'):
            return self.env._position.copy()
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_position'):
            return self.env.unwrapped._position.copy()
        return np.zeros(3)

    def _get_env_velocity(self) -> np.ndarray:
        """Get velocity from inner environment."""
        if hasattr(self.env, '_velocity'):
            return self.env._velocity.copy()
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_velocity'):
            return self.env.unwrapped._velocity.copy()
        return np.zeros(3)

    def _get_env_attitude(self) -> np.ndarray:
        """Get attitude (roll, pitch, yaw) from inner environment."""
        if hasattr(self.env, '_attitude'):
            return self.env._attitude.copy()
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_attitude'):
            return self.env.unwrapped._attitude.copy()
        return np.zeros(3)

    def _get_env_angular_velocity(self) -> np.ndarray:
        """Get angular velocity from inner environment."""
        if hasattr(self.env, '_angular_velocity'):
            return self.env._angular_velocity.copy()
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_angular_velocity'):
            return self.env.unwrapped._angular_velocity.copy()
        return np.zeros(3)

    def _local_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local XY (meters) to GPS coordinates."""
        # Simple flat-earth approximation
        lat = self.config.home_lat + y / 111000.0
        lon = self.config.home_lon + x / (111000.0 * np.cos(np.radians(self.config.home_lat)))
        return lat, lon

    def _world_to_body(self, vec: np.ndarray, attitude: np.ndarray) -> np.ndarray:
        """Transform vector from world frame to body frame."""
        roll, pitch, yaw = attitude

        # Rotation matrices
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Combined rotation (ZYX order)
        R = np.array([
            [cp*cy, cp*sy, -sp],
            [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
            [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]
        ])

        return R @ vec

    def _actuators_to_action(self, actuators: ActuatorControls) -> np.ndarray:
        """Convert PX4 actuator controls to environment action."""
        controls = actuators.controls

        # PX4 actuator mapping depends on vehicle type
        # For multicopter: controls[0:4] are motor outputs
        # For fixed-wing: controls[0:4] are [aileron, elevator, rudder, throttle]

        if hasattr(self.env, 'action_space'):
            action_dim = self.env.action_space.shape[0]
        else:
            action_dim = 4

        # Map to action space
        if action_dim == 4:
            # Quadcopter: 4 motor commands
            action = controls[:4].copy()
        elif action_dim == 5:
            # Fixed-wing: [aileron, elevator, rudder, throttle, flaps]
            action = np.array([
                controls[0],  # aileron (-1 to 1)
                controls[1],  # elevator (-1 to 1)
                controls[2],  # rudder (-1 to 1)
                controls[3],  # throttle (0 to 1)
                0.0,          # flaps
            ])
        else:
            action = controls[:action_dim].copy()

        return action.astype(np.float32)

    def _pace_real_time(self):
        """Pace simulation to run at real-time speed."""
        elapsed_real = time.time() - self._real_time_start
        if self._sim_time > elapsed_real:
            time.sleep(self._sim_time - elapsed_real)


class PX4OffboardController:
    """
    Helper class for sending offboard control commands to PX4.

    Use this to command PX4 from your trained policy while
    PX4 handles low-level control.

    Example:
        controller = PX4OffboardController(bridge)
        controller.arm()
        controller.set_position_target(x=10, y=0, z=-5)  # NED
    """

    def __init__(self, bridge: PX4SITLBridge):
        self.bridge = bridge

    def arm(self):
        """Send arm command to PX4."""
        # MAV_CMD_COMPONENT_ARM_DISARM
        self._send_command_long(
            command=400,
            param1=1.0,  # 1 = arm
        )

    def disarm(self):
        """Send disarm command to PX4."""
        self._send_command_long(
            command=400,
            param1=0.0,  # 0 = disarm
        )

    def takeoff(self, altitude: float = 10.0):
        """Send takeoff command."""
        # MAV_CMD_NAV_TAKEOFF
        self._send_command_long(
            command=22,
            param7=altitude,  # Altitude
        )

    def land(self):
        """Send land command."""
        # MAV_CMD_NAV_LAND
        self._send_command_long(command=21)

    def set_mode_offboard(self):
        """Set PX4 to offboard mode for external control."""
        # MAV_CMD_DO_SET_MODE
        self._send_command_long(
            command=176,
            param1=1.0,    # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            param2=6.0,    # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        )

    def _send_command_long(
        self,
        command: int,
        param1: float = 0,
        param2: float = 0,
        param3: float = 0,
        param4: float = 0,
        param5: float = 0,
        param6: float = 0,
        param7: float = 0,
    ):
        """Send COMMAND_LONG message."""
        import struct
        payload = struct.pack(
            '<7fHBBB',
            param1, param2, param3, param4, param5, param6, param7,
            command,
            self.bridge.config.system_id,
            self.bridge.config.component_id,
            0,  # Confirmation
        )
        self.bridge.mavlink.send_message(76, payload)  # COMMAND_LONG

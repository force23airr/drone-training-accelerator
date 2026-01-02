"""
PX4 SITL Bridge - Connect simulation environments to PX4 autopilot.

This module enables Hardware-In-The-Loop (HIL) / Software-In-The-Loop (SITL)
simulation by bridging our physics simulation to the real PX4 flight stack.

Communication Protocol:
- MAVLink v2 over UDP
- Simulator sends: HIL_SENSOR, HIL_GPS, HIL_STATE_QUATERNION
- PX4 sends: HIL_ACTUATOR_CONTROLS

Usage:
    bridge = PX4SITLBridge()
    bridge.connect()

    while running:
        # Send sensor data from your sim
        bridge.send_hil_sensor(accel, gyro, mag, baro)
        bridge.send_hil_gps(lat, lon, alt, vel)

        # Receive actuator commands from PX4
        controls = bridge.receive_actuator_controls()

    bridge.disconnect()
"""

import time
import struct
import socket
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Callable
from enum import IntEnum
import numpy as np


class MAVLinkMessageID(IntEnum):
    """MAVLink message IDs we use for HIL."""
    HEARTBEAT = 0
    HIL_ACTUATOR_CONTROLS = 93
    HIL_SENSOR = 107
    HIL_GPS = 113
    HIL_STATE_QUATERNION = 115
    COMMAND_LONG = 76
    STATUSTEXT = 253


class MAVComponent(IntEnum):
    """MAVLink component IDs."""
    AUTOPILOT = 1
    SIMULATOR = 51


class MAVType(IntEnum):
    """MAVLink vehicle types."""
    QUADROTOR = 2
    FIXED_WING = 1
    VTOL = 19


class MAVAutopilot(IntEnum):
    """MAVLink autopilot types."""
    PX4 = 12
    ARDUPILOT = 3


class MAVState(IntEnum):
    """MAVLink system states."""
    UNINIT = 0
    BOOT = 1
    CALIBRATING = 2
    STANDBY = 3
    ACTIVE = 4
    CRITICAL = 5
    EMERGENCY = 6


@dataclass
class PX4SITLConfig:
    """Configuration for PX4 SITL connection."""
    # Network settings
    sitl_host: str = "127.0.0.1"
    sitl_port_in: int = 14560   # Receive from PX4
    sitl_port_out: int = 14560  # Send to PX4 (same for UDP)

    # Vehicle configuration
    vehicle_type: MAVType = MAVType.QUADROTOR
    system_id: int = 1
    component_id: int = MAVComponent.SIMULATOR

    # Timing
    sensor_rate_hz: float = 250.0  # HIL_SENSOR rate
    gps_rate_hz: float = 10.0      # HIL_GPS rate
    heartbeat_rate_hz: float = 1.0

    # Physics
    gravity: float = 9.81

    # Noise (for realistic sensor simulation)
    accel_noise_std: float = 0.1      # m/s^2
    gyro_noise_std: float = 0.01      # rad/s
    mag_noise_std: float = 0.01       # gauss
    baro_noise_std: float = 0.5       # meters
    gps_noise_std: float = 0.5        # meters


@dataclass
class SensorData:
    """Sensor data to send to PX4."""
    # IMU
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s^2
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s

    # Magnetometer
    mag: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.0, 0.4]))  # gauss

    # Barometer
    abs_pressure: float = 101325.0  # Pa
    diff_pressure: float = 0.0      # Pa (for airspeed)
    pressure_alt: float = 0.0       # meters
    temperature: float = 25.0       # Celsius


@dataclass
class GPSData:
    """GPS data to send to PX4."""
    lat: float = 47.397742 * 1e7    # degrees * 1e7 (PX4 default home)
    lon: float = 8.545594 * 1e7     # degrees * 1e7
    alt: float = 488.0 * 1000       # mm above MSL
    vel_n: float = 0.0              # cm/s north
    vel_e: float = 0.0              # cm/s east
    vel_d: float = 0.0              # cm/s down
    hdop: float = 0.8               # Horizontal dilution
    vdop: float = 1.0               # Vertical dilution
    satellites: int = 12
    fix_type: int = 3               # 3D fix


@dataclass
class ActuatorControls:
    """Actuator controls received from PX4."""
    controls: np.ndarray = field(default_factory=lambda: np.zeros(16))
    mode: int = 0
    flags: int = 0
    timestamp_us: int = 0


class MAVLinkConnection:
    """
    Low-level MAVLink connection handler.

    Handles UDP socket communication and MAVLink message encoding/decoding.
    """

    def __init__(self, config: PX4SITLConfig):
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.sequence = 0

        # Message stats
        self.messages_sent = 0
        self.messages_received = 0
        self.last_heartbeat = 0.0

    def connect(self) -> bool:
        """Establish UDP connection."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("0.0.0.0", self.config.sitl_port_in))
            self.socket.setblocking(False)
            self.connected = True
            print(f"MAVLink: Listening on UDP port {self.config.sitl_port_in}")
            return True
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            return False

    def disconnect(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False

    def send_message(self, msg_id: int, payload: bytes):
        """Send a MAVLink v2 message."""
        if not self.connected or not self.socket:
            return

        # MAVLink v2 header
        header = struct.pack(
            '<BBBBBBBHB',
            0xFD,                          # Magic (MAVLink v2)
            len(payload),                   # Payload length
            0,                              # Incompatibility flags
            0,                              # Compatibility flags
            self.sequence & 0xFF,           # Sequence
            self.config.system_id,          # System ID
            self.config.component_id,       # Component ID
            msg_id & 0xFFFF,               # Message ID (low 16 bits)
            (msg_id >> 16) & 0xFF,         # Message ID (high 8 bits)
        )

        # Calculate CRC (simplified - real impl needs CRC_EXTRA per message)
        message = header + payload
        crc = self._calculate_crc(message[1:], msg_id)

        packet = message + struct.pack('<H', crc)

        try:
            self.socket.sendto(
                packet,
                (self.config.sitl_host, self.config.sitl_port_out)
            )
            self.sequence += 1
            self.messages_sent += 1
        except Exception as e:
            pass  # Non-blocking, may fail occasionally

    def receive_message(self) -> Optional[Tuple[int, bytes]]:
        """Receive a MAVLink message (non-blocking)."""
        if not self.connected or not self.socket:
            return None

        try:
            data, addr = self.socket.recvfrom(1024)
            if len(data) < 12:  # Minimum MAVLink v2 size
                return None

            # Parse header
            magic = data[0]
            if magic == 0xFD:  # MAVLink v2
                payload_len = data[1]
                msg_id = struct.unpack('<H', data[7:9])[0] | (data[9] << 16)
                payload = data[10:10+payload_len]
                self.messages_received += 1
                return (msg_id, payload)
            elif magic == 0xFE:  # MAVLink v1
                payload_len = data[1]
                msg_id = data[5]
                payload = data[6:6+payload_len]
                self.messages_received += 1
                return (msg_id, payload)

        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            pass

        return None

    def _calculate_crc(self, data: bytes, msg_id: int) -> int:
        """Calculate MAVLink CRC (X.25 checksum)."""
        # CRC_EXTRA values for messages we use
        crc_extra = {
            MAVLinkMessageID.HEARTBEAT: 50,
            MAVLinkMessageID.HIL_SENSOR: 108,
            MAVLinkMessageID.HIL_GPS: 124,
            MAVLinkMessageID.HIL_STATE_QUATERNION: 4,
            MAVLinkMessageID.HIL_ACTUATOR_CONTROLS: 47,
        }

        crc = 0xFFFF
        for byte in data:
            tmp = byte ^ (crc & 0xFF)
            tmp ^= (tmp << 4) & 0xFF
            crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
            crc &= 0xFFFF

        # Add CRC_EXTRA
        extra = crc_extra.get(msg_id, 0)
        tmp = extra ^ (crc & 0xFF)
        tmp ^= (tmp << 4) & 0xFF
        crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)

        return crc & 0xFFFF


class PX4SITLBridge:
    """
    High-level bridge between simulation and PX4 SITL.

    This class manages the bidirectional communication:
    - Sends sensor data (IMU, GPS, baro) to PX4
    - Receives actuator commands from PX4
    - Handles heartbeat and connection monitoring

    Example:
        bridge = PX4SITLBridge(PX4SITLConfig(vehicle_type=MAVType.FIXED_WING))
        bridge.connect()

        # In your simulation loop:
        bridge.send_sensors(sensor_data, gps_data, state)
        actuators = bridge.get_actuator_controls()

        bridge.disconnect()
    """

    def __init__(self, config: Optional[PX4SITLConfig] = None):
        self.config = config or PX4SITLConfig()
        self.mavlink = MAVLinkConnection(self.config)

        # State
        self.connected = False
        self.armed = False
        self.px4_connected = False

        # Latest actuator controls from PX4
        self._actuator_controls = ActuatorControls()
        self._actuator_lock = threading.Lock()

        # Timing
        self._last_sensor_time = 0.0
        self._last_gps_time = 0.0
        self._last_heartbeat_time = 0.0
        self._sim_time_us = 0

        # Receive thread
        self._receive_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks
        self._on_actuator_update: Optional[Callable[[ActuatorControls], None]] = None

    def connect(self) -> bool:
        """Connect to PX4 SITL."""
        if not self.mavlink.connect():
            return False

        self.connected = True
        self._running = True

        # Start receive thread
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

        print("PX4 SITL Bridge: Connected, waiting for PX4...")
        print("  Start PX4 with: make px4_sitl none_iris")
        print("  Or use: PX4_SIM_HOST_ADDR=127.0.0.1 ./build/px4_sitl_default/bin/px4")

        return True

    def disconnect(self):
        """Disconnect from PX4."""
        self._running = False
        if self._receive_thread:
            self._receive_thread.join(timeout=1.0)
        self.mavlink.disconnect()
        self.connected = False
        print("PX4 SITL Bridge: Disconnected")

    def send_heartbeat(self):
        """Send heartbeat to PX4."""
        payload = struct.pack(
            '<IBBBBB',
            0,                              # Custom mode
            self.config.vehicle_type,       # Type
            MAVAutopilot.PX4,              # Autopilot
            0,                              # Base mode
            MAVState.ACTIVE,               # System status
            3,                              # MAVLink version
        )
        self.mavlink.send_message(MAVLinkMessageID.HEARTBEAT, payload)

    def send_hil_sensor(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        mag: np.ndarray,
        abs_pressure: float,
        diff_pressure: float,
        pressure_alt: float,
        temperature: float,
        timestamp_us: Optional[int] = None,
    ):
        """
        Send HIL_SENSOR message to PX4.

        Args:
            accel: Acceleration [x, y, z] in m/s^2 (NED frame)
            gyro: Angular velocity [x, y, z] in rad/s (NED frame)
            mag: Magnetic field [x, y, z] in gauss
            abs_pressure: Absolute pressure in hPa
            diff_pressure: Differential pressure in hPa (for airspeed)
            pressure_alt: Altitude from pressure in meters
            temperature: Temperature in Celsius
            timestamp_us: Timestamp in microseconds (auto-generated if None)
        """
        if timestamp_us is None:
            timestamp_us = self._sim_time_us

        # Add noise for realism
        accel = accel + np.random.normal(0, self.config.accel_noise_std, 3)
        gyro = gyro + np.random.normal(0, self.config.gyro_noise_std, 3)
        mag = mag + np.random.normal(0, self.config.mag_noise_std, 3)
        pressure_alt += np.random.normal(0, self.config.baro_noise_std)

        # HIL_SENSOR fields_updated bitmask
        fields_updated = (
            (1 << 0) |  # xacc
            (1 << 1) |  # yacc
            (1 << 2) |  # zacc
            (1 << 3) |  # xgyro
            (1 << 4) |  # ygyro
            (1 << 5) |  # zgyro
            (1 << 6) |  # xmag
            (1 << 7) |  # ymag
            (1 << 8) |  # zmag
            (1 << 9) |  # abs_pressure
            (1 << 10) | # diff_pressure
            (1 << 11) | # pressure_alt
            (1 << 12)   # temperature
        )

        payload = struct.pack(
            '<Q3f3f3ffffIB',
            timestamp_us,
            accel[0], accel[1], accel[2],
            gyro[0], gyro[1], gyro[2],
            mag[0], mag[1], mag[2],
            abs_pressure / 100.0,  # Convert Pa to hPa
            diff_pressure / 100.0,
            pressure_alt,
            temperature,
            fields_updated,
            0,  # ID (sensor instance)
        )
        self.mavlink.send_message(MAVLinkMessageID.HIL_SENSOR, payload)

    def send_hil_gps(
        self,
        lat: float,
        lon: float,
        alt: float,
        vel_n: float,
        vel_e: float,
        vel_d: float,
        timestamp_us: Optional[int] = None,
        fix_type: int = 3,
        satellites: int = 12,
    ):
        """
        Send HIL_GPS message to PX4.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters (MSL)
            vel_n: North velocity in m/s
            vel_e: East velocity in m/s
            vel_d: Down velocity in m/s
            timestamp_us: Timestamp in microseconds
            fix_type: GPS fix type (3 = 3D fix)
            satellites: Number of satellites visible
        """
        if timestamp_us is None:
            timestamp_us = self._sim_time_us

        # Add GPS noise
        lat += np.random.normal(0, self.config.gps_noise_std / 111000)  # ~111km per degree
        lon += np.random.normal(0, self.config.gps_noise_std / 111000)
        alt += np.random.normal(0, self.config.gps_noise_std)

        # Calculate ground speed and course
        vel_ground = np.sqrt(vel_n**2 + vel_e**2)
        course = np.degrees(np.arctan2(vel_e, vel_n)) % 360

        payload = struct.pack(
            '<QBiiihhHHHhHBB',
            timestamp_us,
            fix_type,
            int(lat * 1e7),          # lat (degE7)
            int(lon * 1e7),          # lon (degE7)
            int(alt * 1000),         # alt (mm)
            int(100),                # eph (cm) - horizontal accuracy
            int(150),                # epv (cm) - vertical accuracy
            int(vel_ground * 100),   # vel (cm/s)
            int(vel_n * 100),        # vn (cm/s)
            int(vel_e * 100),        # ve (cm/s)
            int(vel_d * 100),        # vd (cm/s)
            int(course * 100),       # cog (cdeg)
            satellites,
            0,                       # id
        )
        self.mavlink.send_message(MAVLinkMessageID.HIL_GPS, payload)

    def send_hil_state_quaternion(
        self,
        attitude_q: np.ndarray,
        angular_vel: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        accel: np.ndarray,
        lat: float,
        lon: float,
        alt: float,
        timestamp_us: Optional[int] = None,
    ):
        """
        Send HIL_STATE_QUATERNION message for complete state update.

        Args:
            attitude_q: Quaternion [w, x, y, z]
            angular_vel: Angular velocity [roll_rate, pitch_rate, yaw_rate] rad/s
            position: Position [x, y, z] in meters (local NED)
            velocity: Velocity [vx, vy, vz] in m/s (NED)
            accel: Acceleration [ax, ay, az] in m/s^2 (NED)
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude MSL in meters
            timestamp_us: Timestamp in microseconds
        """
        if timestamp_us is None:
            timestamp_us = self._sim_time_us

        # Airspeed from velocity (simplified)
        true_airspeed = np.linalg.norm(velocity)
        indicated_airspeed = true_airspeed * 0.95  # Simplified

        payload = struct.pack(
            '<Q4f3f3fiiihhhhhh',
            timestamp_us,
            attitude_q[0], attitude_q[1], attitude_q[2], attitude_q[3],
            angular_vel[0], angular_vel[1], angular_vel[2],
            int(lat * 1e7),
            int(lon * 1e7),
            int(alt * 1000),
            int(velocity[0] * 100),
            int(velocity[1] * 100),
            int(velocity[2] * 100),
            int(indicated_airspeed * 100),
            int(true_airspeed * 100),
            int(accel[0] * 1000),
            int(accel[1] * 1000),
            int(accel[2] * 1000),
        )
        self.mavlink.send_message(MAVLinkMessageID.HIL_STATE_QUATERNION, payload)

    def send_sensors(
        self,
        sensor_data: SensorData,
        gps_data: GPSData,
        state: Optional[Dict[str, Any]] = None,
        dt: float = 0.004,  # 250 Hz default
    ):
        """
        Convenience method to send all sensor data at appropriate rates.

        Args:
            sensor_data: IMU and barometer data
            gps_data: GPS data
            state: Optional full state for HIL_STATE_QUATERNION
            dt: Time step in seconds
        """
        current_time = time.time()
        self._sim_time_us += int(dt * 1e6)

        # Send heartbeat at 1 Hz
        if current_time - self._last_heartbeat_time >= 1.0 / self.config.heartbeat_rate_hz:
            self.send_heartbeat()
            self._last_heartbeat_time = current_time

        # Send HIL_SENSOR at sensor rate
        sensor_period = 1.0 / self.config.sensor_rate_hz
        if current_time - self._last_sensor_time >= sensor_period:
            self.send_hil_sensor(
                accel=sensor_data.accel,
                gyro=sensor_data.gyro,
                mag=sensor_data.mag,
                abs_pressure=sensor_data.abs_pressure,
                diff_pressure=sensor_data.diff_pressure,
                pressure_alt=sensor_data.pressure_alt,
                temperature=sensor_data.temperature,
            )
            self._last_sensor_time = current_time

        # Send HIL_GPS at GPS rate
        gps_period = 1.0 / self.config.gps_rate_hz
        if current_time - self._last_gps_time >= gps_period:
            self.send_hil_gps(
                lat=gps_data.lat / 1e7,
                lon=gps_data.lon / 1e7,
                alt=gps_data.alt / 1000,
                vel_n=gps_data.vel_n / 100,
                vel_e=gps_data.vel_e / 100,
                vel_d=gps_data.vel_d / 100,
            )
            self._last_gps_time = current_time

    def get_actuator_controls(self) -> ActuatorControls:
        """Get latest actuator controls from PX4."""
        with self._actuator_lock:
            return ActuatorControls(
                controls=self._actuator_controls.controls.copy(),
                mode=self._actuator_controls.mode,
                flags=self._actuator_controls.flags,
                timestamp_us=self._actuator_controls.timestamp_us,
            )

    def set_actuator_callback(self, callback: Callable[[ActuatorControls], None]):
        """Set callback for actuator updates."""
        self._on_actuator_update = callback

    def _receive_loop(self):
        """Background thread for receiving MAVLink messages."""
        while self._running:
            result = self.mavlink.receive_message()
            if result:
                msg_id, payload = result
                self._handle_message(msg_id, payload)
            else:
                time.sleep(0.001)  # Small sleep when no data

    def _handle_message(self, msg_id: int, payload: bytes):
        """Handle incoming MAVLink message."""
        if msg_id == MAVLinkMessageID.HEARTBEAT:
            # PX4 is alive
            if not self.px4_connected:
                self.px4_connected = True
                print("PX4 SITL Bridge: PX4 connected!")

        elif msg_id == MAVLinkMessageID.HIL_ACTUATOR_CONTROLS:
            # Parse actuator controls
            if len(payload) >= 81:
                data = struct.unpack('<Q16fBB', payload[:81])
                timestamp_us = data[0]
                controls = np.array(data[1:17])
                mode = data[17]
                flags = data[18]

                with self._actuator_lock:
                    self._actuator_controls.controls = controls
                    self._actuator_controls.mode = mode
                    self._actuator_controls.flags = flags
                    self._actuator_controls.timestamp_us = timestamp_us

                if self._on_actuator_update:
                    self._on_actuator_update(self._actuator_controls)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (rad) to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def ned_to_enu(vec: np.ndarray) -> np.ndarray:
    """Convert NED (North-East-Down) to ENU (East-North-Up)."""
    return np.array([vec[1], vec[0], -vec[2]])


def enu_to_ned(vec: np.ndarray) -> np.ndarray:
    """Convert ENU (East-North-Up) to NED (North-East-Down)."""
    return np.array([vec[1], vec[0], -vec[2]])

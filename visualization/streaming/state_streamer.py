"""
State Streamer for Dogfight Visualization

Publishes simulation state via ZeroMQ for real-time visualization.
Designed to add minimal overhead to the training loop.

Usage:
    streamer = StateStreamer(port=5555)
    streamer.start()

    # In simulation loop:
    streamer.publish_frame(env)

    streamer.stop()
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

from .protocols import (
    FrameState,
    UAVState,
    WeaponState,
    CombatEvent,
    MatchInfo,
    ProtocolSerializer,
    MessageType,
    EventType,
    WeaponType as ProtoWeaponType,
    ReplayWriter,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamerStats:
    """Statistics for the streamer."""
    frames_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0
    avg_frame_time_ms: float = 0.0
    subscribers: int = 0


class StateStreamer:
    """
    Publishes DogfightEnv state via ZeroMQ PUB socket.

    Features:
    - Non-blocking publishing (won't slow training)
    - Automatic frame numbering
    - Optional replay recording
    - Heartbeat for connection monitoring
    """

    def __init__(
        self,
        port: int = 5555,
        bind_address: str = "tcp://*",
        enable_replay: bool = False,
        replay_path: Optional[str] = None,
        high_water_mark: int = 10,
    ):
        """
        Initialize state streamer.

        Args:
            port: ZeroMQ port to publish on
            bind_address: Address to bind to
            enable_replay: Whether to record replay file
            replay_path: Path for replay file (auto-generated if None)
            high_water_mark: ZeroMQ HWM (drop frames if subscriber is slow)
        """
        if not HAS_ZMQ:
            raise ImportError("pyzmq required: pip install pyzmq")

        self.port = port
        self.bind_address = bind_address
        self.enable_replay = enable_replay
        self.replay_path = replay_path
        self.high_water_mark = high_water_mark

        self.serializer = ProtocolSerializer(use_msgpack=True)
        self.stats = StreamerStats()

        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._replay_writer: Optional[ReplayWriter] = None
        self._frame_number = 0
        self._match_info: Optional[MatchInfo] = None
        self._running = False
        self._pending_events: List[CombatEvent] = []

        # Frame timing
        self._frame_times: List[float] = []
        self._last_frame_time = 0.0

    def start(self):
        """Start the streamer."""
        if self._running:
            return

        logger.info(f"Starting StateStreamer on {self.bind_address}:{self.port}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, self.high_water_mark)
        self._socket.bind(f"{self.bind_address}:{self.port}")

        if self._match_info:
            data = self.serializer.serialize(MessageType.MATCH_START, self._match_info.to_dict())
            self._socket.send(data, zmq.NOBLOCK)

        if self.enable_replay:
            if self.replay_path is None:
                self.replay_path = f"replay_{int(time.time())}.dfrp"
            self._replay_writer = ReplayWriter(self.replay_path, self._match_info)
            self._replay_writer.__enter__()
            logger.info(f"Recording replay to {self.replay_path}")

        self._running = True
        self._frame_number = 0

    def stop(self):
        """Stop the streamer."""
        if not self._running:
            return

        logger.info("Stopping StateStreamer")

        if self._replay_writer:
            self._replay_writer.__exit__(None, None, None)
            self._replay_writer = None

        if self._socket:
            self._socket.close()
            self._socket = None

        if self._context:
            self._context.term()
            self._context = None

        self._running = False

    def set_match_info(self, env) -> MatchInfo:
        """
        Set match info from environment.

        Args:
            env: DogfightEnv instance

        Returns:
            MatchInfo object
        """
        config = env.config

        uav_types = {}
        for drone_id, drone in env.drones.items():
            uav_types[drone_id] = "jet_a"  # Default type

        self._match_info = MatchInfo(
            match_id=f"match_{int(time.time())}",
            arena_size=config.arena_size,
            arena_height_min=config.arena_height_min,
            arena_height_max=config.arena_height_max,
            red_count=config.num_red,
            blue_count=config.num_blue,
            kills_to_win=config.kills_to_win,
            max_match_time=config.max_match_time,
            respawn_enabled=config.respawn_enabled,
            uav_types=uav_types,
            start_time=time.time(),
        )

        if self._replay_writer:
            self._replay_writer.match_info = self._match_info

        # Send match start
        if self._socket:
            data = self.serializer.serialize(MessageType.MATCH_START, self._match_info.to_dict())
            self._socket.send(data, zmq.NOBLOCK)

        return self._match_info

    def add_event(self, event: CombatEvent):
        """Add a combat event to be sent with next frame."""
        self._pending_events.append(event)

    def publish_frame(self, env) -> bool:
        """
        Publish current environment state.

        Args:
            env: DogfightEnv instance

        Returns:
            True if published successfully
        """
        if not self._running or not self._socket:
            return False

        if self._match_info is None:
            self.set_match_info(env)

        start_time = time.perf_counter()

        try:
            # Build frame state
            frame = self._build_frame_state(env)

            # Serialize
            data = self.serializer.serialize_frame(frame)

            # Publish (non-blocking)
            self._socket.send(data, zmq.NOBLOCK)

            # Record to replay
            if self._replay_writer:
                self._replay_writer.write_frame(frame)

            # Update stats
            self.stats.frames_sent += 1
            self.stats.bytes_sent += len(data)

            # Track timing
            elapsed = (time.perf_counter() - start_time) * 1000
            self._frame_times.append(elapsed)
            if len(self._frame_times) > 100:
                self._frame_times.pop(0)
            self.stats.avg_frame_time_ms = sum(self._frame_times) / len(self._frame_times)

            # Clear pending events
            self._pending_events.clear()

            self._frame_number += 1
            return True

        except zmq.Again:
            # No subscribers or HWM reached
            return False
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
            self.stats.errors += 1
            return False

    def _build_frame_state(self, env) -> FrameState:
        """Build FrameState from environment."""
        uavs = []
        for drone_id, drone in env.drones.items():
            uav = self._drone_to_uav_state(drone)
            uavs.append(uav)

        # Include arena info in first frame
        arena_size = None
        arena_height_min = None
        arena_height_max = None
        if self._frame_number == 0:
            arena_size = env.config.arena_size
            arena_height_min = env.config.arena_height_min
            arena_height_max = env.config.arena_height_max

        return FrameState(
            timestamp=time.time(),
            match_time=env.match_time,
            frame_number=self._frame_number,
            red_score=env.red_kills,
            blue_score=env.blue_kills,
            uavs=uavs,
            events=self._pending_events.copy(),
            arena_size=arena_size,
            arena_height_min=arena_height_min,
            arena_height_max=arena_height_max,
        )

    def _drone_to_uav_state(self, drone) -> UAVState:
        """Convert CombatDrone to UAVState."""
        # Convert numpy arrays to tuples
        position = tuple(float(x) for x in drone.position)
        velocity = tuple(float(x) for x in drone.velocity)
        orientation = tuple(float(x) for x in drone.orientation)
        angular_velocity = tuple(float(x) for x in drone.angular_velocity)

        # Calculate derived values
        speed = float(np.linalg.norm(drone.velocity))
        altitude = float(drone.position[2])

        # Use actual G-force from physics simulation (realistic, capped at max_g + 1)
        g_force = float(getattr(drone, '_actual_g_force', 1.0))
        g_force = max(1.0, min(g_force, 12.0))  # Cap at 12G absolute max

        # Use tactical AI maneuver name if available, else detect from state
        current_maneuver = getattr(drone, '_current_maneuver', None)
        if not current_maneuver:
            current_maneuver = self._detect_maneuver(drone, speed, g_force)

        # Convert weapons
        weapons = []
        for w in drone.weapons:
            ws = WeaponState(
                weapon_type=self._convert_weapon_type(w.weapon_type),
                ammo=w.ammo,
                max_ammo=w.max_ammo,
                cooldown=w.cooldown,
                lock_progress=w.current_lock / w.lock_time if w.lock_time > 0 else 0,
            )
            weapons.append(ws)

        return UAVState(
            id=drone.drone_id,
            team=drone.team,
            uav_type="jet_a",
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            speed=speed,
            altitude=altitude,
            health=float(drone.health),
            max_health=float(drone.max_health),
            alive=drone.is_alive,
            kills=drone.kills,
            deaths=drone.deaths,
            weapons=weapons,
            locked_target=drone.locked_target,
            locked_by=list(drone.locked_by),
            current_maneuver=current_maneuver,
            g_force=g_force,
        )

    def _convert_weapon_type(self, weapon_type) -> int:
        """Convert DogfightEnv weapon type to protocol type."""
        type_map = {
            "gun": ProtoWeaponType.GUN,
            "missile_ir": ProtoWeaponType.MISSILE_IR,
            "missile_radar": ProtoWeaponType.MISSILE_RADAR,
            "laser": ProtoWeaponType.LASER,
        }
        return type_map.get(weapon_type.value, ProtoWeaponType.GUN)

    def _detect_maneuver(self, drone, speed: float, g_force: float) -> Optional[str]:
        """Detect current maneuver from drone state."""
        roll = drone.orientation[0]
        pitch = drone.orientation[1]
        roll_rate = drone.angular_velocity[0]
        pitch_rate = drone.angular_velocity[1]

        # High-G turn
        if g_force > 4.0:
            return "HIGH-G TURN"

        # Barrel roll
        if abs(roll_rate) > 2.0:
            return "BARREL ROLL"

        # Steep climb
        if pitch > 0.5 and drone.velocity[2] > 50:
            return "STEEP CLIMB"

        # Dive
        if pitch < -0.3 and drone.velocity[2] < -30:
            return "DIVE"

        # Split-S (inverted + pulling down)
        if abs(roll) > 2.5 and pitch_rate < -0.5:
            return "SPLIT-S"

        # Immelmann (climbing + roll)
        if pitch > 0.3 and abs(roll_rate) > 1.0:
            return "IMMELMANN"

        # Break turn
        if g_force > 3.0 and abs(roll) > 1.0:
            return "BREAK TURN"

        # Level flight
        if abs(roll) < 0.2 and abs(pitch) < 0.2:
            return "LEVEL"

        return None

    def send_heartbeat(self):
        """Send heartbeat message."""
        if self._socket:
            try:
                data = self.serializer.serialize(MessageType.HEARTBEAT, {
                    'timestamp': time.time(),
                    'frames_sent': self.stats.frames_sent,
                })
                self._socket.send(data, zmq.NOBLOCK)
            except zmq.Again:
                pass

    def get_stats(self) -> StreamerStats:
        """Get streamer statistics."""
        return self.stats

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class StateReceiver:
    """
    Receives state from StateStreamer via ZeroMQ SUB socket.

    Used by the Panda3D renderer to receive simulation state.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 100,
    ):
        """
        Initialize state receiver.

        Args:
            host: Host to connect to
            port: Port to connect to
            timeout_ms: Receive timeout in milliseconds
        """
        if not HAS_ZMQ:
            raise ImportError("pyzmq required: pip install pyzmq")

        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

        self.serializer = ProtocolSerializer(use_msgpack=True)

        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._running = False

        # State
        self.match_info: Optional[MatchInfo] = None
        self.latest_frame: Optional[FrameState] = None
        self.frames_received = 0
        self.last_receive_time = 0.0

        # Frame buffer for interpolation
        self._frame_buffer: List[FrameState] = []
        self._buffer_size = 3

    def connect(self):
        """Connect to the streamer."""
        if self._running:
            return

        address = f"tcp://{self.host}:{self.port}"
        logger.info(f"Connecting to StateStreamer at {address}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
        self._socket.connect(address)

        self._running = True

    def disconnect(self):
        """Disconnect from the streamer."""
        if not self._running:
            return

        logger.info("Disconnecting from StateStreamer")

        if self._socket:
            self._socket.close()
            self._socket = None

        if self._context:
            self._context.term()
            self._context = None

        self._running = False

    def receive(self, non_blocking: bool = False) -> Optional[FrameState]:
        """
        Receive next frame (non-blocking with timeout).

        Args:
            non_blocking: Use ZeroMQ non-blocking receive

        Returns:
            FrameState if available, None otherwise
        """
        if not self._running or not self._socket:
            return None

        try:
            if non_blocking:
                data = self._socket.recv(flags=zmq.NOBLOCK)
            else:
                data = self._socket.recv()
            msg_type, payload = self.serializer.deserialize(data)

            if msg_type == MessageType.FRAME_STATE:
                frame = FrameState.from_dict(payload)
                self.latest_frame = frame
                self.frames_received += 1
                self.last_receive_time = time.time()

                # Add to buffer
                self._frame_buffer.append(frame)
                if len(self._frame_buffer) > self._buffer_size:
                    self._frame_buffer.pop(0)

                return frame

            elif msg_type == MessageType.MATCH_START:
                self.match_info = MatchInfo.from_dict(payload)
                logger.info(f"Match started: {self.match_info.match_id}")

            elif msg_type == MessageType.MATCH_END:
                logger.info("Match ended")

            elif msg_type == MessageType.HEARTBEAT:
                pass  # Connection alive

        except zmq.Again:
            # Timeout - no data available
            pass
        except Exception as e:
            logger.error(f"Error receiving: {e}")

        return None

    def get_interpolated_frame(self, target_time: float) -> Optional[FrameState]:
        """
        Get interpolated frame for smooth rendering.

        Args:
            target_time: Target timestamp

        Returns:
            Interpolated FrameState
        """
        if len(self._frame_buffer) < 2:
            return self.latest_frame

        # Find frames to interpolate between
        prev_frame = None
        next_frame = None

        for i, frame in enumerate(self._frame_buffer):
            if frame.timestamp >= target_time:
                next_frame = frame
                if i > 0:
                    prev_frame = self._frame_buffer[i - 1]
                break

        if prev_frame is None or next_frame is None:
            return self.latest_frame

        # Calculate interpolation factor
        dt = next_frame.timestamp - prev_frame.timestamp
        if dt <= 0:
            return next_frame

        t = (target_time - prev_frame.timestamp) / dt
        t = max(0.0, min(1.0, t))

        # Interpolate UAV positions
        prev_map = {uav.id: uav for uav in prev_frame.uavs}
        next_map = {uav.id: uav for uav in next_frame.uavs}

        interpolated_uavs = []
        for uav in next_frame.uavs:
            prev_uav = prev_map.get(uav.id)
            next_uav = next_map[uav.id]
            if prev_uav is None:
                interpolated_uavs.append(next_uav)
            else:
                interp_uav = self._interpolate_uav(prev_uav, next_uav, t)
                interpolated_uavs.append(interp_uav)

        return FrameState(
            timestamp=target_time,
            match_time=prev_frame.match_time + t * (next_frame.match_time - prev_frame.match_time),
            frame_number=next_frame.frame_number,
            red_score=next_frame.red_score,
            blue_score=next_frame.blue_score,
            uavs=interpolated_uavs,
            events=next_frame.events,
            arena_size=next_frame.arena_size,
            arena_height_min=next_frame.arena_height_min,
            arena_height_max=next_frame.arena_height_max,
        )

    def _interpolate_uav(self, prev: UAVState, next_uav: UAVState, t: float) -> UAVState:
        """Interpolate between two UAV states."""

        def lerp_tuple(a, b, t):
            return tuple(a[i] + t * (b[i] - a[i]) for i in range(len(a)))

        def lerp_angle_tuple(a, b, t):
            # Handle angle wrapping for orientation
            result = []
            for i in range(len(a)):
                diff = b[i] - a[i]
                # Normalize to [-pi, pi]
                while diff > 3.14159:
                    diff -= 2 * 3.14159
                while diff < -3.14159:
                    diff += 2 * 3.14159
                result.append(a[i] + t * diff)
            return tuple(result)

        return UAVState(
            id=next_uav.id,
            team=next_uav.team,
            uav_type=next_uav.uav_type,
            position=lerp_tuple(prev.position, next_uav.position, t),
            velocity=lerp_tuple(prev.velocity, next_uav.velocity, t),
            orientation=lerp_angle_tuple(prev.orientation, next_uav.orientation, t),
            angular_velocity=lerp_tuple(prev.angular_velocity, next_uav.angular_velocity, t),
            speed=prev.speed + t * (next_uav.speed - prev.speed),
            altitude=prev.altitude + t * (next_uav.altitude - prev.altitude),
            health=prev.health + t * (next_uav.health - prev.health),
            max_health=next_uav.max_health,
            alive=next_uav.alive,
            kills=next_uav.kills,
            deaths=next_uav.deaths,
            weapons=next_uav.weapons,
            locked_target=next_uav.locked_target,
            locked_by=next_uav.locked_by,
            current_maneuver=next_uav.current_maneuver,
            g_force=prev.g_force + t * (next_uav.g_force - prev.g_force),
        )

    @property
    def is_connected(self) -> bool:
        """Check if receiving data recently."""
        return time.time() - self.last_receive_time < 1.0

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

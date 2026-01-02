"""
Streaming Protocols for Dogfight Visualization

Defines data structures for real-time state streaming between
the simulation and visualization renderer.

Supports both JSON (debugging) and MessagePack (performance) serialization.
"""

import json
import os
import struct
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import IntEnum

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


class MessageType(IntEnum):
    """Message types for the streaming protocol."""
    FRAME_STATE = 1
    MATCH_START = 2
    MATCH_END = 3
    REPLAY_HEADER = 4
    HEARTBEAT = 5


class EventType(IntEnum):
    """Combat event types."""
    HIT = 1
    KILL = 2
    RESPAWN = 3
    MISSILE_LAUNCH = 4
    MISSILE_LOCK = 5
    OUT_OF_BOUNDS = 6
    CRASH = 7


class WeaponType(IntEnum):
    """Weapon types matching DogfightEnv."""
    GUN = 1
    MISSILE_IR = 2
    MISSILE_RADAR = 3
    LASER = 4


@dataclass
class WeaponState:
    """State of a single weapon."""
    weapon_type: int
    ammo: int
    max_ammo: int
    cooldown: float
    lock_progress: float  # 0-1, for missiles

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaponState":
        return cls(**data)


@dataclass
class UAVState:
    """Complete state of a single UAV for visualization."""
    id: int
    team: int  # 0=red, 1=blue
    uav_type: str  # "jet_a", "jet_b", "drone"

    # Kinematics
    position: Tuple[float, float, float]  # x, y, z in meters
    velocity: Tuple[float, float, float]  # vx, vy, vz in m/s
    orientation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    angular_velocity: Tuple[float, float, float]  # p, q, r in rad/s

    # Derived
    speed: float  # m/s
    altitude: float  # meters

    # Combat state
    health: float  # 0-100
    max_health: float
    alive: bool
    kills: int
    deaths: int

    # Weapons
    weapons: List[WeaponState] = field(default_factory=list)

    # Targeting
    locked_target: Optional[int] = None  # ID of target being locked
    locked_by: List[int] = field(default_factory=list)  # IDs locking this UAV

    # Maneuver (for HUD display)
    current_maneuver: Optional[str] = None
    g_force: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['weapons'] = [w.to_dict() if isinstance(w, WeaponState) else w for w in self.weapons]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UAVState":
        weapons = [WeaponState.from_dict(w) if isinstance(w, dict) else w
                   for w in data.get('weapons', [])]
        data['weapons'] = weapons
        return cls(**data)


@dataclass
class CombatEvent:
    """A combat event (hit, kill, etc.)."""
    event_type: int  # EventType value
    timestamp: float
    attacker_id: int
    target_id: int
    weapon: int  # WeaponType value
    damage: float
    position: Tuple[float, float, float]  # Where it happened

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CombatEvent":
        return cls(**data)


@dataclass
class FrameState:
    """Complete state for a single frame of visualization."""
    # Timing
    timestamp: float  # Unix timestamp
    match_time: float  # Time since match start
    frame_number: int

    # Scores
    red_score: int
    blue_score: int

    # All UAVs
    uavs: List[UAVState]

    # Events this frame
    events: List[CombatEvent] = field(default_factory=list)

    # Arena info (sent once, cached by renderer)
    arena_size: Optional[float] = None
    arena_height_min: Optional[float] = None
    arena_height_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'timestamp': self.timestamp,
            'match_time': self.match_time,
            'frame_number': self.frame_number,
            'red_score': self.red_score,
            'blue_score': self.blue_score,
            'uavs': [u.to_dict() for u in self.uavs],
            'events': [e.to_dict() for e in self.events],
        }
        if self.arena_size is not None:
            d['arena_size'] = self.arena_size
            d['arena_height_min'] = self.arena_height_min
            d['arena_height_max'] = self.arena_height_max
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameState":
        uavs = [UAVState.from_dict(u) for u in data.get('uavs', [])]
        events = [CombatEvent.from_dict(e) for e in data.get('events', [])]
        return cls(
            timestamp=data['timestamp'],
            match_time=data['match_time'],
            frame_number=data['frame_number'],
            red_score=data['red_score'],
            blue_score=data['blue_score'],
            uavs=uavs,
            events=events,
            arena_size=data.get('arena_size'),
            arena_height_min=data.get('arena_height_min'),
            arena_height_max=data.get('arena_height_max'),
        )


@dataclass
class MatchInfo:
    """Match metadata sent at start."""
    match_id: str
    arena_size: float
    arena_height_min: float
    arena_height_max: float
    red_count: int
    blue_count: int
    kills_to_win: int
    max_match_time: float
    respawn_enabled: bool
    uav_types: Dict[int, str]  # drone_id -> type
    start_time: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchInfo":
        return cls(**data)


@dataclass
class ReplayHeader:
    """Header for replay files."""
    version: int = 1
    match_info: Optional[MatchInfo] = None
    total_frames: int = 0
    duration_seconds: float = 0.0
    recorded_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.match_info:
            d['match_info'] = self.match_info.to_dict()
        return d


class ProtocolSerializer:
    """
    Serializer for streaming protocol.

    Supports JSON (human-readable) and MessagePack (compact/fast).
    """

    def __init__(self, use_msgpack: bool = True):
        """
        Initialize serializer.

        Args:
            use_msgpack: Use MessagePack if available (faster, smaller)
        """
        self.use_msgpack = use_msgpack and HAS_MSGPACK

    def serialize(self, msg_type: MessageType, data: Any) -> bytes:
        """
        Serialize a message for transmission.

        Format: [1 byte type][4 bytes length][payload]
        """
        if isinstance(data, (FrameState, MatchInfo, CombatEvent, UAVState)):
            payload_dict = data.to_dict()
        else:
            payload_dict = data

        if self.use_msgpack:
            payload = msgpack.packb(payload_dict, use_bin_type=True)
        else:
            payload = json.dumps(payload_dict).encode('utf-8')

        # Header: type (1 byte) + length (4 bytes)
        header = struct.pack('>BI', msg_type, len(payload))
        return header + payload

    def deserialize(self, data: bytes) -> Tuple[MessageType, Dict[str, Any]]:
        """
        Deserialize a message.

        Returns:
            Tuple of (message_type, payload_dict)
        """
        if len(data) < 5:
            raise ValueError("Message too short")

        msg_type, length = struct.unpack('>BI', data[:5])
        payload_bytes = data[5:5 + length]

        if self.use_msgpack:
            payload = msgpack.unpackb(payload_bytes, raw=False)
        else:
            payload = json.loads(payload_bytes.decode('utf-8'))

        return MessageType(msg_type), payload

    def serialize_frame(self, frame: FrameState) -> bytes:
        """Convenience method for frame serialization."""
        return self.serialize(MessageType.FRAME_STATE, frame)

    def deserialize_frame(self, data: bytes) -> FrameState:
        """Convenience method for frame deserialization."""
        msg_type, payload = self.deserialize(data)
        if msg_type != MessageType.FRAME_STATE:
            raise ValueError(f"Expected FRAME_STATE, got {msg_type}")
        return FrameState.from_dict(payload)


class ReplayWriter:
    """Writes replay files with length-delimited frames."""

    def __init__(self, filepath: str, match_info: Optional[MatchInfo] = None):
        """
        Initialize replay writer.

        Args:
            filepath: Path to replay file
            match_info: Match metadata
        """
        self.filepath = filepath
        self.match_info = match_info
        self.serializer = ProtocolSerializer(use_msgpack=True)
        self.frame_count = 0
        self.start_time = time.time()
        self._recorded_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._last_frame_match_time = 0.0
        self._file = None

    def __enter__(self):
        self._file = open(self.filepath, 'wb')
        # Write header placeholder (will update on close)
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _write_header(self):
        """Write replay header."""
        header = ReplayHeader(
            version=1,
            match_info=self.match_info,
            total_frames=0,
            duration_seconds=0.0,
            recorded_at=self._recorded_at,
        )
        header_bytes = self.serializer.serialize(MessageType.REPLAY_HEADER, header.to_dict())
        # Write length-prefixed header
        self._file.write(struct.pack('>I', len(header_bytes)))
        self._file.write(header_bytes)

    def write_frame(self, frame: FrameState):
        """Write a frame to the replay file."""
        if self._file is None:
            raise RuntimeError("Writer not opened")

        frame_bytes = self.serializer.serialize_frame(frame)
        # Length-prefixed
        self._file.write(struct.pack('>I', len(frame_bytes)))
        self._file.write(frame_bytes)
        self.frame_count += 1
        self._last_frame_match_time = frame.match_time

    def close(self):
        """Close the replay file and update header."""
        if not self._file:
            return

        self._file.close()
        self._file = None

        header = ReplayHeader(
            version=1,
            match_info=self.match_info,
            total_frames=self.frame_count,
            duration_seconds=self._last_frame_match_time,
            recorded_at=self._recorded_at,
        )
        header_bytes = self.serializer.serialize(MessageType.REPLAY_HEADER, header.to_dict())

        src_path = Path(self.filepath)
        tmp_path = src_path.with_suffix(src_path.suffix + ".tmp")

        with src_path.open('rb') as src:
            length_bytes = src.read(4)
            if len(length_bytes) < 4:
                return
            old_length = struct.unpack('>I', length_bytes)[0]
            src.seek(4 + old_length)

            with tmp_path.open('wb') as dst:
                dst.write(struct.pack('>I', len(header_bytes)))
                dst.write(header_bytes)

                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

        os.replace(tmp_path, src_path)


class ReplayReader:
    """Reads replay files."""

    def __init__(self, filepath: str):
        """
        Initialize replay reader.

        Args:
            filepath: Path to replay file
        """
        self.filepath = filepath
        self.serializer = ProtocolSerializer(use_msgpack=True)
        self.header: Optional[ReplayHeader] = None
        self._file = None
        self._frames_read = 0

    def __enter__(self):
        self._file = open(self.filepath, 'rb')
        self._read_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def _read_header(self):
        """Read replay header."""
        length_bytes = self._file.read(4)
        if len(length_bytes) < 4:
            raise ValueError("Invalid replay file")

        length = struct.unpack('>I', length_bytes)[0]
        header_bytes = self._file.read(length)

        msg_type, payload = self.serializer.deserialize(header_bytes)
        if msg_type != MessageType.REPLAY_HEADER:
            raise ValueError("Expected REPLAY_HEADER")

        self.header = ReplayHeader(**payload)

    def read_frame(self) -> Optional[FrameState]:
        """Read next frame from replay."""
        if self._file is None:
            raise RuntimeError("Reader not opened")

        length_bytes = self._file.read(4)
        if len(length_bytes) < 4:
            return None  # EOF

        length = struct.unpack('>I', length_bytes)[0]
        frame_bytes = self._file.read(length)

        if len(frame_bytes) < length:
            return None  # Truncated

        frame = self.serializer.deserialize_frame(frame_bytes)
        self._frames_read += 1
        return frame

    def __iter__(self):
        """Iterate over all frames."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    @property
    def frames_read(self) -> int:
        return self._frames_read

#!/usr/bin/env python3
"""
Replay Controller

Manages playback of recorded dogfight matches with:
- Play/pause/stop controls
- Speed adjustment (0.25x to 4x)
- Timeline scrubbing
- Frame-by-frame stepping
- Event jumping (jump to kills, etc.)
"""

import time
from enum import Enum, auto
from typing import Optional, List, Callable, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, '.')

from visualization.streaming.protocols import FrameState, ReplayReader, CombatEvent


class PlaybackState(Enum):
    """Playback state."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


class PlaybackSpeed(Enum):
    """Playback speed presets."""
    QUARTER = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    QUADRUPLE = 4.0

    @classmethod
    def from_value(cls, value: float) -> 'PlaybackSpeed':
        """Get closest speed preset."""
        for speed in cls:
            if abs(speed.value - value) < 0.01:
                return speed
        return cls.NORMAL

    def next(self) -> 'PlaybackSpeed':
        """Get next faster speed."""
        speeds = list(PlaybackSpeed)
        idx = speeds.index(self)
        return speeds[min(idx + 1, len(speeds) - 1)]

    def prev(self) -> 'PlaybackSpeed':
        """Get next slower speed."""
        speeds = list(PlaybackSpeed)
        idx = speeds.index(self)
        return speeds[max(idx - 1, 0)]


@dataclass
class ReplayEvent:
    """Annotated event in replay timeline."""
    time: float
    frame_index: int
    event_type: str
    description: str
    attacker_id: Optional[int] = None
    target_id: Optional[int] = None


class ReplayController:
    """
    Controls playback of recorded dogfight replays.

    Features:
    - Load replay files
    - Play/pause/stop controls
    - Speed adjustment
    - Seek to time or frame
    - Jump to events
    - Frame-by-frame stepping
    """

    def __init__(self, replay_path: Optional[str] = None):
        """
        Initialize replay controller.

        Args:
            replay_path: Path to replay file to load
        """
        self.replay_path = replay_path
        self.reader: Optional[ReplayReader] = None

        # Playback state
        self.state = PlaybackState.STOPPED
        self.speed = PlaybackSpeed.NORMAL

        # Frame data
        self.frames: List[FrameState] = []
        self.current_frame_index: int = 0
        self.events: List[ReplayEvent] = []

        # Timing
        self.playback_time: float = 0.0
        self.last_update_time: float = 0.0
        self.duration: float = 0.0

        # Callbacks
        self.on_frame_change: Optional[Callable[[FrameState], None]] = None
        self.on_state_change: Optional[Callable[[PlaybackState], None]] = None

        if replay_path:
            self.load(replay_path)

    def load(self, replay_path: str) -> bool:
        """
        Load replay from file.

        Args:
            replay_path: Path to replay file

        Returns:
            True if loaded successfully
        """
        self.replay_path = replay_path
        self.frames.clear()
        self.events.clear()

        try:
            with ReplayReader(replay_path) as reader:
                # Read all frames into memory
                while True:
                    frame = reader.read_frame()
                    if frame is None:
                        break
                    self.frames.append(frame)

        except Exception as e:
            print(f"Failed to load replay: {e}")
            return False

        if not self.frames:
            print("Replay file contains no frames")
            return False

        # Calculate duration
        self.duration = self.frames[-1].match_time - self.frames[0].match_time

        # Extract events
        self._extract_events()

        # Reset state
        self.current_frame_index = 0
        self.playback_time = self.frames[0].match_time
        self.state = PlaybackState.STOPPED

        print(f"Loaded replay: {len(self.frames)} frames, {self.duration:.1f}s duration")
        print(f"Found {len(self.events)} notable events")

        return True

    def _extract_events(self):
        """Extract notable events from replay."""
        self.events.clear()

        for i, frame in enumerate(self.frames):
            for event in frame.events:
                if event.event_type == 2:  # KILL
                    self.events.append(ReplayEvent(
                        time=frame.match_time,
                        frame_index=i,
                        event_type="kill",
                        description=f"UAV {event.attacker_id} killed UAV {event.target_id}",
                        attacker_id=event.attacker_id,
                        target_id=event.target_id,
                    ))
                elif event.event_type == 3:  # RESPAWN
                    self.events.append(ReplayEvent(
                        time=frame.match_time,
                        frame_index=i,
                        event_type="respawn",
                        description=f"UAV {event.target_id} respawned",
                        target_id=event.target_id,
                    ))

    # =========================================================================
    # Playback Controls
    # =========================================================================

    def play(self):
        """Start or resume playback."""
        if self.state == PlaybackState.STOPPED:
            self.current_frame_index = 0
            self.playback_time = self.frames[0].match_time if self.frames else 0

        self.state = PlaybackState.PLAYING
        self.last_update_time = time.time()

        if self.on_state_change:
            self.on_state_change(self.state)

    def pause(self):
        """Pause playback."""
        self.state = PlaybackState.PAUSED

        if self.on_state_change:
            self.on_state_change(self.state)

    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def stop(self):
        """Stop playback and reset to beginning."""
        self.state = PlaybackState.STOPPED
        self.current_frame_index = 0
        self.playback_time = self.frames[0].match_time if self.frames else 0

        if self.on_state_change:
            self.on_state_change(self.state)

    def set_speed(self, speed: PlaybackSpeed):
        """Set playback speed."""
        self.speed = speed

    def speed_up(self):
        """Increase playback speed."""
        self.speed = self.speed.next()

    def slow_down(self):
        """Decrease playback speed."""
        self.speed = self.speed.prev()

    # =========================================================================
    # Seeking
    # =========================================================================

    def seek_to_time(self, target_time: float):
        """
        Seek to specific time.

        Args:
            target_time: Target match time in seconds
        """
        if not self.frames:
            return

        # Clamp time
        start_time = self.frames[0].match_time
        end_time = self.frames[-1].match_time
        target_time = max(start_time, min(end_time, target_time))

        # Find closest frame
        best_index = 0
        best_diff = float('inf')

        for i, frame in enumerate(self.frames):
            diff = abs(frame.match_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_index = i

        self.current_frame_index = best_index
        self.playback_time = target_time

        self._emit_frame()

    def seek_to_frame(self, frame_index: int):
        """
        Seek to specific frame.

        Args:
            frame_index: Target frame index
        """
        if not self.frames:
            return

        frame_index = max(0, min(len(self.frames) - 1, frame_index))
        self.current_frame_index = frame_index
        self.playback_time = self.frames[frame_index].match_time

        self._emit_frame()

    def seek_to_progress(self, progress: float):
        """
        Seek to progress (0.0 to 1.0).

        Args:
            progress: Progress through replay (0=start, 1=end)
        """
        if not self.frames:
            return

        progress = max(0.0, min(1.0, progress))
        start_time = self.frames[0].match_time
        target_time = start_time + progress * self.duration

        self.seek_to_time(target_time)

    def step_forward(self, frames: int = 1):
        """
        Step forward by frames.

        Args:
            frames: Number of frames to step
        """
        self.seek_to_frame(self.current_frame_index + frames)

    def step_backward(self, frames: int = 1):
        """
        Step backward by frames.

        Args:
            frames: Number of frames to step
        """
        self.seek_to_frame(self.current_frame_index - frames)

    def jump_to_next_event(self, event_type: Optional[str] = None):
        """
        Jump to next event.

        Args:
            event_type: Optional filter for event type ("kill", "respawn")
        """
        current_time = self.playback_time

        for event in self.events:
            if event.time > current_time + 0.1:  # Small buffer
                if event_type is None or event.event_type == event_type:
                    self.seek_to_time(event.time)
                    return

    def jump_to_prev_event(self, event_type: Optional[str] = None):
        """
        Jump to previous event.

        Args:
            event_type: Optional filter for event type
        """
        current_time = self.playback_time

        for event in reversed(self.events):
            if event.time < current_time - 0.1:
                if event_type is None or event.event_type == event_type:
                    self.seek_to_time(event.time)
                    return

    # =========================================================================
    # Update Loop
    # =========================================================================

    def update(self) -> Optional[FrameState]:
        """
        Update playback and return current frame.

        Should be called every frame.

        Returns:
            Current frame state or None if no frames
        """
        if not self.frames:
            return None

        if self.state == PlaybackState.PLAYING:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

            # Advance playback time
            self.playback_time += dt * self.speed.value

            # Find frame for current time
            while (self.current_frame_index < len(self.frames) - 1 and
                   self.frames[self.current_frame_index + 1].match_time <= self.playback_time):
                self.current_frame_index += 1

            # Check for end
            if self.current_frame_index >= len(self.frames) - 1:
                self.pause()

        return self.get_current_frame()

    def get_current_frame(self) -> Optional[FrameState]:
        """Get current frame."""
        if not self.frames or self.current_frame_index >= len(self.frames):
            return None
        return self.frames[self.current_frame_index]

    def _emit_frame(self):
        """Emit current frame to callback."""
        if self.on_frame_change:
            frame = self.get_current_frame()
            if frame:
                self.on_frame_change(frame)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        if not self.frames or self.duration <= 0:
            return 0.0

        start_time = self.frames[0].match_time
        elapsed = self.playback_time - start_time
        return max(0.0, min(1.0, elapsed / self.duration))

    @property
    def current_time(self) -> float:
        """Get current playback time."""
        return self.playback_time

    @property
    def total_frames(self) -> int:
        """Get total number of frames."""
        return len(self.frames)

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self.state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self.state == PlaybackState.PAUSED

    def get_events_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> List[ReplayEvent]:
        """Get events within time range."""
        return [e for e in self.events if start_time <= e.time <= end_time]

    def get_frame_at_time(self, target_time: float) -> Optional[FrameState]:
        """Get frame closest to time."""
        if not self.frames:
            return None

        best_frame = self.frames[0]
        best_diff = float('inf')

        for frame in self.frames:
            diff = abs(frame.match_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame

        return best_frame

    def format_time(self, seconds: float) -> str:
        """Format time as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

"""
Visualization Module for Drone Training Accelerator

Professional-grade 3D visualization for dogfight combat training.

Quick Start:
    # In training script:
    from visualization import StateStreamer
    streamer = StateStreamer(port=5555)
    streamer.start()
    # ... in step loop:
    streamer.publish_frame(env)

    # In separate terminal:
    python -m visualization.renderer.dogfight_viewer --connect localhost:5555

    # For replay:
    python -m visualization.renderer.dogfight_viewer --replay path/to/replay.dfrp

Components:
- streaming: ZeroMQ state streaming and replay recording
- renderer: Panda3D 3D visualization with effects
- cameras: Chase, spectator, tactical, auto camera systems
- hud: Telemetry overlays, kill feed, scoreboard
- replay: Playback controls and timeline
- effects: Explosions, missiles, trails, damage indicators
"""

from .streaming import (
    StateStreamer,
    StateReceiver,
    FrameState,
    UAVState,
    CombatEvent,
    MatchInfo,
)

from .replay import (
    ReplayController,
    PlaybackState,
    PlaybackSpeed,
    TimelineWidget,
)

__all__ = [
    # Streaming
    "StateStreamer",
    "StateReceiver",
    "FrameState",
    "UAVState",
    "CombatEvent",
    "MatchInfo",
    # Replay
    "ReplayController",
    "PlaybackState",
    "PlaybackSpeed",
    "TimelineWidget",
]

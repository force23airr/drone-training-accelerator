#!/usr/bin/env python3
"""
Timeline Widget for Replay Visualization

Visual timeline UI with:
- Scrubber bar with drag support
- Event markers (kills, respawns)
- Time display
- Playback controls
- Speed indicator
"""

from typing import Optional, List, Callable, Tuple, Any
from dataclasses import dataclass

try:
    from direct.gui.DirectGui import (
        DirectFrame,
        DirectButton,
        DirectSlider,
        DirectLabel,
    )
    from direct.gui.OnscreenText import OnscreenText
    from panda3d.core import (
        Point3, Vec3, Vec4, LColor,
        NodePath, GeomNode, Geom, GeomVertexFormat, GeomVertexData,
        GeomVertexWriter, GeomTriangles, GeomLines,
        CardMaker, TextNode,
        TransparencyAttrib,
        MouseWatcher,
    )
    HAS_PANDA3D = True
except ImportError:
    HAS_PANDA3D = False
    # Stub classes for when Panda3D is not available
    Vec4 = tuple
    NodePath = Any

from .replay_controller import ReplayController, PlaybackState, PlaybackSpeed


@dataclass
class EventMarker:
    """Visual marker for timeline event."""
    time: float
    progress: float
    event_type: str
    color: Any  # Vec4 when Panda3D available
    description: str


class TimelineWidget:
    """
    Visual timeline widget for replay playback.

    Features:
    - Horizontal timeline bar
    - Draggable scrubber
    - Event markers
    - Play/pause button
    - Speed controls
    - Time display
    """

    def __init__(
        self,
        parent: NodePath,
        controller: ReplayController,
        width: float = 1.6,
        height: float = 0.1,
        position: Tuple[float, float] = (0, -0.85),
    ):
        """
        Initialize timeline widget.

        Args:
            parent: Parent node
            controller: Replay controller
            width: Widget width (in aspect2d units)
            height: Widget height
            position: Widget position (x, y)
        """
        if not HAS_PANDA3D:
            raise ImportError("panda3d required")

        self.parent = parent
        self.controller = controller
        self.width = width
        self.height = height
        self.position = position

        # Event colors
        self.event_colors = {
            "kill": Vec4(1, 0.3, 0.3, 1),     # Red
            "respawn": Vec4(0.3, 1, 0.3, 1),  # Green
            "default": Vec4(1, 1, 0, 1),      # Yellow
        }

        # Create UI elements
        self.frame: Optional[DirectFrame] = None
        self.slider: Optional[DirectSlider] = None
        self.markers: List[NodePath] = []

        self._create_ui()
        self._create_markers()

    def _create_ui(self):
        """Create UI elements."""
        # Main frame
        self.frame = DirectFrame(
            parent=self.parent,
            frameColor=(0.1, 0.1, 0.15, 0.9),
            frameSize=(
                -self.width / 2 - 0.15,
                self.width / 2 + 0.15,
                -self.height - 0.03,
                self.height + 0.03,
            ),
            pos=(self.position[0], 0, self.position[1]),
        )

        # Play/Pause button
        self.play_button = DirectButton(
            parent=self.frame,
            text="||",
            text_scale=0.04,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.04, 0.04, -0.03, 0.03),
            pos=(-self.width / 2 - 0.08, 0, 0),
            command=self._on_play_pause,
        )

        # Timeline slider
        self.slider = DirectSlider(
            parent=self.frame,
            range=(0, 1),
            value=0,
            pageSize=0.1,
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(
                -self.width / 2,
                self.width / 2,
                -0.015,
                0.015,
            ),
            thumb_frameColor=(0.8, 0.8, 0.9, 1),
            thumb_frameSize=(-0.015, 0.015, -0.03, 0.03),
            pos=(0, 0, 0),
            command=self._on_slider_change,
        )

        # Current time display
        self.time_label = DirectLabel(
            parent=self.frame,
            text="0:00 / 0:00",
            text_scale=0.03,
            text_fg=(0.8, 0.8, 0.8, 1),
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -self.height + 0.01),
        )

        # Speed display
        self.speed_label = DirectLabel(
            parent=self.frame,
            text="1.0x",
            text_scale=0.03,
            text_fg=(1, 1, 0, 1),
            text_align=TextNode.ARight,
            frameColor=(0, 0, 0, 0),
            pos=(self.width / 2 + 0.1, 0, 0),
        )

        # Speed buttons
        self.slow_button = DirectButton(
            parent=self.frame,
            text="<",
            text_scale=0.03,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.02, 0.02, -0.02, 0.02),
            pos=(self.width / 2 + 0.04, 0, 0.04),
            command=self._on_slow_down,
        )

        self.fast_button = DirectButton(
            parent=self.frame,
            text=">",
            text_scale=0.03,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.02, 0.02, -0.02, 0.02),
            pos=(self.width / 2 + 0.04, 0, -0.04),
            command=self._on_speed_up,
        )

        # Step buttons
        self.step_back_button = DirectButton(
            parent=self.frame,
            text="|<",
            text_scale=0.03,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.025, 0.025, -0.02, 0.02),
            pos=(-self.width / 2 + 0.04, 0, -self.height + 0.01),
            command=self._on_step_back,
        )

        self.step_forward_button = DirectButton(
            parent=self.frame,
            text=">|",
            text_scale=0.03,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.025, 0.025, -0.02, 0.02),
            pos=(-self.width / 2 + 0.1, 0, -self.height + 0.01),
            command=self._on_step_forward,
        )

        # Jump to event buttons
        self.prev_event_button = DirectButton(
            parent=self.frame,
            text="<<",
            text_scale=0.03,
            text_fg=(1, 0.5, 0.5, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.025, 0.025, -0.02, 0.02),
            pos=(self.width / 2 - 0.1, 0, -self.height + 0.01),
            command=self._on_prev_event,
        )

        self.next_event_button = DirectButton(
            parent=self.frame,
            text=">>",
            text_scale=0.03,
            text_fg=(1, 0.5, 0.5, 1),
            frameColor=(0.2, 0.2, 0.25, 1),
            frameSize=(-0.025, 0.025, -0.02, 0.02),
            pos=(self.width / 2 - 0.04, 0, -self.height + 0.01),
            command=self._on_next_event,
        )

        # Keyboard hints
        self.hints_label = DirectLabel(
            parent=self.frame,
            text="Space: Play/Pause | < >: Step | []: Speed | E: Next Kill",
            text_scale=0.02,
            text_fg=(0.5, 0.5, 0.5, 1),
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, self.height + 0.02),
        )

    def _create_markers(self):
        """Create event markers on timeline."""
        # Clear existing markers
        for marker in self.markers:
            marker.removeNode()
        self.markers.clear()

        if not self.controller.events:
            return

        # Create marker for each event
        for event in self.controller.events:
            progress = event.time / max(1, self.controller.duration)
            x_pos = (progress - 0.5) * self.width

            color = self.event_colors.get(
                event.event_type,
                self.event_colors["default"]
            )

            marker = self._create_marker_node(x_pos, color)
            self.markers.append(marker)

    def _create_marker_node(self, x_pos: float, color: Vec4) -> NodePath:
        """Create a single marker node."""
        cm = CardMaker('marker')
        cm.setFrame(-0.008, 0.008, 0.015, 0.045)

        marker = self.frame.attachNewNode(cm.generate())
        marker.setPos(x_pos, 0, 0)
        marker.setColor(color)
        marker.setTransparency(TransparencyAttrib.MAlpha)

        return marker

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_play_pause(self):
        """Handle play/pause button click."""
        self.controller.toggle_play_pause()

    def _on_slider_change(self):
        """Handle slider drag."""
        if self.slider:
            progress = self.slider['value']
            self.controller.seek_to_progress(progress)

    def _on_slow_down(self):
        """Handle slow down button."""
        self.controller.slow_down()

    def _on_speed_up(self):
        """Handle speed up button."""
        self.controller.speed_up()

    def _on_step_back(self):
        """Handle step back button."""
        self.controller.step_backward(10)

    def _on_step_forward(self):
        """Handle step forward button."""
        self.controller.step_forward(10)

    def _on_prev_event(self):
        """Handle previous event button."""
        self.controller.jump_to_prev_event("kill")

    def _on_next_event(self):
        """Handle next event button."""
        self.controller.jump_to_next_event("kill")

    # =========================================================================
    # Update
    # =========================================================================

    def update(self):
        """Update widget state."""
        # Update slider position
        if self.slider and not getattr(self.slider, "dragging", False):
            self.slider['value'] = self.controller.progress

        # Update time display
        current = self.controller.format_time(self.controller.current_time)
        total = self.controller.format_time(
            self.controller.duration + (self.controller.frames[0].match_time if self.controller.frames else 0)
        )
        self.time_label['text'] = f"{current} / {total}"

        # Update play button
        if self.controller.is_playing:
            self.play_button['text'] = "||"
        else:
            self.play_button['text'] = ">"

        # Update speed display
        self.speed_label['text'] = f"{self.controller.speed.value}x"

    def show(self):
        """Show the widget."""
        if self.frame:
            self.frame.show()

    def hide(self):
        """Hide the widget."""
        if self.frame:
            self.frame.hide()

    def cleanup(self):
        """Clean up resources."""
        for marker in self.markers:
            marker.removeNode()

        if self.frame:
            self.frame.destroy()


class MiniMap:
    """
    Mini-map overlay showing UAV positions.

    Top-down view of all UAVs in the arena.
    """

    def __init__(
        self,
        parent: NodePath,
        size: float = 0.3,
        position: Tuple[float, float] = (1.1, 0.7),
    ):
        """
        Initialize mini-map.

        Args:
            parent: Parent node
            size: Map size (aspect2d units)
            position: Map position (x, y)
        """
        self.parent = parent
        self.size = size
        self.position = position

        # Arena scale
        self.arena_size = 2000.0

        # Create UI
        self.frame: Optional[DirectFrame] = None
        self.uav_markers: dict = {}

        self._create_ui()

    def _create_ui(self):
        """Create UI elements."""
        # Background frame
        self.frame = DirectFrame(
            parent=self.parent,
            frameColor=(0.1, 0.1, 0.15, 0.8),
            frameSize=(
                -self.size / 2,
                self.size / 2,
                -self.size / 2,
                self.size / 2,
            ),
            pos=(self.position[0], 0, self.position[1]),
        )

        # Border
        cm = CardMaker('border')
        cm.setFrame(
            -self.size / 2 - 0.005,
            self.size / 2 + 0.005,
            -self.size / 2 - 0.005,
            self.size / 2 + 0.005,
        )
        border = self.frame.attachNewNode(cm.generate())
        border.setPos(0, -0.1, 0)
        border.setColor(Vec4(0.3, 0.3, 0.4, 1))

        # Grid lines
        for i in range(-2, 3):
            line_x = self.frame.attachNewNode(cm.generate())
            line_x.setScale(self.size, 1, 0.002)
            line_x.setPos(0, -0.05, i * self.size / 4)
            line_x.setColor(Vec4(0.2, 0.2, 0.25, 0.5))

            line_y = self.frame.attachNewNode(cm.generate())
            line_y.setScale(0.002, 1, self.size)
            line_y.setPos(i * self.size / 4, -0.05, 0)
            line_y.setColor(Vec4(0.2, 0.2, 0.25, 0.5))

    def update(self, uavs: list):
        """
        Update UAV positions on mini-map.

        Args:
            uavs: List of UAVState objects
        """
        # Create/update markers for each UAV
        seen_ids = set()

        for uav in uavs:
            if not uav.alive:
                continue

            seen_ids.add(uav.id)

            # Convert world position to map position
            map_x = (uav.position[0] / self.arena_size) * self.size
            map_y = (uav.position[1] / self.arena_size) * self.size

            # Clamp to map bounds
            map_x = max(-self.size / 2, min(self.size / 2, map_x))
            map_y = max(-self.size / 2, min(self.size / 2, map_y))

            if uav.id not in self.uav_markers:
                # Create new marker
                cm = CardMaker('uav_marker')
                cm.setFrame(-0.01, 0.01, -0.01, 0.01)
                marker = self.frame.attachNewNode(cm.generate())
                self.uav_markers[uav.id] = marker

            marker = self.uav_markers[uav.id]
            marker.setPos(map_x, 0, map_y)

            # Team color
            if uav.team == 0:
                marker.setColor(Vec4(1, 0.3, 0.3, 1))
            else:
                marker.setColor(Vec4(0.3, 0.5, 1, 1))

        # Remove markers for dead/missing UAVs
        for uav_id in list(self.uav_markers.keys()):
            if uav_id not in seen_ids:
                self.uav_markers[uav_id].removeNode()
                del self.uav_markers[uav_id]

    def set_arena_size(self, size: float):
        """Set arena size for position scaling."""
        self.arena_size = size

    def cleanup(self):
        """Clean up resources."""
        for marker in self.uav_markers.values():
            marker.removeNode()
        if self.frame:
            self.frame.destroy()

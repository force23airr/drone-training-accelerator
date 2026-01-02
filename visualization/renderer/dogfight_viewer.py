#!/usr/bin/env python3
"""
Panda3D Dogfight Viewer

Real-time 3D visualization of fixed-wing UAV combat.

Features:
- Multiple camera modes (chase, spectator, tactical, auto)
- Smooth interpolation between frames
- Combat effects (trails, explosions, muzzle flash)
- HUD overlays (telemetry, kill feed, scoreboard)
- Replay playback

Usage:
    python -m visualization.renderer.dogfight_viewer --connect localhost:5555
    python -m visualization.renderer.dogfight_viewer --replay path/to/replay.dfrp
"""

import sys
import time
import math
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# Check for Panda3D
try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from direct.gui.OnscreenText import OnscreenText
    from direct.gui.DirectGui import DirectFrame
    from panda3d.core import (
        Point3, Vec3, Vec4, LColor,
        NodePath, GeomNode, Geom, GeomVertexFormat, GeomVertexData,
        GeomVertexWriter, GeomTriangles, GeomLines, GeomLinestrips,
        CardMaker, TextNode, TextureStage,
        AmbientLight, DirectionalLight, PointLight,
        Material, Fog,
        WindowProperties, ClockObject,
        TransparencyAttrib,
        loadPrcFileData,
    )
    HAS_PANDA3D = True
except ImportError:
    HAS_PANDA3D = False

# Local imports
sys.path.insert(0, '.')
from visualization.streaming import StateReceiver, FrameState, UAVState, ReplayReader
from visualization.renderer.effects import EffectsManager
from visualization.replay import ReplayController, TimelineWidget, MiniMap

logger = logging.getLogger(__name__)


# Configure Panda3D before creating ShowBase
if HAS_PANDA3D:
    loadPrcFileData('', 'window-title Dogfight Combat Viewer')
    loadPrcFileData('', 'sync-video #t')  # VSync
    loadPrcFileData('', 'show-frame-rate-meter #t')
    loadPrcFileData('', 'text-encoding utf8')


@dataclass
class CameraState:
    """Current camera state."""
    mode: str = "chase"  # chase, spectator, tactical, auto
    target_uav_id: Optional[int] = None
    position: Vec3 = None
    look_at: Vec3 = None
    smoothing: float = 0.1

    def __post_init__(self):
        if self.position is None:
            self.position = Vec3(0, -100, 50)
        if self.look_at is None:
            self.look_at = Vec3(0, 0, 0)


class UAVModel:
    """3D model for a single UAV."""

    def __init__(self, parent: NodePath, uav_id: int, team: int):
        """
        Create UAV model.

        Args:
            parent: Parent node
            uav_id: UAV identifier
            team: Team (0=red, 1=blue)
        """
        self.uav_id = uav_id
        self.team = team
        self.root = parent.attachNewNode(f"uav_{uav_id}")

        # Team colors - more distinct
        if team == 0:  # Red team
            self.color = Vec4(1, 0.15, 0.15, 1)
            self.trail_color = Vec4(1, 0.4, 0.4, 0.6)
            self.label_color = (1, 0.3, 0.3, 1)
        else:  # Blue team
            self.color = Vec4(0.15, 0.4, 1, 1)
            self.trail_color = Vec4(0.4, 0.6, 1, 0.6)
            self.label_color = (0.3, 0.5, 1, 1)

        # Create simple jet model (cone + wings)
        self._create_model()

        # Trail
        self.trail_points: deque = deque(maxlen=120)  # 2 seconds at 60fps
        self.trail_node: Optional[NodePath] = None

        # Effects
        self.is_alive = True
        self.explosion_time = 0.0

    def _create_model(self):
        """Create simple jet geometry."""
        # Fuselage (elongated box)
        fuselage = self._create_box(2, 8, 1.5)
        fuselage.reparentTo(self.root)
        fuselage.setColor(self.color)

        # Nose cone
        nose = self._create_cone(1, 3)
        nose.reparentTo(self.root)
        nose.setPos(0, 5.5, 0)
        nose.setColor(self.color * 0.8)

        # Wings
        wing_l = self._create_box(6, 3, 0.3)
        wing_l.reparentTo(self.root)
        wing_l.setPos(-4, 0, 0)
        wing_l.setColor(self.color)

        wing_r = self._create_box(6, 3, 0.3)
        wing_r.reparentTo(self.root)
        wing_r.setPos(4, 0, 0)
        wing_r.setColor(self.color)

        # Tail
        tail_v = self._create_box(0.3, 2, 3)
        tail_v.reparentTo(self.root)
        tail_v.setPos(0, -3, 2)
        tail_v.setColor(self.color)

        # Horizontal stabilizer
        tail_h = self._create_box(4, 1.5, 0.2)
        tail_h.reparentTo(self.root)
        tail_h.setPos(0, -3, 0)
        tail_h.setColor(self.color)

        # Engine glow
        self.engine_glow = self._create_sphere(0.8)
        self.engine_glow.reparentTo(self.root)
        self.engine_glow.setPos(0, -4, 0)
        self.engine_glow.setColor(Vec4(1, 0.6, 0.2, 0.8))
        self.engine_glow.setTransparency(TransparencyAttrib.MAlpha)

        # Scale entire model
        self.root.setScale(2)

    def _create_box(self, sx, sy, sz) -> NodePath:
        """Create a simple box."""
        cm = CardMaker('box')
        cm.setFrame(-sx/2, sx/2, -sz/2, sz/2)

        box = self.root.attachNewNode('box')

        # Front
        front = box.attachNewNode(cm.generate())
        front.setPos(0, sy/2, 0)

        # Back
        back = box.attachNewNode(cm.generate())
        back.setPos(0, -sy/2, 0)
        back.setH(180)

        # Left
        left = box.attachNewNode(cm.generate())
        left.setPos(-sx/2, 0, 0)
        left.setH(-90)

        # Right
        right = box.attachNewNode(cm.generate())
        right.setPos(sx/2, 0, 0)
        right.setH(90)

        # Top
        cm2 = CardMaker('top')
        cm2.setFrame(-sx/2, sx/2, -sy/2, sy/2)
        top = box.attachNewNode(cm2.generate())
        top.setPos(0, 0, sz/2)
        top.setP(-90)

        # Bottom
        bottom = box.attachNewNode(cm2.generate())
        bottom.setPos(0, 0, -sz/2)
        bottom.setP(90)

        return box

    def _create_cone(self, radius, height) -> NodePath:
        """Create a simple cone."""
        # Use a pyramid approximation
        return self._create_box(radius * 2, height, radius * 2)

    def _create_sphere(self, radius) -> NodePath:
        """Create a simple sphere (octahedron approximation)."""
        return self._create_box(radius * 2, radius * 2, radius * 2)

    def update(self, state: UAVState, dt: float):
        """Update model from state."""
        self.is_alive = state.alive

        if state.alive:
            # Position (swap Y/Z for Panda3D coordinate system)
            self.root.setPos(
                state.position[0],
                state.position[1],
                state.position[2]
            )

            # Orientation (roll, pitch, yaw in radians)
            self.root.setHpr(
                math.degrees(state.orientation[2]),   # Yaw -> H
                math.degrees(state.orientation[1]),   # Pitch -> P
                math.degrees(state.orientation[0])    # Roll -> R
            )

            # Engine glow based on speed
            glow_intensity = min(1.0, state.speed / 300.0)
            self.engine_glow.setScale(0.5 + glow_intensity)
            self.engine_glow.setColor(Vec4(1, 0.4 + glow_intensity * 0.4, 0.1, 0.8))

            # Add trail point
            self.trail_points.append(Point3(*state.position))

            self.root.show()
        else:
            # Death animation
            self.explosion_time += dt
            if self.explosion_time < 2.0:
                # Falling debris
                self.root.setZ(self.root.getZ() - 50 * dt)
                self.root.setR(self.root.getR() + 180 * dt)
                self.root.setP(self.root.getP() + 90 * dt)
            else:
                self.root.hide()

    def update_trail(self):
        """Update trail geometry."""
        if len(self.trail_points) < 2:
            return

        if self.trail_node:
            self.trail_node.removeNode()

        # Create line strip for trail
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('trail', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        for i, point in enumerate(self.trail_points):
            vertex.addData3f(point)
            alpha = i / len(self.trail_points) * 0.6
            c = self.trail_color
            color.addData4f(c.x, c.y, c.z, alpha)

        lines = GeomLinestrips(Geom.UHStatic)
        lines.addConsecutiveVertices(0, len(self.trail_points))
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('trail_geom')
        node.addGeom(geom)

        self.trail_node = self.root.getParent().attachNewNode(node)
        self.trail_node.setTransparency(TransparencyAttrib.MAlpha)
        self.trail_node.setRenderModeThickness(2)

    def cleanup(self):
        """Remove model."""
        if self.trail_node:
            self.trail_node.removeNode()
        self.root.removeNode()


class DogfightViewer(ShowBase):
    """
    Main Panda3D application for dogfight visualization.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        replay_path: Optional[str] = None,
    ):
        """
        Initialize viewer.

        Args:
            host: StateStreamer host
            port: StateStreamer port
            replay_path: Path to replay file (instead of live stream)
        """
        if not HAS_PANDA3D:
            raise ImportError("panda3d required: pip install panda3d")

        super().__init__()

        self.host = host
        self.port = port
        self.replay_path = replay_path

        # State
        self.receiver: Optional[StateReceiver] = None
        self.replay_reader: Optional[ReplayReader] = None
        self.latest_frame: Optional[FrameState] = None

        # Models
        self.uav_models: Dict[int, UAVModel] = {}
        self.arena_node: Optional[NodePath] = None
        self._arena_size: Optional[float] = None
        self._arena_height_min: Optional[float] = None
        self._arena_height_max: Optional[float] = None

        # Camera
        self.camera_state = CameraState()
        self.selected_uav_id: Optional[int] = None  # Will be set when UAVs appear

        # HUD
        self.hud_elements: Dict[str, any] = {}

        # Effects
        self.effects_manager: Optional[EffectsManager] = None

        # Replay
        self.replay_controller: Optional[ReplayController] = None
        self.timeline_widget: Optional[TimelineWidget] = None
        self.mini_map: Optional[MiniMap] = None

        # Initialize
        self._setup_scene()
        self._setup_lighting()
        self._setup_camera()
        self._setup_controls()
        self._setup_hud()
        self._setup_effects()
        self._setup_minimap()

        # Connect to stream or load replay
        if replay_path:
            self._start_replay()
        else:
            self._connect_stream()

        # Start update task
        self.taskMgr.add(self._update_task, 'update')
        self.taskMgr.add(self._update_trails_task, 'update_trails')

    def _setup_scene(self):
        """Setup the 3D scene."""
        # Set background color (sky)
        self.setBackgroundColor(0.4, 0.6, 0.9, 1)

        # Create arena
        self._create_arena(2000, 100, 2000)

        # Fog for depth
        fog = Fog('fog')
        fog.setColor(0.5, 0.6, 0.8)
        fog.setExpDensity(0.0003)
        self.render.setFog(fog)

    def _create_arena(self, size: float, height_min: float, height_max: float):
        """Create arena ground and boundaries."""
        if self.arena_node:
            self.arena_node.removeNode()
        self.arena_node = self.render.attachNewNode('arena')
        self._arena_size = size
        self._arena_height_min = height_min
        self._arena_height_max = height_max

        # Ground plane
        cm = CardMaker('ground')
        half = size / 2
        cm.setFrame(-half, half, -half, half)
        ground = self.arena_node.attachNewNode(cm.generate())
        ground.setP(-90)
        ground.setZ(0)

        # Grid texture effect
        ground.setColor(Vec4(0.3, 0.4, 0.3, 1))

        # Grid lines
        self._create_grid(size, 200, height_min)

        # Altitude reference bands
        for alt in [100, 500, 1000, 1500]:
            if alt < height_max:
                self._create_altitude_ring(size, alt)

        # Boundary markers
        self._create_boundary(size, height_min, height_max)

    def _create_grid(self, size: float, spacing: float, height: float):
        """Create ground grid."""
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('grid', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        half = size / 2
        num_lines = int(size / spacing) + 1

        for i in range(num_lines):
            x = -half + i * spacing

            # X lines
            vertex.addData3f(x, -half, height)
            vertex.addData3f(x, half, height)
            color.addData4f(0.4, 0.5, 0.4, 0.3)
            color.addData4f(0.4, 0.5, 0.4, 0.3)

            # Y lines
            vertex.addData3f(-half, x, height)
            vertex.addData3f(half, x, height)
            color.addData4f(0.4, 0.5, 0.4, 0.3)
            color.addData4f(0.4, 0.5, 0.4, 0.3)

        lines = GeomLines(Geom.UHStatic)
        for i in range(num_lines * 4):
            if i % 2 == 0:
                lines.addVertices(i, i + 1)
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('grid')
        node.addGeom(geom)

        grid_np = self.arena_node.attachNewNode(node)
        grid_np.setTransparency(TransparencyAttrib.MAlpha)

    def _create_altitude_ring(self, size: float, altitude: float):
        """Create altitude reference ring."""
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('ring', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        half = size / 2
        c = Vec4(0.5, 0.5, 0.5, 0.2)

        # Box at altitude
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            vertex.addData3f(x1, y1, altitude)
            vertex.addData3f(x2, y2, altitude)
            color.addData4f(c.x, c.y, c.z, c.w)
            color.addData4f(c.x, c.y, c.z, c.w)

        lines = GeomLines(Geom.UHStatic)
        for i in range(0, 8, 2):
            lines.addVertices(i, i + 1)
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode(f'alt_ring_{altitude}')
        node.addGeom(geom)

        ring_np = self.arena_node.attachNewNode(node)
        ring_np.setTransparency(TransparencyAttrib.MAlpha)

    def _create_boundary(self, size: float, height_min: float, height_max: float):
        """Create arena boundary visualization."""
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('boundary', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        half = size / 2
        c = Vec4(1, 1, 0, 0.5)  # Yellow boundary

        # Vertical lines at corners
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for x, y in corners:
            vertex.addData3f(x, y, height_min)
            vertex.addData3f(x, y, height_max)
            color.addData4f(c.x, c.y, c.z, c.w)
            color.addData4f(c.x, c.y, c.z, c.w)

        lines = GeomLines(Geom.UHStatic)
        for i in range(0, 8, 2):
            lines.addVertices(i, i + 1)
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('boundary')
        node.addGeom(geom)

        boundary_np = self.arena_node.attachNewNode(node)
        boundary_np.setTransparency(TransparencyAttrib.MAlpha)
        boundary_np.setRenderModeThickness(2)

    def _setup_lighting(self):
        """Setup scene lighting."""
        # Ambient light
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.4, 0.4, 0.5, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        # Sun light
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(1, 0.95, 0.9, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self.render.setLight(sun_np)

        # Fill light
        fill = DirectionalLight('fill')
        fill.setColor(Vec4(0.3, 0.3, 0.4, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(-135, -30, 0)
        self.render.setLight(fill_np)

    def _setup_camera(self):
        """Setup camera."""
        self.disableMouse()
        self.camera.setPos(0, -200, 100)
        self.camera.lookAt(0, 0, 0)

    def _setup_controls(self):
        """Setup keyboard/mouse controls."""
        # Camera mode
        self.accept('1', self._set_camera_mode, ['chase'])
        self.accept('2', self._set_camera_mode, ['spectator'])
        self.accept('3', self._set_camera_mode, ['tactical'])
        self.accept('4', self._set_camera_mode, ['auto'])

        # Target selection
        self.accept('tab', self._cycle_target)
        self.accept('shift-tab', self._cycle_target, [-1])

        # Toggle velocity vectors
        self.accept('v', self._toggle_velocity_vectors)

        # Spectator controls
        self.accept('w', self._move_spectator, [Vec3(0, 100, 0)])
        self.accept('s', self._move_spectator, [Vec3(0, -100, 0)])
        self.accept('a', self._move_spectator, [Vec3(-100, 0, 0)])
        self.accept('d', self._move_spectator, [Vec3(100, 0, 0)])
        self.accept('space', self._move_spectator, [Vec3(0, 0, 100)])
        self.accept('shift', self._move_spectator, [Vec3(0, 0, -100)])

        # Exit
        self.accept('escape', sys.exit)

    def _set_camera_mode(self, mode: str):
        """Set camera mode."""
        self.camera_state.mode = mode
        logger.info(f"Camera mode: {mode}")

    def _cycle_target(self, direction: int = 1):
        """Cycle through UAV targets."""
        if not self.uav_models:
            return

        ids = sorted(self.uav_models.keys())
        if self.selected_uav_id in ids:
            idx = ids.index(self.selected_uav_id)
            idx = (idx + direction) % len(ids)
            self.selected_uav_id = ids[idx]
        else:
            self.selected_uav_id = ids[0]

        logger.info(f"Selected UAV: {self.selected_uav_id}")

    def _move_spectator(self, delta: Vec3):
        """Move spectator camera."""
        if self.camera_state.mode == 'spectator':
            self.camera_state.position += delta * 0.1

    def _toggle_velocity_vectors(self):
        """Toggle velocity vector display."""
        if self.effects_manager:
            self.effects_manager.toggle_velocity_vectors()
            logger.info(f"Velocity vectors: {'ON' if self.effects_manager.show_velocity_vectors else 'OFF'}")

    def _setup_hud(self):
        """Setup HUD elements."""
        # Score - larger and more prominent
        self.hud_elements['score'] = OnscreenText(
            text="RED 0 - 0 BLUE",
            pos=(0, 0.92),
            scale=0.09,
            fg=(1, 1, 1, 1),
            shadow=(0, 0, 0, 0.9),
            align=TextNode.ACenter,
        )

        # Time
        self.hud_elements['time'] = OnscreenText(
            text="Time: 0:00",
            pos=(0, 0.82),
            scale=0.05,
            fg=(0.9, 0.9, 0.9, 1),
            shadow=(0, 0, 0, 0.5),
            align=TextNode.ACenter,
        )

        # Camera mode indicator - more visible
        self.hud_elements['camera'] = OnscreenText(
            text="[1] CHASE CAM",
            pos=(-1.3, 0.92),
            scale=0.045,
            fg=(0.2, 1, 0.2, 1),  # Green
            shadow=(0, 0, 0, 0.8),
            align=TextNode.ALeft,
        )

        # Controls hint - clearer
        self.hud_elements['controls'] = OnscreenText(
            text="[1-4] Camera  [Tab] Switch UAV  [V] Vectors  [Space] Play/Pause",
            pos=(0, -0.95),
            scale=0.028,
            fg=(0.6, 0.6, 0.6, 1),
            align=TextNode.ACenter,
        )

        # Kill feed (top right) - moved up for visibility
        self.hud_elements['killfeed'] = OnscreenText(
            text="",
            pos=(1.3, 0.7),
            scale=0.038,
            fg=(1, 0.8, 0.2, 1),  # Gold color for kills
            shadow=(0, 0, 0, 0.7),
            align=TextNode.ARight,
            mayChange=True,
        )

        # Selected UAV info - clearer panel
        self.hud_elements['selected'] = OnscreenText(
            text="",
            pos=(-1.3, 0.75),
            scale=0.04,
            fg=(1, 1, 1, 1),
            shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # Target indicator label
        self.hud_elements['target_label'] = OnscreenText(
            text="TRACKING:",
            pos=(-1.3, 0.82),
            scale=0.03,
            fg=(0.7, 0.7, 0.7, 1),
            align=TextNode.ALeft,
        )

    def _setup_effects(self):
        """Setup effects manager."""
        self.effects_manager = EffectsManager(self.render)

    def _setup_minimap(self):
        """Setup mini-map overlay."""
        self.mini_map = MiniMap(self.aspect2d, size=0.25, position=(1.1, 0.7))

    def _connect_stream(self):
        """Connect to live stream."""
        self.receiver = StateReceiver(self.host, self.port)
        self.receiver.connect()
        logger.info(f"Connected to stream at {self.host}:{self.port}")

    def _start_replay(self):
        """Start replay playback with controller and timeline."""
        self.replay_controller = ReplayController(self.replay_path)

        # Create timeline widget
        self.timeline_widget = TimelineWidget(
            self.aspect2d,
            self.replay_controller,
            width=1.6,
            height=0.08,
            position=(0, -0.88),
        )

        # Setup replay keyboard controls
        self.accept('space', self._toggle_replay_playback)
        self.accept('[', self.replay_controller.slow_down)
        self.accept(']', self.replay_controller.speed_up)
        self.accept('arrow_left', self._step_replay_back)
        self.accept('arrow_right', self._step_replay_forward)
        self.accept('e', lambda: self.replay_controller.jump_to_next_event("kill"))
        self.accept('q', lambda: self.replay_controller.jump_to_prev_event("kill"))

        # Start playback
        self.replay_controller.play()
        logger.info(f"Playing replay: {self.replay_path}")

    def _toggle_replay_playback(self):
        """Toggle replay play/pause."""
        if self.replay_controller:
            self.replay_controller.toggle_play_pause()

    def _step_replay_back(self):
        """Step replay backward."""
        if self.replay_controller:
            self.replay_controller.step_backward(5)

    def _step_replay_forward(self):
        """Step replay forward."""
        if self.replay_controller:
            self.replay_controller.step_forward(5)

    def _update_task(self, task) -> int:
        """Main update task."""
        dt = globalClock.getDt()

        # Get frame
        if self.receiver:
            frame = self.receiver.receive()
            if frame:
                self.latest_frame = frame
        elif self.replay_controller:
            frame = self.replay_controller.update()
            if frame:
                self.latest_frame = frame

        if self.latest_frame:
            self._update_from_frame(self.latest_frame, dt)
            self._update_hud(self.latest_frame)

            # Process combat events for effects
            if self.effects_manager and self.latest_frame.events:
                uav_positions = {}
                for uav in self.latest_frame.uavs:
                    uav_positions[uav.id] = Vec3(*uav.position)
                self.effects_manager.process_combat_events(
                    self.latest_frame.events,
                    uav_positions
                )

            # Update mini-map
            if self.mini_map:
                self.mini_map.update(self.latest_frame.uavs)

        # Update effects
        if self.effects_manager:
            self.effects_manager.update(dt)

        # Update timeline widget
        if self.timeline_widget:
            self.timeline_widget.update()

        # Update camera
        self._update_camera(dt)

        return Task.cont

    def _update_trails_task(self, task) -> int:
        """Update trail geometries (less frequently)."""
        for model in self.uav_models.values():
            model.update_trail()
        return Task.again

    def _update_from_frame(self, frame: FrameState, dt: float):
        """Update scene from frame state."""
        # Update arena if needed
        if frame.arena_size and self.arena_node:
            height_min = frame.arena_height_min if frame.arena_height_min is not None else (self._arena_height_min or 0.0)
            height_max = frame.arena_height_max if frame.arena_height_max is not None else (self._arena_height_max or 0.0)
            if (
                self._arena_size != frame.arena_size
                or self._arena_height_min != height_min
                or self._arena_height_max != height_max
            ):
                self._create_arena(frame.arena_size, height_min, height_max)
                if self.mini_map:
                    self.mini_map.set_arena_size(frame.arena_size)

        # Update/create UAV models
        seen_ids = set()
        for uav in frame.uavs:
            seen_ids.add(uav.id)

            if uav.id not in self.uav_models:
                # Create new model
                model = UAVModel(self.render, uav.id, uav.team)
                self.uav_models[uav.id] = model
                logger.info(f"Created UAV model {uav.id}")

            self.uav_models[uav.id].update(uav, dt)

            # Update velocity vectors
            if self.effects_manager:
                self.effects_manager.update_velocity_vector(
                    uav.id,
                    Vec3(*uav.position),
                    Vec3(*uav.velocity),
                )

        # Remove old models
        for uav_id in list(self.uav_models.keys()):
            if uav_id not in seen_ids:
                self.uav_models[uav_id].cleanup()
                del self.uav_models[uav_id]

        # Set default target
        if self.selected_uav_id is None and self.uav_models:
            self.selected_uav_id = min(self.uav_models.keys())

    def _update_camera(self, dt: float):
        """Update camera position."""
        mode = self.camera_state.mode
        alpha = self.camera_state.smoothing

        if mode == 'chase':
            self._update_chase_camera(dt, alpha)
        elif mode == 'spectator':
            self._update_spectator_camera(dt, alpha)
        elif mode == 'tactical':
            self._update_tactical_camera(dt, alpha)
        elif mode == 'auto':
            self._update_auto_camera(dt, alpha)

        # Apply smoothed position
        current_pos = self.camera.getPos()
        target_pos = self.camera_state.position
        new_pos = current_pos + (target_pos - current_pos) * alpha

        self.camera.setPos(new_pos)
        self.camera.lookAt(self.camera_state.look_at)

    def _update_chase_camera(self, dt: float, alpha: float):
        """Update chase camera behind target UAV."""
        # If no valid target, try to find one
        if self.selected_uav_id is None or self.selected_uav_id not in self.uav_models:
            if self.uav_models:
                self.selected_uav_id = min(self.uav_models.keys())
            else:
                return  # No UAVs yet

        model = self.uav_models.get(self.selected_uav_id)
        if model is None:
            return

        # If selected UAV is dead, switch to next alive UAV
        if not model.is_alive:
            for uav_id, m in self.uav_models.items():
                if m.is_alive:
                    self.selected_uav_id = uav_id
                    model = m
                    break

        # Get UAV position and orientation
        uav_pos = model.root.getPos()
        uav_hpr = model.root.getHpr()

        # Camera offset: behind and above the UAV
        distance_back = 100  # Further back for better view
        height_above = 30

        # Get UAV's forward direction from heading
        heading_rad = math.radians(uav_hpr.x)
        pitch_rad = math.radians(uav_hpr.y) * 0.2  # Slight pitch influence

        # Forward vector (where UAV is pointing)
        forward = Vec3(
            math.cos(heading_rad),
            math.sin(heading_rad),
            0  # Keep camera level
        )

        # Camera position: behind and above
        cam_pos = Vec3(
            uav_pos.x - forward.x * distance_back,
            uav_pos.y - forward.y * distance_back,
            uav_pos.z + height_above
        )

        # Look at the UAV position (slightly ahead)
        look_at = Vec3(
            uav_pos.x + forward.x * 20,
            uav_pos.y + forward.y * 20,
            uav_pos.z
        )

        # Update camera state
        self.camera_state.position = cam_pos
        self.camera_state.look_at = look_at

    def _update_spectator_camera(self, dt: float, alpha: float):
        """Update free spectator camera."""
        # Look at center or selected UAV
        if self.selected_uav_id in self.uav_models:
            model = self.uav_models[self.selected_uav_id]
            self.camera_state.look_at = model.root.getPos()
        else:
            self.camera_state.look_at = Vec3(0, 0, 500)

    def _update_tactical_camera(self, dt: float, alpha: float):
        """Update top-down tactical camera."""
        self.camera_state.position = Vec3(0, 0, 2000)
        self.camera_state.look_at = Vec3(0, 0, 0)

    def _update_auto_camera(self, dt: float, alpha: float):
        """Update AI-directed camera."""
        # Find most interesting action
        if self.latest_frame and self.latest_frame.events:
            # Focus on latest event
            event = self.latest_frame.events[-1]
            if event.attacker_id in self.uav_models:
                self.selected_uav_id = event.attacker_id

        # Use chase mode logic
        self._update_chase_camera(dt, alpha * 0.5)  # Slower for cinematic

    def _update_hud(self, frame: FrameState):
        """Update HUD elements."""
        # Score with team colors in text
        self.hud_elements['score'].setText(
            f"RED {frame.red_score}  -  {frame.blue_score} BLUE"
        )

        # Time
        mins = int(frame.match_time // 60)
        secs = int(frame.match_time % 60)
        self.hud_elements['time'].setText(f"{mins}:{secs:02d}")

        # Camera mode with hotkey
        mode_labels = {
            'chase': '[1] CHASE CAM',
            'spectator': '[2] FREE CAM',
            'tactical': '[3] TACTICAL',
            'auto': '[4] AUTO CAM',
        }
        self.hud_elements['camera'].setText(
            mode_labels.get(self.camera_state.mode, '[?] UNKNOWN')
        )

        # Selected UAV info - cleaner format
        if self.selected_uav_id is not None:
            for uav in frame.uavs:
                if uav.id == self.selected_uav_id:
                    team = "RED" if uav.team == 0 else "BLUE"
                    team_color = (1, 0.3, 0.3, 1) if uav.team == 0 else (0.3, 0.5, 1, 1)

                    # Update target label color
                    self.hud_elements['target_label'].setFg(team_color)

                    status = "ALIVE" if uav.alive else "DEAD"
                    maneuver = uav.current_maneuver or "---"

                    # Health bar representation
                    health_pct = uav.health / 100.0
                    health_bars = int(health_pct * 10)
                    health_str = "|" * health_bars + "." * (10 - health_bars)

                    self.hud_elements['selected'].setText(
                        f"{team} #{uav.id}\n"
                        f"HP [{health_str}] {uav.health:.0f}%\n"
                        f"SPD: {uav.speed:.0f} m/s\n"
                        f"ALT: {uav.altitude:.0f} m\n"
                        f"G-FORCE: {uav.g_force:.1f}G\n"
                        f"K/D: {uav.kills}/{uav.deaths}\n"
                        f"[{maneuver}]"
                    )

                    # Color the selected UAV text based on team
                    self.hud_elements['selected'].setFg(team_color)
                    break

        # Kill feed
        if frame.events:
            feed_lines = []
            for event in frame.events[-5:]:
                if event.event_type == 2:  # KILL
                    feed_lines.append(
                        f"[{frame.match_time:.0f}s] {event.attacker_id} -> {event.target_id}"
                    )
            self.hud_elements['killfeed'].setText("\n".join(feed_lines))


def main():
    parser = argparse.ArgumentParser(description='Dogfight Combat Viewer')
    parser.add_argument('--host', default='localhost', help='Streamer host')
    parser.add_argument('--port', type=int, default=5555, help='Streamer port')
    parser.add_argument('--connect', help='Host:port shorthand')
    parser.add_argument('--replay', help='Path to replay file')
    args = parser.parse_args()

    # Parse connect string
    host = args.host
    port = args.port
    if args.connect:
        if ':' in args.connect:
            host, port = args.connect.split(':')
            port = int(port)
        else:
            host = args.connect

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("DOGFIGHT COMBAT VIEWER")
    print("=" * 60)
    print()
    print("Controls:")
    print("  1-4: Camera modes (Chase/Spectator/Tactical/Auto)")
    print("  Tab: Cycle target UAV")
    print("  V: Toggle velocity vectors")
    print("  WASD/Space/Shift: Move spectator camera")
    print()
    print("Replay Controls:")
    print("  Space: Play/Pause")
    print("  [ ]: Slow down / Speed up")
    print("  Left/Right: Step frames")
    print("  Q/E: Jump to prev/next kill")
    print()
    print("  ESC: Exit")
    print()

    if args.replay:
        print(f"Playing replay: {args.replay}")
        viewer = DogfightViewer(replay_path=args.replay)
    else:
        print(f"Connecting to: {host}:{port}")
        viewer = DogfightViewer(host=host, port=port)

    viewer.run()


if __name__ == '__main__':
    main()

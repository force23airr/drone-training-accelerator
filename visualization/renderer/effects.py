#!/usr/bin/env python3
"""
Visual Effects for Dogfight Visualization

Combat effects:
- Explosions (particle burst + expanding sphere)
- Missile trails (smoke particles)
- Muzzle flash (quick flash + tracer)
- Lock indicator (cone visualization)
- Damage sparks (hit effects)
- Velocity vectors (optional heading arrows)
"""

import math
import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import deque

try:
    from panda3d.core import (
        Point3, Vec3, Vec4, LColor,
        NodePath, GeomNode, Geom, GeomVertexFormat, GeomVertexData,
        GeomVertexWriter, GeomTriangles, GeomLines, GeomLinestrips,
        GeomPoints, CardMaker, TextNode,
        TransparencyAttrib,
    )
    HAS_PANDA3D = True
except ImportError:
    HAS_PANDA3D = False


@dataclass
class Particle:
    """Single particle in an effect."""
    position: Vec3
    velocity: Vec3
    color: Vec4
    size: float
    lifetime: float
    max_lifetime: float

    @property
    def alpha(self) -> float:
        """Fade out over lifetime."""
        return max(0, self.lifetime / self.max_lifetime)


class ParticleSystem:
    """Generic particle system for visual effects."""

    def __init__(self, parent: NodePath, max_particles: int = 100):
        self.parent = parent
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.node: Optional[NodePath] = None

    def emit(
        self,
        position: Vec3,
        velocity: Vec3,
        color: Vec4,
        size: float = 1.0,
        lifetime: float = 1.0,
        spread: float = 0.0,
        count: int = 1,
    ):
        """Emit particles."""
        for _ in range(count):
            # Add random spread
            spread_vel = Vec3(
                velocity.x + random.uniform(-spread, spread),
                velocity.y + random.uniform(-spread, spread),
                velocity.z + random.uniform(-spread, spread),
            )

            particle = Particle(
                position=Point3(position),
                velocity=spread_vel,
                color=color,
                size=size,
                lifetime=lifetime,
                max_lifetime=lifetime,
            )

            self.particles.append(particle)

        # Limit particle count
        while len(self.particles) > self.max_particles:
            self.particles.pop(0)

    def update(self, dt: float):
        """Update all particles."""
        alive_particles = []

        for p in self.particles:
            p.lifetime -= dt
            if p.lifetime > 0:
                # Physics update
                p.position += p.velocity * dt
                p.velocity.z -= 9.8 * dt  # Gravity
                alive_particles.append(p)

        self.particles = alive_particles
        self._rebuild_geometry()

    def _rebuild_geometry(self):
        """Rebuild particle geometry."""
        if self.node:
            self.node.removeNode()
            self.node = None

        if not self.particles:
            return

        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('particles', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        for p in self.particles:
            vertex.addData3f(p.position)
            c = p.color
            color.addData4f(c.x, c.y, c.z, c.w * p.alpha)

        points = GeomPoints(Geom.UHStatic)
        points.addConsecutiveVertices(0, len(self.particles))
        points.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(points)

        node = GeomNode('particle_geom')
        node.addGeom(geom)

        self.node = self.parent.attachNewNode(node)
        self.node.setTransparency(TransparencyAttrib.MAlpha)
        self.node.setRenderModeThickness(4)

    def cleanup(self):
        """Clean up resources."""
        if self.node:
            self.node.removeNode()
        self.particles.clear()


class ExplosionEffect:
    """Explosion visual effect with particles and expanding sphere."""

    def __init__(self, parent: NodePath, position: Vec3, size: float = 50.0):
        self.parent = parent
        self.position = position
        self.size = size
        self.time = 0.0
        self.duration = 2.0
        self.alive = True

        # Create root node
        self.root = parent.attachNewNode('explosion')
        self.root.setPos(position)

        # Particles
        self.particles = ParticleSystem(parent, max_particles=50)

        # Emit initial burst
        for _ in range(30):
            vel = Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-0.5, 1.5),
            ).normalized() * random.uniform(30, 80)

            self.particles.emit(
                position=position,
                velocity=vel,
                color=Vec4(1, 0.6, 0.2, 1),  # Orange
                size=3.0,
                lifetime=random.uniform(0.5, 1.5),
            )

        # Smoke particles
        for _ in range(20):
            vel = Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(0.5, 2),
            ).normalized() * random.uniform(10, 30)

            self.particles.emit(
                position=position,
                velocity=vel,
                color=Vec4(0.3, 0.3, 0.3, 0.8),  # Gray smoke
                size=5.0,
                lifetime=random.uniform(1.0, 2.5),
            )

        # Create expanding sphere
        self.sphere_node = self._create_sphere()

    def _create_sphere(self) -> NodePath:
        """Create expanding shockwave sphere."""
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('sphere', format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        # Create sphere wireframe (simple rings)
        segments = 16
        rings = 8

        for ring in range(rings):
            phi = math.pi * ring / rings
            z = math.cos(phi)
            r = math.sin(phi)

            for seg in range(segments + 1):
                theta = 2 * math.pi * seg / segments
                x = r * math.cos(theta)
                y = r * math.sin(theta)

                vertex.addData3f(x, y, z)
                color.addData4f(1, 0.8, 0.3, 0.8)

        lines = GeomLinestrips(Geom.UHStatic)
        for ring in range(rings):
            lines.addConsecutiveVertices(ring * (segments + 1), segments + 1)
            lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('explosion_sphere')
        node.addGeom(geom)

        sphere_np = self.root.attachNewNode(node)
        sphere_np.setTransparency(TransparencyAttrib.MAlpha)
        sphere_np.setRenderModeThickness(2)

        return sphere_np

    def update(self, dt: float):
        """Update explosion effect."""
        self.time += dt

        if self.time > self.duration:
            self.alive = False
            return

        # Update particles
        self.particles.update(dt)

        # Expand sphere
        progress = self.time / self.duration
        scale = self.size * progress
        self.sphere_node.setScale(scale)

        # Fade out
        alpha = 1.0 - progress
        self.sphere_node.setAlphaScale(alpha)

    def cleanup(self):
        """Clean up resources."""
        self.particles.cleanup()
        self.root.removeNode()


class MissileTrail:
    """Smoke trail behind a missile."""

    def __init__(self, parent: NodePath, start_position: Vec3):
        self.parent = parent
        self.trail_points: deque = deque(maxlen=60)  # 1 second at 60fps
        self.trail_points.append(start_position)
        self.trail_node: Optional[NodePath] = None
        self.alive = True
        self.fade_time = 0.0
        self.fading = False

    def update_position(self, position: Vec3):
        """Add new position to trail."""
        if not self.fading:
            self.trail_points.append(Point3(position))

    def start_fade(self):
        """Begin fading out the trail."""
        self.fading = True

    def update(self, dt: float):
        """Update trail geometry."""
        if self.fading:
            self.fade_time += dt
            if self.fade_time > 2.0:
                self.alive = False
                return

        self._rebuild_geometry()

    def _rebuild_geometry(self):
        """Rebuild trail geometry."""
        if self.trail_node:
            self.trail_node.removeNode()
            self.trail_node = None

        if len(self.trail_points) < 2:
            return

        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('missile_trail', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        fade_alpha = 1.0 - (self.fade_time / 2.0) if self.fading else 1.0

        for i, point in enumerate(self.trail_points):
            vertex.addData3f(point)
            # Gray smoke that fades
            alpha = (i / len(self.trail_points)) * 0.6 * fade_alpha
            color.addData4f(0.7, 0.7, 0.7, alpha)

        lines = GeomLinestrips(Geom.UHStatic)
        lines.addConsecutiveVertices(0, len(self.trail_points))
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('trail_geom')
        node.addGeom(geom)

        self.trail_node = self.parent.attachNewNode(node)
        self.trail_node.setTransparency(TransparencyAttrib.MAlpha)
        self.trail_node.setRenderModeThickness(3)

    def cleanup(self):
        """Clean up resources."""
        if self.trail_node:
            self.trail_node.removeNode()


class MuzzleFlash:
    """Gun muzzle flash effect."""

    def __init__(self, parent: NodePath, position: Vec3, direction: Vec3):
        self.parent = parent
        self.time = 0.0
        self.duration = 0.1  # Very quick flash
        self.alive = True

        # Create flash sprite
        self.flash = parent.attachNewNode('muzzle_flash')
        self.flash.setPos(position)

        # Orange flash
        cm = CardMaker('flash')
        cm.setFrame(-2, 2, -2, 2)
        flash_card = self.flash.attachNewNode(cm.generate())
        flash_card.setColor(Vec4(1, 0.8, 0.3, 1))
        flash_card.setTransparency(TransparencyAttrib.MAlpha)
        flash_card.setBillboardPointEye()

        # Create tracer line
        self.tracer = self._create_tracer(position, direction)

    def _create_tracer(self, start: Vec3, direction: Vec3) -> NodePath:
        """Create tracer bullet line."""
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('tracer', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        end = start + direction.normalized() * 100

        vertex.addData3f(start)
        vertex.addData3f(end)
        color.addData4f(1, 1, 0.5, 1)  # Yellow tracer
        color.addData4f(1, 1, 0.5, 0)  # Fade to transparent

        lines = GeomLines(Geom.UHStatic)
        lines.addVertices(0, 1)
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('tracer')
        node.addGeom(geom)

        tracer_np = self.parent.attachNewNode(node)
        tracer_np.setTransparency(TransparencyAttrib.MAlpha)
        tracer_np.setRenderModeThickness(2)

        return tracer_np

    def update(self, dt: float):
        """Update muzzle flash."""
        self.time += dt

        if self.time > self.duration:
            self.alive = False
            return

        # Fade out
        alpha = 1.0 - (self.time / self.duration)
        self.flash.setAlphaScale(alpha)
        self.tracer.setAlphaScale(alpha)

    def cleanup(self):
        """Clean up resources."""
        self.flash.removeNode()
        self.tracer.removeNode()


class LockIndicator:
    """Missile lock cone visualization."""

    def __init__(
        self,
        parent: NodePath,
        attacker_id: int,
        target_id: int,
        lock_progress: float = 0.0,
    ):
        self.parent = parent
        self.attacker_id = attacker_id
        self.target_id = target_id
        self.lock_progress = lock_progress
        self.alive = True

        self.cone_node: Optional[NodePath] = None
        self.line_node: Optional[NodePath] = None

    def update(
        self,
        dt: float,
        attacker_pos: Optional[Vec3] = None,
        target_pos: Optional[Vec3] = None,
        lock_progress: float = 0.0,
    ):
        """Update lock indicator."""
        self.lock_progress = lock_progress

        if attacker_pos is None or target_pos is None:
            self.alive = False
            return

        self._rebuild_geometry(attacker_pos, target_pos)

    def _rebuild_geometry(self, attacker_pos: Vec3, target_pos: Vec3):
        """Rebuild lock indicator geometry."""
        # Clean up old
        if self.line_node:
            self.line_node.removeNode()

        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('lock', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        # Color based on lock progress (yellow -> red)
        r = 1.0
        g = 1.0 - self.lock_progress
        b = 0.0
        alpha = 0.3 + self.lock_progress * 0.5

        # Dashed line between attacker and target
        direction = target_pos - attacker_pos
        distance = direction.length()
        num_dashes = max(2, int(distance / 50))

        for i in range(num_dashes):
            t1 = i / num_dashes
            t2 = (i + 0.5) / num_dashes

            p1 = attacker_pos + direction * t1
            p2 = attacker_pos + direction * t2

            vertex.addData3f(p1)
            vertex.addData3f(p2)
            color.addData4f(r, g, b, alpha)
            color.addData4f(r, g, b, alpha)

        lines = GeomLines(Geom.UHStatic)
        for i in range(0, num_dashes * 2, 2):
            lines.addVertices(i, i + 1)
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('lock_line')
        node.addGeom(geom)

        self.line_node = self.parent.attachNewNode(node)
        self.line_node.setTransparency(TransparencyAttrib.MAlpha)
        self.line_node.setRenderModeThickness(2)

    def cleanup(self):
        """Clean up resources."""
        if self.cone_node:
            self.cone_node.removeNode()
        if self.line_node:
            self.line_node.removeNode()


class DamageIndicator:
    """Damage spark effect when UAV is hit."""

    def __init__(self, parent: NodePath, position: Vec3, damage: float):
        self.parent = parent
        self.time = 0.0
        self.duration = 0.5
        self.alive = True

        # Particles based on damage
        self.particles = ParticleSystem(parent, max_particles=20)

        particle_count = min(20, max(5, int(damage / 5)))

        for _ in range(particle_count):
            vel = Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(0, 1),
            ).normalized() * random.uniform(20, 50)

            self.particles.emit(
                position=position,
                velocity=vel,
                color=Vec4(1, 0.8, 0.2, 1),  # Sparks
                size=2.0,
                lifetime=random.uniform(0.2, 0.5),
            )

    def update(self, dt: float):
        """Update damage effect."""
        self.time += dt

        if self.time > self.duration:
            self.alive = False
            return

        self.particles.update(dt)

    def cleanup(self):
        """Clean up resources."""
        self.particles.cleanup()


class VelocityVector:
    """Velocity/heading arrow for a UAV."""

    def __init__(self, parent: NodePath, uav_id: int):
        self.parent = parent
        self.uav_id = uav_id
        self.arrow_node: Optional[NodePath] = None
        self.visible = False

    def update(self, position: Vec3, velocity: Vec3, visible: bool = True):
        """Update velocity vector."""
        self.visible = visible

        if not visible:
            if self.arrow_node:
                self.arrow_node.hide()
            return

        self._rebuild_geometry(position, velocity)
        if self.arrow_node:
            self.arrow_node.show()

    def _rebuild_geometry(self, position: Vec3, velocity: Vec3):
        """Rebuild arrow geometry."""
        if self.arrow_node:
            self.arrow_node.removeNode()

        speed = velocity.length()
        if speed < 1:
            return

        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('velocity', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        # Arrow shaft
        direction = velocity.normalized()
        length = min(50, speed / 5)  # Scale with speed
        end = position + direction * length

        vertex.addData3f(position)
        vertex.addData3f(end)
        color.addData4f(0, 1, 0, 0.8)
        color.addData4f(0, 1, 0, 0.8)

        # Arrow head
        head_length = length * 0.2
        head_width = head_length * 0.5

        # Get perpendicular vectors for arrow head
        up = Vec3(0, 0, 1)
        right = direction.cross(up).normalized() * head_width

        tip = end
        left_point = end - direction * head_length + right
        right_point = end - direction * head_length - right

        vertex.addData3f(tip)
        vertex.addData3f(left_point)
        vertex.addData3f(tip)
        vertex.addData3f(right_point)
        color.addData4f(0, 1, 0, 0.8)
        color.addData4f(0, 1, 0, 0.8)
        color.addData4f(0, 1, 0, 0.8)
        color.addData4f(0, 1, 0, 0.8)

        lines = GeomLines(Geom.UHStatic)
        lines.addVertices(0, 1)  # Shaft
        lines.addVertices(2, 3)  # Left head
        lines.addVertices(4, 5)  # Right head
        lines.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(lines)

        node = GeomNode('velocity_arrow')
        node.addGeom(geom)

        self.arrow_node = self.parent.attachNewNode(node)
        self.arrow_node.setTransparency(TransparencyAttrib.MAlpha)
        self.arrow_node.setRenderModeThickness(2)

    def cleanup(self):
        """Clean up resources."""
        if self.arrow_node:
            self.arrow_node.removeNode()


class EffectsManager:
    """Manages all visual effects in the scene."""

    def __init__(self, parent: NodePath):
        self.parent = parent
        self.root = parent.attachNewNode('effects')

        # Active effects
        self.explosions: List[ExplosionEffect] = []
        self.missile_trails: Dict[int, MissileTrail] = {}
        self.muzzle_flashes: List[MuzzleFlash] = []
        self.lock_indicators: Dict[Tuple[int, int], LockIndicator] = {}
        self.damage_indicators: List[DamageIndicator] = []
        self.velocity_vectors: Dict[int, VelocityVector] = {}

        # Settings
        self.show_velocity_vectors = False

    def create_explosion(self, position: Vec3, size: float = 50.0):
        """Create explosion at position."""
        effect = ExplosionEffect(self.root, position, size)
        self.explosions.append(effect)

    def create_missile_trail(self, missile_id: int, position: Vec3):
        """Create or update missile trail."""
        if missile_id not in self.missile_trails:
            self.missile_trails[missile_id] = MissileTrail(self.root, position)
        else:
            self.missile_trails[missile_id].update_position(position)

    def end_missile_trail(self, missile_id: int):
        """Start fading missile trail."""
        if missile_id in self.missile_trails:
            self.missile_trails[missile_id].start_fade()

    def create_muzzle_flash(self, position: Vec3, direction: Vec3):
        """Create muzzle flash effect."""
        effect = MuzzleFlash(self.root, position, direction)
        self.muzzle_flashes.append(effect)

    def update_lock_indicator(
        self,
        attacker_id: int,
        target_id: int,
        attacker_pos: Vec3,
        target_pos: Vec3,
        lock_progress: float,
    ):
        """Update or create lock indicator."""
        key = (attacker_id, target_id)

        if key not in self.lock_indicators:
            self.lock_indicators[key] = LockIndicator(
                self.root, attacker_id, target_id, lock_progress
            )

        self.lock_indicators[key].update(
            0, attacker_pos, target_pos, lock_progress
        )

    def remove_lock_indicator(self, attacker_id: int, target_id: int):
        """Remove lock indicator."""
        key = (attacker_id, target_id)
        if key in self.lock_indicators:
            self.lock_indicators[key].cleanup()
            del self.lock_indicators[key]

    def create_damage_effect(self, position: Vec3, damage: float):
        """Create damage spark effect."""
        effect = DamageIndicator(self.root, position, damage)
        self.damage_indicators.append(effect)

    def update_velocity_vector(
        self,
        uav_id: int,
        position: Vec3,
        velocity: Vec3,
    ):
        """Update velocity vector for UAV."""
        if uav_id not in self.velocity_vectors:
            self.velocity_vectors[uav_id] = VelocityVector(self.root, uav_id)

        self.velocity_vectors[uav_id].update(
            position, velocity, self.show_velocity_vectors
        )

    def toggle_velocity_vectors(self):
        """Toggle velocity vector visibility."""
        self.show_velocity_vectors = not self.show_velocity_vectors

    def update(self, dt: float):
        """Update all effects."""
        # Update explosions
        alive_explosions = []
        for effect in self.explosions:
            effect.update(dt)
            if effect.alive:
                alive_explosions.append(effect)
            else:
                effect.cleanup()
        self.explosions = alive_explosions

        # Update missile trails
        dead_trails = []
        for missile_id, trail in self.missile_trails.items():
            trail.update(dt)
            if not trail.alive:
                dead_trails.append(missile_id)
        for missile_id in dead_trails:
            self.missile_trails[missile_id].cleanup()
            del self.missile_trails[missile_id]

        # Update muzzle flashes
        alive_flashes = []
        for effect in self.muzzle_flashes:
            effect.update(dt)
            if effect.alive:
                alive_flashes.append(effect)
            else:
                effect.cleanup()
        self.muzzle_flashes = alive_flashes

        # Update damage indicators
        alive_damage = []
        for effect in self.damage_indicators:
            effect.update(dt)
            if effect.alive:
                alive_damage.append(effect)
            else:
                effect.cleanup()
        self.damage_indicators = alive_damage

    def process_combat_events(self, events: List, uav_positions: Dict[int, Vec3]):
        """Process combat events to create effects."""
        for event in events:
            if event.event_type == 1:  # HIT
                if event.target_id in uav_positions:
                    self.create_damage_effect(
                        uav_positions[event.target_id],
                        event.damage,
                    )

            elif event.event_type == 2:  # KILL
                if event.target_id in uav_positions:
                    self.create_explosion(
                        uav_positions[event.target_id],
                        size=80.0,
                    )

    def cleanup(self):
        """Clean up all effects."""
        for effect in self.explosions:
            effect.cleanup()
        for trail in self.missile_trails.values():
            trail.cleanup()
        for effect in self.muzzle_flashes:
            effect.cleanup()
        for indicator in self.lock_indicators.values():
            indicator.cleanup()
        for effect in self.damage_indicators:
            effect.cleanup()
        for vector in self.velocity_vectors.values():
            vector.cleanup()

        self.root.removeNode()

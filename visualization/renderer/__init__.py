"""
Panda3D Renderer for Dogfight Visualization

Real-time 3D rendering of UAV combat.
"""

from .dogfight_viewer import DogfightViewer, UAVModel, CameraState
from .effects import (
    EffectsManager,
    ExplosionEffect,
    MissileTrail,
    MuzzleFlash,
    LockIndicator,
    DamageIndicator,
    VelocityVector,
    ParticleSystem,
)

__all__ = [
    "DogfightViewer",
    "UAVModel",
    "CameraState",
    # Effects
    "EffectsManager",
    "ExplosionEffect",
    "MissileTrail",
    "MuzzleFlash",
    "LockIndicator",
    "DamageIndicator",
    "VelocityVector",
    "ParticleSystem",
]

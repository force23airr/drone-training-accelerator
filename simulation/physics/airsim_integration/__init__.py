"""
AirSim Integration Module

Provides photorealistic simulation using Microsoft AirSim + Unreal Engine.
Use this for:
- Visualization of trained policies
- Testing in realistic environments (cities, forests, etc.)
- Sensor simulation (cameras, LiDAR, depth)
- Hardware-in-the-loop testing

Training workflow:
1. Train in PyBullet (fast, headless)
2. Export policy
3. Test/visualize in AirSim (photorealistic)
"""

from simulation.physics.airsim_integration.airsim_backend import (
    AirSimBackend,
    AIRSIM_AVAILABLE,
)
from simulation.physics.airsim_integration.airsim_environment import (
    AirSimDroneEnv,
    RealisticEnvironmentConfig,
    EnvironmentType,
)
from simulation.physics.airsim_integration.policy_bridge import (
    PolicyDeploymentBridge,
    deploy_to_airsim,
    visualize_policy,
)

__all__ = [
    "AirSimBackend",
    "AirSimDroneEnv",
    "RealisticEnvironmentConfig",
    "EnvironmentType",
    "PolicyDeploymentBridge",
    "deploy_to_airsim",
    "visualize_policy",
    "AIRSIM_AVAILABLE",
]

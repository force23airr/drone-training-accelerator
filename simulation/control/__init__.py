"""
Control Module

Implements drone control algorithms including:
- PID controllers (cascaded position/velocity/attitude)
- Motor mixing for various configurations
- Trajectory tracking controllers

Based on PX4 autopilot architecture and gym-pybullet-drones.
"""

from simulation.control.pid_controller import (
    PIDGains,
    PIDController,
    CascadedDroneController,
    SimplePIDController,
)

__all__ = [
    "PIDGains",
    "PIDController",
    "CascadedDroneController",
    "SimplePIDController",
]

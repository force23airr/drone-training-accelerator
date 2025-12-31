"""
Physics engine abstractions and implementations.

Supports multiple physics backends with a unified interface.

Available backends:
- PyBulletBackend: Fast, Python-native physics (default)
- GazeboBackend: ROS2/Gazebo integration (stub, future implementation)
- ValidatedDynamics: gym-pybullet-drones integration for research-validated dynamics

Usage:
    from simulation.physics import PyBulletBackend

    backend = PyBulletBackend()
    backend.initialize(render_mode="human")
    # ... use backend for simulation
    backend.shutdown()

For validated drone dynamics:
    from simulation.physics import ValidatedQuadcopterDynamics, DroneType

    dynamics = ValidatedQuadcopterDynamics(drone_type=DroneType.CRAZYFLIE_2X)
    thrust = dynamics.compute_motor_thrust(rpm_array)
"""

from simulation.physics.simulator_backend import SimulatorBackend
from simulation.physics.pybullet_backend import PyBulletBackend

# Validated dynamics (requires gym-pybullet-drones)
from simulation.physics.validated_dynamics import (
    DroneType,
    ValidatedQuadcopterDynamics,
    create_crazyflie_dynamics,
    create_race_drone_dynamics,
    GYM_PYBULLET_DRONES_AVAILABLE,
)

# Conditionally import ValidatedDroneEnvironment
if GYM_PYBULLET_DRONES_AVAILABLE:
    from simulation.physics.validated_dynamics import ValidatedDroneEnvironment

# Gazebo backend is a stub - will raise NotImplementedError
# from simulation.physics.gazebo_backend import GazeboBackend

__all__ = [
    # Core backends
    "SimulatorBackend",
    "PyBulletBackend",
    "get_backend",
    # Validated dynamics
    "DroneType",
    "ValidatedQuadcopterDynamics",
    "create_crazyflie_dynamics",
    "create_race_drone_dynamics",
    "GYM_PYBULLET_DRONES_AVAILABLE",
]

if GYM_PYBULLET_DRONES_AVAILABLE:
    __all__.append("ValidatedDroneEnvironment")


def get_backend(backend_name: str = "pybullet") -> SimulatorBackend:
    """
    Factory function to get a physics backend by name.

    Args:
        backend_name: 'pybullet' or 'gazebo'

    Returns:
        Instantiated backend

    Raises:
        ValueError: If backend name is unknown
        NotImplementedError: If backend is not yet implemented
    """
    backends = {
        "pybullet": PyBulletBackend,
        # "gazebo": GazeboBackend,
    }

    if backend_name not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")

    return backends[backend_name]()

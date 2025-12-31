"""
Simulator Backend Abstract Base Class

Defines the interface for physics simulation backends.
Allows swapping between PyBullet, Gazebo, Isaac Sim, etc.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class SimulatorBackend(ABC):
    """
    Abstract base class for physics simulation backends.

    This interface allows the drone environment to work with
    different physics engines without code changes.
    """

    @abstractmethod
    def initialize(self, render_mode: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the physics simulation.

        Args:
            render_mode: 'human' for GUI, 'rgb_array' for offscreen, None for headless
            **kwargs: Backend-specific initialization parameters
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up and disconnect from the simulation."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        pass

    @abstractmethod
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance the simulation by one timestep.

        Args:
            dt: Timestep duration (uses default if None)
        """
        pass

    @abstractmethod
    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        """
        Set the gravity vector.

        Args:
            gravity: Gravity vector (x, y, z) in m/s²
        """
        pass

    @abstractmethod
    def set_timestep(self, dt: float) -> None:
        """
        Set the simulation timestep.

        Args:
            dt: Timestep in seconds
        """
        pass

    # =========================================================================
    # Object Management
    # =========================================================================

    @abstractmethod
    def load_urdf(
        self,
        urdf_path: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        **kwargs
    ) -> int:
        """
        Load a URDF model into the simulation.

        Args:
            urdf_path: Path to URDF file
            position: Initial position (x, y, z)
            orientation: Initial orientation as quaternion (x, y, z, w)

        Returns:
            Object ID for referencing the loaded model
        """
        pass

    @abstractmethod
    def create_box(
        self,
        half_extents: Tuple[float, float, float],
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """
        Create a box primitive in the simulation.

        Args:
            half_extents: Half dimensions (x, y, z)
            position: Position (x, y, z)
            mass: Mass in kg (0 for static)
            color: RGBA color

        Returns:
            Object ID
        """
        pass

    @abstractmethod
    def create_sphere(
        self,
        radius: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """
        Create a sphere primitive in the simulation.

        Args:
            radius: Sphere radius
            position: Position (x, y, z)
            mass: Mass in kg (0 for static)
            color: RGBA color

        Returns:
            Object ID
        """
        pass

    @abstractmethod
    def create_cylinder(
        self,
        radius: float,
        height: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """
        Create a cylinder primitive in the simulation.

        Args:
            radius: Cylinder radius
            height: Cylinder height
            position: Position (x, y, z)
            mass: Mass in kg (0 for static)
            color: RGBA color

        Returns:
            Object ID
        """
        pass

    @abstractmethod
    def remove_object(self, object_id: int) -> None:
        """
        Remove an object from the simulation.

        Args:
            object_id: ID of object to remove
        """
        pass

    # =========================================================================
    # State Queries
    # =========================================================================

    @abstractmethod
    def get_position(self, object_id: int) -> np.ndarray:
        """
        Get object position.

        Args:
            object_id: Object ID

        Returns:
            Position array [x, y, z]
        """
        pass

    @abstractmethod
    def get_orientation(self, object_id: int) -> np.ndarray:
        """
        Get object orientation as quaternion.

        Args:
            object_id: Object ID

        Returns:
            Quaternion array [x, y, z, w]
        """
        pass

    @abstractmethod
    def get_velocity(self, object_id: int) -> np.ndarray:
        """
        Get object linear velocity.

        Args:
            object_id: Object ID

        Returns:
            Velocity array [vx, vy, vz]
        """
        pass

    @abstractmethod
    def get_angular_velocity(self, object_id: int) -> np.ndarray:
        """
        Get object angular velocity.

        Args:
            object_id: Object ID

        Returns:
            Angular velocity array [wx, wy, wz]
        """
        pass

    @abstractmethod
    def get_state(self, object_id: int) -> Dict[str, np.ndarray]:
        """
        Get complete object state.

        Args:
            object_id: Object ID

        Returns:
            Dict with 'position', 'orientation', 'velocity', 'angular_velocity'
        """
        pass

    # =========================================================================
    # State Setting
    # =========================================================================

    @abstractmethod
    def set_position(
        self,
        object_id: int,
        position: Tuple[float, float, float]
    ) -> None:
        """
        Set object position.

        Args:
            object_id: Object ID
            position: New position (x, y, z)
        """
        pass

    @abstractmethod
    def set_orientation(
        self,
        object_id: int,
        orientation: Tuple[float, float, float, float]
    ) -> None:
        """
        Set object orientation.

        Args:
            object_id: Object ID
            orientation: New orientation quaternion (x, y, z, w)
        """
        pass

    @abstractmethod
    def set_velocity(
        self,
        object_id: int,
        velocity: Tuple[float, float, float]
    ) -> None:
        """
        Set object linear velocity.

        Args:
            object_id: Object ID
            velocity: New velocity (vx, vy, vz)
        """
        pass

    # =========================================================================
    # Force Application
    # =========================================================================

    @abstractmethod
    def apply_force(
        self,
        object_id: int,
        force: Tuple[float, float, float],
        position: Tuple[float, float, float] = (0, 0, 0),
        link_id: int = -1,
        frame: str = "world"
    ) -> None:
        """
        Apply force to an object.

        Args:
            object_id: Object ID
            force: Force vector (fx, fy, fz) in Newtons
            position: Point of application relative to link
            link_id: Link index (-1 for base)
            frame: 'world' or 'link' frame
        """
        pass

    @abstractmethod
    def apply_torque(
        self,
        object_id: int,
        torque: Tuple[float, float, float],
        link_id: int = -1
    ) -> None:
        """
        Apply torque to an object.

        Args:
            object_id: Object ID
            torque: Torque vector (tx, ty, tz) in Nm
            link_id: Link index (-1 for base)
        """
        pass

    # =========================================================================
    # Collision Detection
    # =========================================================================

    @abstractmethod
    def get_contacts(self, object_id: int) -> List[Dict[str, Any]]:
        """
        Get contact points for an object.

        Args:
            object_id: Object ID

        Returns:
            List of contact info dicts with keys:
            - 'other_id': ID of other object
            - 'position': Contact position
            - 'normal': Contact normal
            - 'force': Contact force magnitude
        """
        pass

    @abstractmethod
    def check_collision(self, object_id_a: int, object_id_b: int) -> bool:
        """
        Check if two objects are in collision.

        Args:
            object_id_a: First object ID
            object_id_b: Second object ID

        Returns:
            True if objects are colliding
        """
        pass

    # =========================================================================
    # Rendering
    # =========================================================================

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current frame.

        Returns:
            RGB image array if render_mode is 'rgb_array', else None
        """
        pass

    @abstractmethod
    def get_camera_image(
        self,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get camera image from specified viewpoint.

        Args:
            width: Image width
            height: Image height
            view_matrix: 4x4 view matrix
            projection_matrix: 4x4 projection matrix

        Returns:
            Tuple of (rgb_image, depth_image, segmentation_mask)
        """
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @abstractmethod
    def euler_to_quaternion(
        self,
        euler: Tuple[float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Convert Euler angles to quaternion.

        Args:
            euler: Euler angles (roll, pitch, yaw) in radians

        Returns:
            Quaternion (x, y, z, w)
        """
        pass

    @abstractmethod
    def quaternion_to_euler(
        self,
        quaternion: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles.

        Args:
            quaternion: Quaternion (x, y, z, w)

        Returns:
            Euler angles (roll, pitch, yaw) in radians
        """
        pass

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Get current simulation timestep."""
        pass

    @property
    @abstractmethod
    def time(self) -> float:
        """Get current simulation time."""
        pass

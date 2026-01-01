"""
AirSim Backend Implementation

Provides photorealistic drone simulation using Microsoft AirSim + Unreal Engine.
This backend is designed for visualization and testing, not training (use PyBullet for training).

Requirements:
- AirSim simulator running (download from https://github.com/microsoft/AirSim/releases)
- pip install airsim

Usage:
    backend = AirSimBackend()
    backend.initialize(render_mode="human", environment="CityEnvironment")
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import time

from simulation.physics.simulator_backend import SimulatorBackend

# Check if AirSim is available
try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None


class AirSimBackend(SimulatorBackend):
    """
    AirSim physics/rendering backend for photorealistic drone simulation.

    This backend connects to a running AirSim instance (Unreal Engine).
    Use for:
    - Visualizing trained policies
    - Testing in realistic environments
    - Sensor simulation (cameras, LiDAR, depth)
    - Demo/presentation purposes

    NOT recommended for RL training (too slow). Use PyBulletBackend for training.
    """

    # Pre-built environments available in AirSim
    AVAILABLE_ENVIRONMENTS = [
        "Blocks",           # Simple blocks environment (default)
        "CityEnviron",      # Urban city streets
        "Neighborhood",     # Suburban neighborhood
        "LandscapeMountains",  # Mountain terrain
        "Africa",           # African savanna
        "Forest",           # Dense forest
        "TrapCam",          # Wildlife camera environment
        "Warehouse",        # Indoor warehouse
        "Building99",       # Microsoft Building 99
        "Custom",           # User-defined Unreal project
    ]

    def __init__(self):
        """Initialize AirSim backend (not connected yet)."""
        if not AIRSIM_AVAILABLE:
            raise ImportError(
                "AirSim not installed. Install with: pip install airsim\n"
                "Also need AirSim simulator running: "
                "https://github.com/microsoft/AirSim/releases"
            )

        self.client: Optional[airsim.MultirotorClient] = None
        self._timestep = 1/240  # Match PyBullet default
        self._time = 0.0
        self._render_mode: Optional[str] = None
        self._drone_name = "Drone0"
        self._objects: Dict[int, str] = {}  # object_id -> AirSim object name
        self._next_object_id = 1
        self._initialized = False

    def initialize(
        self,
        render_mode: Optional[str] = None,
        environment: str = "Blocks",
        drone_name: str = "Drone0",
        **kwargs
    ) -> None:
        """
        Connect to AirSim simulator.

        Args:
            render_mode: 'human' for visual, 'rgb_array' for image capture
            environment: Which AirSim environment to use (must match running instance)
            drone_name: Name of drone in AirSim settings
        """
        self._render_mode = render_mode
        self._drone_name = drone_name

        # Connect to AirSim
        print(f"Connecting to AirSim...")
        print(f"  Environment: {environment}")
        print(f"  Drone: {drone_name}")

        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True, self._drone_name)
            self.client.armDisarm(True, self._drone_name)

            self._initialized = True
            print("Connected to AirSim successfully!")

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to AirSim: {e}\n"
                "Make sure AirSim simulator is running.\n"
                "Download: https://github.com/microsoft/AirSim/releases"
            )

    def shutdown(self) -> None:
        """Disconnect from AirSim."""
        if self.client:
            self.client.armDisarm(False, self._drone_name)
            self.client.enableApiControl(False, self._drone_name)
            self.client = None
        self._initialized = False
        print("Disconnected from AirSim")

    def reset(self) -> None:
        """Reset drone to initial position."""
        if not self._initialized:
            raise RuntimeError("AirSim not initialized")

        self.client.reset()
        self.client.enableApiControl(True, self._drone_name)
        self.client.armDisarm(True, self._drone_name)
        self._time = 0.0

        # Takeoff to hover
        self.client.takeoffAsync(vehicle_name=self._drone_name).join()

    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation (AirSim runs in real-time or sim time).

        Args:
            dt: Timestep (used for timing, AirSim manages its own physics)
        """
        if dt is None:
            dt = self._timestep
        self._time += dt

        # AirSim runs continuously - we just track time
        # For synchronized stepping, would need to use simPause/simContinue

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        """Set gravity (AirSim uses Unreal Engine gravity settings)."""
        # AirSim gravity is set in Unreal Engine project settings
        # Cannot be changed via API
        pass

    def set_timestep(self, dt: float) -> None:
        """Set simulation timestep."""
        self._timestep = dt

    # =========================================================================
    # Object Management
    # =========================================================================

    def load_urdf(
        self,
        urdf_path: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        **kwargs
    ) -> int:
        """
        Load a model (AirSim doesn't use URDF directly).

        For AirSim, models are placed in the Unreal project.
        This method is for compatibility - returns drone ID.
        """
        # Return the drone as object 0
        object_id = 0
        self._objects[object_id] = self._drone_name
        return object_id

    def create_box(
        self,
        half_extents: Tuple[float, float, float],
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """
        Create a box obstacle in AirSim.

        Note: AirSim has limited runtime object spawning.
        For complex environments, build them in Unreal Editor.
        """
        object_id = self._next_object_id
        self._next_object_id += 1

        # AirSim simSpawnObject for runtime spawning (limited)
        scale = airsim.Vector3r(
            half_extents[0] * 2,
            half_extents[1] * 2,
            half_extents[2] * 2
        )
        pose = airsim.Pose(
            airsim.Vector3r(position[0], position[1], -position[2]),  # NED coordinates
            airsim.Quaternionr(orientation[0], orientation[1], orientation[2], orientation[3])
            if 'orientation' in kwargs else airsim.Quaternionr(0, 0, 0, 1)
        )

        obj_name = f"Box_{object_id}"
        try:
            self.client.simSpawnObject(
                obj_name,
                "Cube",  # Built-in Unreal primitive
                pose,
                scale,
                physics_enabled=mass > 0
            )
            self._objects[object_id] = obj_name
        except Exception as e:
            print(f"Warning: Could not spawn box: {e}")

        return object_id

    def create_sphere(
        self,
        radius: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """Create a sphere obstacle in AirSim."""
        object_id = self._next_object_id
        self._next_object_id += 1

        scale = airsim.Vector3r(radius * 2, radius * 2, radius * 2)
        pose = airsim.Pose(
            airsim.Vector3r(position[0], position[1], -position[2]),
            airsim.Quaternionr(0, 0, 0, 1)
        )

        obj_name = f"Sphere_{object_id}"
        try:
            self.client.simSpawnObject(obj_name, "Sphere", pose, scale)
            self._objects[object_id] = obj_name
        except Exception as e:
            print(f"Warning: Could not spawn sphere: {e}")

        return object_id

    def create_cylinder(
        self,
        radius: float,
        height: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """Create a cylinder obstacle in AirSim."""
        object_id = self._next_object_id
        self._next_object_id += 1

        scale = airsim.Vector3r(radius * 2, radius * 2, height)
        pose = airsim.Pose(
            airsim.Vector3r(position[0], position[1], -position[2]),
            airsim.Quaternionr(0, 0, 0, 1)
        )

        obj_name = f"Cylinder_{object_id}"
        try:
            self.client.simSpawnObject(obj_name, "Cylinder", pose, scale)
            self._objects[object_id] = obj_name
        except Exception as e:
            print(f"Warning: Could not spawn cylinder: {e}")

        return object_id

    def remove_object(self, object_id: int) -> None:
        """Remove an object from AirSim."""
        if object_id in self._objects:
            obj_name = self._objects[object_id]
            try:
                self.client.simDestroyObject(obj_name)
            except:
                pass
            del self._objects[object_id]

    # =========================================================================
    # State Queries
    # =========================================================================

    def get_position(self, object_id: int = 0) -> np.ndarray:
        """Get drone position in world coordinates."""
        state = self.client.getMultirotorState(vehicle_name=self._drone_name)
        pos = state.kinematics_estimated.position
        # Convert from NED to standard coordinates
        return np.array([pos.x_val, pos.y_val, -pos.z_val])

    def get_orientation(self, object_id: int = 0) -> np.ndarray:
        """Get drone orientation as quaternion."""
        state = self.client.getMultirotorState(vehicle_name=self._drone_name)
        q = state.kinematics_estimated.orientation
        return np.array([q.x_val, q.y_val, q.z_val, q.w_val])

    def get_velocity(self, object_id: int = 0) -> np.ndarray:
        """Get drone linear velocity."""
        state = self.client.getMultirotorState(vehicle_name=self._drone_name)
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, -vel.z_val])

    def get_angular_velocity(self, object_id: int = 0) -> np.ndarray:
        """Get drone angular velocity."""
        state = self.client.getMultirotorState(vehicle_name=self._drone_name)
        ang_vel = state.kinematics_estimated.angular_velocity
        return np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])

    def get_state(self, object_id: int = 0) -> Dict[str, np.ndarray]:
        """Get complete drone state."""
        return {
            'position': self.get_position(object_id),
            'orientation': self.get_orientation(object_id),
            'velocity': self.get_velocity(object_id),
            'angular_velocity': self.get_angular_velocity(object_id),
        }

    # =========================================================================
    # State Setting
    # =========================================================================

    def set_position(
        self,
        object_id: int,
        position: Tuple[float, float, float]
    ) -> None:
        """Set drone position."""
        pose = self.client.simGetVehiclePose(vehicle_name=self._drone_name)
        pose.position = airsim.Vector3r(position[0], position[1], -position[2])
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self._drone_name)

    def set_orientation(
        self,
        object_id: int,
        orientation: Tuple[float, float, float, float]
    ) -> None:
        """Set drone orientation."""
        pose = self.client.simGetVehiclePose(vehicle_name=self._drone_name)
        pose.orientation = airsim.Quaternionr(
            orientation[0], orientation[1], orientation[2], orientation[3]
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self._drone_name)

    def set_velocity(
        self,
        object_id: int,
        velocity: Tuple[float, float, float]
    ) -> None:
        """Set drone velocity (via movement command)."""
        # AirSim uses velocity commands rather than direct velocity setting
        self.client.moveByVelocityAsync(
            velocity[0], velocity[1], -velocity[2],
            duration=0.1,
            vehicle_name=self._drone_name
        )

    # =========================================================================
    # Force Application (Motor Control)
    # =========================================================================

    def apply_force(
        self,
        object_id: int,
        force: Tuple[float, float, float],
        position: Tuple[float, float, float] = (0, 0, 0),
        link_id: int = -1,
        frame: str = "world"
    ) -> None:
        """
        Apply force to drone (converted to motor commands).

        AirSim doesn't support direct force application.
        Use motor commands or velocity/position control instead.
        """
        # Convert force to approximate thrust command
        # This is simplified - real implementation would need proper conversion
        total_force = np.linalg.norm(force)
        if total_force > 0:
            # Approximate: move in force direction
            direction = np.array(force) / total_force
            self.client.moveByVelocityAsync(
                direction[0] * 2,
                direction[1] * 2,
                -direction[2] * 2,
                duration=0.1,
                vehicle_name=self._drone_name
            )

    def apply_torque(
        self,
        object_id: int,
        torque: Tuple[float, float, float],
        link_id: int = -1
    ) -> None:
        """Apply torque (converted to attitude command)."""
        # Convert torque to attitude rate command
        # Simplified implementation
        pass

    def apply_motor_commands(
        self,
        motor_speeds: np.ndarray,
        duration: float = 0.1
    ) -> None:
        """
        Apply direct motor speed commands.

        Args:
            motor_speeds: Array of motor speeds (PWM values 0-1)
            duration: Command duration
        """
        # AirSim low-level motor control
        self.client.moveByMotorPWMsAsync(
            motor_speeds[0],
            motor_speeds[1],
            motor_speeds[2],
            motor_speeds[3],
            duration,
            vehicle_name=self._drone_name
        )

    # =========================================================================
    # High-Level Control Commands
    # =========================================================================

    def move_to_position(
        self,
        position: Tuple[float, float, float],
        velocity: float = 5.0
    ) -> None:
        """Move drone to target position."""
        self.client.moveToPositionAsync(
            position[0], position[1], -position[2],
            velocity,
            vehicle_name=self._drone_name
        ).join()

    def hover(self) -> None:
        """Command drone to hover in place."""
        self.client.hoverAsync(vehicle_name=self._drone_name).join()

    def land(self) -> None:
        """Land the drone."""
        self.client.landAsync(vehicle_name=self._drone_name).join()

    # =========================================================================
    # Collision Detection
    # =========================================================================

    def get_contacts(self, object_id: int = 0) -> List[Dict[str, Any]]:
        """Get collision info."""
        collision = self.client.simGetCollisionInfo(vehicle_name=self._drone_name)
        if collision.has_collided:
            return [{
                'other_id': -1,  # Unknown in AirSim
                'position': np.array([
                    collision.position.x_val,
                    collision.position.y_val,
                    -collision.position.z_val
                ]),
                'normal': np.array([
                    collision.normal.x_val,
                    collision.normal.y_val,
                    collision.normal.z_val
                ]),
                'force': collision.penetration_depth
            }]
        return []

    def check_collision(self, object_id_a: int, object_id_b: int) -> bool:
        """Check if drone has collided."""
        collision = self.client.simGetCollisionInfo(vehicle_name=self._drone_name)
        return collision.has_collided

    # =========================================================================
    # Rendering & Cameras
    # =========================================================================

    def render(self) -> Optional[np.ndarray]:
        """Get current camera frame."""
        if self._render_mode == 'rgb_array':
            return self.get_front_camera_image()
        return None

    def get_front_camera_image(
        self,
        width: int = 640,
        height: int = 480
    ) -> np.ndarray:
        """Get front camera RGB image."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self._drone_name)

        if responses:
            response = responses[0]
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img.reshape(response.height, response.width, 3)
            return img
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_depth_image(self) -> np.ndarray:
        """Get depth camera image."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True)
        ], vehicle_name=self._drone_name)

        if responses:
            response = responses[0]
            depth = airsim.list_to_2d_float_array(
                response.image_data_float,
                response.width,
                response.height
            )
            return depth
        return np.zeros((480, 640), dtype=np.float32)

    def get_segmentation_image(self) -> np.ndarray:
        """Get segmentation mask."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
        ], vehicle_name=self._drone_name)

        if responses:
            response = responses[0]
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img.reshape(response.height, response.width, 3)
            return img
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_camera_image(
        self,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get camera images (RGB, depth, segmentation)."""
        rgb = self.get_front_camera_image(width, height)
        depth = self.get_depth_image()
        segmentation = self.get_segmentation_image()
        return rgb, depth, segmentation

    def get_lidar_data(self) -> np.ndarray:
        """Get LiDAR point cloud."""
        lidar_data = self.client.getLidarData(vehicle_name=self._drone_name)
        if len(lidar_data.point_cloud) >= 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 3)
            return points
        return np.zeros((0, 3), dtype=np.float32)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def euler_to_quaternion(
        self,
        euler: Tuple[float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion."""
        q = airsim.to_quaternion(euler[1], euler[0], euler[2])  # pitch, roll, yaw
        return (q.x_val, q.y_val, q.z_val, q.w_val)

    def quaternion_to_euler(
        self,
        quaternion: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles."""
        q = airsim.Quaternionr(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        return (roll, pitch, yaw)

    @property
    def timestep(self) -> float:
        """Get simulation timestep."""
        return self._timestep

    @property
    def time(self) -> float:
        """Get current simulation time."""
        return self._time

    # =========================================================================
    # Weather Control (AirSim-specific)
    # =========================================================================

    def set_weather(
        self,
        rain: float = 0.0,
        snow: float = 0.0,
        fog: float = 0.0,
        dust: float = 0.0,
    ) -> None:
        """
        Set weather conditions in AirSim.

        Args:
            rain: Rain intensity 0-1
            snow: Snow intensity 0-1
            fog: Fog density 0-1
            dust: Dust intensity 0-1
        """
        self.client.simEnableWeather(True)

        if rain > 0:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain)
        if snow > 0:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, snow)
        if fog > 0:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)
        if dust > 0:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, dust)

    def set_time_of_day(
        self,
        enabled: bool = True,
        start_datetime: str = "2024-06-21 12:00:00",
        celestial_clock_speed: float = 1.0
    ) -> None:
        """
        Set time of day for lighting.

        Args:
            enabled: Enable time-of-day simulation
            start_datetime: Starting date/time
            celestial_clock_speed: Speed multiplier for day/night cycle
        """
        self.client.simSetTimeOfDay(
            enabled,
            start_datetime=start_datetime,
            is_start_datetime_dst=True,
            celestial_clock_speed=celestial_clock_speed,
            update_interval_secs=60,
            move_sun=True
        )

    def set_wind(self, velocity: Tuple[float, float, float]) -> None:
        """Set wind velocity vector."""
        wind = airsim.Vector3r(velocity[0], velocity[1], velocity[2])
        self.client.simSetWind(wind)

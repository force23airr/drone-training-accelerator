"""
PyBullet Physics Backend

Implementation of SimulatorBackend using PyBullet physics engine.
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np

import pybullet as p
import pybullet_data

from simulation.physics.simulator_backend import SimulatorBackend


class PyBulletBackend(SimulatorBackend):
    """
    PyBullet implementation of the simulator backend.

    PyBullet is a fast, lightweight physics engine well-suited for
    RL training due to its speed and Python integration.
    """

    def __init__(self):
        """Initialize PyBullet backend."""
        self._client_id: Optional[int] = None
        self._timestep: float = 1.0 / 240.0
        self._time: float = 0.0
        self._render_mode: Optional[str] = None
        self._ground_id: Optional[int] = None

    def initialize(
        self,
        render_mode: Optional[str] = None,
        timestep: float = 1.0 / 240.0,
        gravity: Tuple[float, float, float] = (0, 0, -9.81),
        **kwargs
    ) -> None:
        """Initialize PyBullet simulation."""
        self._render_mode = render_mode
        self._timestep = timestep

        # Disconnect if already connected
        if self._client_id is not None:
            p.disconnect(self._client_id)

        # Connect to PyBullet
        if render_mode == "human":
            self._client_id = p.connect(p.GUI)
            # Configure GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._client_id)
        else:
            self._client_id = p.connect(p.DIRECT)

        # Set up physics parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity, physicsClientId=self._client_id)
        p.setTimeStep(timestep, physicsClientId=self._client_id)

        # Set solver parameters for stability
        p.setPhysicsEngineParameter(
            fixedTimeStep=timestep,
            numSolverIterations=50,
            numSubSteps=4,
            physicsClientId=self._client_id
        )

        # Load ground plane
        self._ground_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self._client_id
        )

        self._time = 0.0

    def shutdown(self) -> None:
        """Disconnect from PyBullet."""
        if self._client_id is not None:
            p.disconnect(self._client_id)
            self._client_id = None

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        p.resetSimulation(physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        p.setTimeStep(self._timestep, physicsClientId=self._client_id)

        # Reload ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._ground_id = p.loadURDF("plane.urdf", physicsClientId=self._client_id)

        self._time = 0.0

    def step(self, dt: Optional[float] = None) -> None:
        """Step simulation forward."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        p.stepSimulation(physicsClientId=self._client_id)
        self._time += self._timestep

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        """Set gravity vector."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        p.setGravity(*gravity, physicsClientId=self._client_id)

    def set_timestep(self, dt: float) -> None:
        """Set simulation timestep."""
        self._timestep = dt
        if self._client_id is not None:
            p.setTimeStep(dt, physicsClientId=self._client_id)

    # =========================================================================
    # Object Management
    # =========================================================================

    def load_urdf(
        self,
        urdf_path: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        use_fixed_base: bool = False,
        global_scaling: float = 1.0,
        **kwargs
    ) -> int:
        """Load URDF model."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        return p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=use_fixed_base,
            globalScaling=global_scaling,
            physicsClientId=self._client_id
        )

    def create_box(
        self,
        half_extents: Tuple[float, float, float],
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """Create a box primitive."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._client_id
        )

        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            physicsClientId=self._client_id
        )

    def create_sphere(
        self,
        radius: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """Create a sphere primitive."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self._client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self._client_id
        )

        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            physicsClientId=self._client_id
        )

    def create_cylinder(
        self,
        radius: float,
        height: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        **kwargs
    ) -> int:
        """Create a cylinder primitive."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self._client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=self._client_id
        )

        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            physicsClientId=self._client_id
        )

    def remove_object(self, object_id: int) -> None:
        """Remove object from simulation."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        p.removeBody(object_id, physicsClientId=self._client_id)

    # =========================================================================
    # State Queries
    # =========================================================================

    def get_position(self, object_id: int) -> np.ndarray:
        """Get object position."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        pos, _ = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self._client_id
        )
        return np.array(pos)

    def get_orientation(self, object_id: int) -> np.ndarray:
        """Get object orientation as quaternion."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        _, orn = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self._client_id
        )
        return np.array(orn)

    def get_velocity(self, object_id: int) -> np.ndarray:
        """Get object linear velocity."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        vel, _ = p.getBaseVelocity(
            object_id, physicsClientId=self._client_id
        )
        return np.array(vel)

    def get_angular_velocity(self, object_id: int) -> np.ndarray:
        """Get object angular velocity."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        _, ang_vel = p.getBaseVelocity(
            object_id, physicsClientId=self._client_id
        )
        return np.array(ang_vel)

    def get_state(self, object_id: int) -> Dict[str, np.ndarray]:
        """Get complete object state."""
        return {
            "position": self.get_position(object_id),
            "orientation": self.get_orientation(object_id),
            "velocity": self.get_velocity(object_id),
            "angular_velocity": self.get_angular_velocity(object_id),
        }

    # =========================================================================
    # State Setting
    # =========================================================================

    def set_position(
        self,
        object_id: int,
        position: Tuple[float, float, float]
    ) -> None:
        """Set object position."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        _, orn = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self._client_id
        )
        p.resetBasePositionAndOrientation(
            object_id, position, orn, physicsClientId=self._client_id
        )

    def set_orientation(
        self,
        object_id: int,
        orientation: Tuple[float, float, float, float]
    ) -> None:
        """Set object orientation."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        pos, _ = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self._client_id
        )
        p.resetBasePositionAndOrientation(
            object_id, pos, orientation, physicsClientId=self._client_id
        )

    def set_velocity(
        self,
        object_id: int,
        velocity: Tuple[float, float, float]
    ) -> None:
        """Set object linear velocity."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")
        _, ang_vel = p.getBaseVelocity(
            object_id, physicsClientId=self._client_id
        )
        p.resetBaseVelocity(
            object_id, velocity, ang_vel, physicsClientId=self._client_id
        )

    # =========================================================================
    # Force Application
    # =========================================================================

    def apply_force(
        self,
        object_id: int,
        force: Tuple[float, float, float],
        position: Tuple[float, float, float] = (0, 0, 0),
        link_id: int = -1,
        frame: str = "world"
    ) -> None:
        """Apply force to object."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        flags = p.WORLD_FRAME if frame == "world" else p.LINK_FRAME

        p.applyExternalForce(
            object_id,
            link_id,
            forceObj=force,
            posObj=position,
            flags=flags,
            physicsClientId=self._client_id
        )

    def apply_torque(
        self,
        object_id: int,
        torque: Tuple[float, float, float],
        link_id: int = -1
    ) -> None:
        """Apply torque to object."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        p.applyExternalTorque(
            object_id,
            link_id,
            torqueObj=torque,
            flags=p.WORLD_FRAME,
            physicsClientId=self._client_id
        )

    # =========================================================================
    # Collision Detection
    # =========================================================================

    def get_contacts(self, object_id: int) -> List[Dict[str, Any]]:
        """Get contact points for an object."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        contacts = p.getContactPoints(
            bodyA=object_id,
            physicsClientId=self._client_id
        )

        result = []
        for contact in contacts:
            result.append({
                "other_id": contact[2],
                "position": np.array(contact[5]),
                "normal": np.array(contact[7]),
                "force": contact[9],
            })

        return result

    def check_collision(self, object_id_a: int, object_id_b: int) -> bool:
        """Check if two objects are in collision."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        contacts = p.getContactPoints(
            bodyA=object_id_a,
            bodyB=object_id_b,
            physicsClientId=self._client_id
        )

        return len(contacts) > 0

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(self) -> Optional[np.ndarray]:
        """Render current frame."""
        if self._render_mode != "rgb_array":
            return None

        # Default camera view
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 1],
            distance=3,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._client_id
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.33,
            nearVal=0.1,
            farVal=100,
            physicsClientId=self._client_id
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self._client_id
        )

        return np.array(rgba)[:, :, :3]

    def get_camera_image(
        self,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get camera image from specified viewpoint."""
        if self._client_id is None:
            raise RuntimeError("Backend not initialized")

        _, _, rgba, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix.flatten().tolist(),
            projectionMatrix=projection_matrix.flatten().tolist(),
            physicsClientId=self._client_id
        )

        rgb = np.array(rgba)[:, :, :3]
        depth = np.array(depth)
        seg = np.array(seg)

        return rgb, depth, seg

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def euler_to_quaternion(
        self,
        euler: Tuple[float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion."""
        return p.getQuaternionFromEuler(euler)

    def quaternion_to_euler(
        self,
        quaternion: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles."""
        return p.getEulerFromQuaternion(quaternion)

    @property
    def timestep(self) -> float:
        """Get current timestep."""
        return self._timestep

    @property
    def time(self) -> float:
        """Get current simulation time."""
        return self._time

    @property
    def client_id(self) -> Optional[int]:
        """Get PyBullet client ID."""
        return self._client_id

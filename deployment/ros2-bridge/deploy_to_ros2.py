"""
ROS2 Deployment Bridge

Bridge for deploying trained RL models to ROS2-based drone systems.
Supports real-time inference and integration with PX4/ArduPilot via ROS2.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np

# ROS2 imports are optional - only required when deploying
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Float32MultiArray
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from sensor_msgs.msg import Imu
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Placeholder for type hints


class ROS2DeploymentBridge:
    """
    Bridge for deploying trained models to ROS2 drone systems.

    This class provides:
    - ROS2 node management
    - Sensor data subscription and processing
    - Control command publishing
    - Real-time model inference

    Usage:
        bridge = ROS2DeploymentBridge(model_path="path/to/model.zip")
        bridge.start()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        node_name: str = "drone_policy_node",
        control_topic: str = "/drone/control",
        pose_topic: str = "/drone/pose",
        velocity_topic: str = "/drone/velocity",
        imu_topic: str = "/drone/imu",
        control_frequency: float = 50.0,
    ):
        """
        Initialize ROS2 deployment bridge.

        Args:
            model_path: Path to trained model file (.zip)
            node_name: Name for ROS2 node
            control_topic: Topic for publishing control commands
            pose_topic: Topic for pose subscription
            velocity_topic: Topic for velocity subscription
            imu_topic: Topic for IMU subscription
            control_frequency: Control loop frequency in Hz
        """
        if not ROS2_AVAILABLE:
            raise ImportError(
                "ROS2 (rclpy) is not available. Install ROS2 and rclpy to use deployment features."
            )

        self.model_path = model_path
        self.node_name = node_name
        self.control_topic = control_topic
        self.pose_topic = pose_topic
        self.velocity_topic = velocity_topic
        self.imu_topic = imu_topic
        self.control_frequency = control_frequency

        # State
        self._model = None
        self._node = None
        self._current_obs = np.zeros(13, dtype=np.float32)
        self._last_pose = None
        self._last_velocity = None
        self._last_imu = None

        # Load model if provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load a trained model for inference.

        Args:
            model_path: Path to model file
        """
        from stable_baselines3 import PPO, SAC, TD3

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Try loading with different algorithms
        for AlgoClass in [PPO, SAC, TD3]:
            try:
                self._model = AlgoClass.load(str(path))
                print(f"Loaded model from: {path}")
                return
            except Exception:
                continue

        raise ValueError(f"Could not load model from: {model_path}")

    def _create_node(self) -> "PolicyNode":
        """Create ROS2 node for policy execution."""
        return PolicyNode(
            name=self.node_name,
            model=self._model,
            control_topic=self.control_topic,
            pose_topic=self.pose_topic,
            velocity_topic=self.velocity_topic,
            imu_topic=self.imu_topic,
            control_frequency=self.control_frequency,
        )

    def start(self):
        """Start the ROS2 node and control loop."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        rclpy.init()
        self._node = self._create_node()

        try:
            print(f"Starting ROS2 policy node: {self.node_name}")
            print(f"  Control topic: {self.control_topic}")
            print(f"  Control frequency: {self.control_frequency} Hz")
            rclpy.spin(self._node)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the ROS2 node."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if rclpy.ok():
            rclpy.shutdown()

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "model_loaded": self._model is not None,
            "node_active": self._node is not None,
            "control_frequency": self.control_frequency,
        }


if ROS2_AVAILABLE:
    class PolicyNode(Node):
        """ROS2 Node for executing trained policy."""

        def __init__(
            self,
            name: str,
            model,
            control_topic: str,
            pose_topic: str,
            velocity_topic: str,
            imu_topic: str,
            control_frequency: float,
        ):
            super().__init__(name)

            self.model = model
            self.control_frequency = control_frequency

            # Current observation state
            self._obs = np.zeros(13, dtype=np.float32)
            self._pose_received = False
            self._velocity_received = False

            # QoS profile for sensor data
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            # Publishers
            self.control_pub = self.create_publisher(
                Float32MultiArray,
                control_topic,
                10
            )

            # Subscribers
            self.pose_sub = self.create_subscription(
                PoseStamped,
                pose_topic,
                self._pose_callback,
                sensor_qos
            )

            self.velocity_sub = self.create_subscription(
                TwistStamped,
                velocity_topic,
                self._velocity_callback,
                sensor_qos
            )

            self.imu_sub = self.create_subscription(
                Imu,
                imu_topic,
                self._imu_callback,
                sensor_qos
            )

            # Control timer
            timer_period = 1.0 / control_frequency
            self.control_timer = self.create_timer(
                timer_period,
                self._control_callback
            )

            self.get_logger().info(f"Policy node initialized at {control_frequency} Hz")

        def _pose_callback(self, msg: "PoseStamped"):
            """Handle pose updates."""
            pos = msg.pose.position
            orn = msg.pose.orientation

            self._obs[0] = pos.x
            self._obs[1] = pos.y
            self._obs[2] = pos.z
            self._obs[6] = orn.x
            self._obs[7] = orn.y
            self._obs[8] = orn.z
            self._obs[9] = orn.w

            self._pose_received = True

        def _velocity_callback(self, msg: "TwistStamped"):
            """Handle velocity updates."""
            lin = msg.twist.linear
            ang = msg.twist.angular

            self._obs[3] = lin.x
            self._obs[4] = lin.y
            self._obs[5] = lin.z
            self._obs[10] = ang.x
            self._obs[11] = ang.y
            self._obs[12] = ang.z

            self._velocity_received = True

        def _imu_callback(self, msg: "Imu"):
            """Handle IMU updates (optional, for higher-rate angular velocity)."""
            ang = msg.angular_velocity
            self._obs[10] = ang.x
            self._obs[11] = ang.y
            self._obs[12] = ang.z

        def _control_callback(self):
            """Execute policy and publish control commands."""
            if not (self._pose_received and self._velocity_received):
                return  # Wait for sensor data

            # Run policy inference
            action, _ = self.model.predict(self._obs, deterministic=True)

            # Publish control command
            msg = Float32MultiArray()
            msg.data = action.tolist()
            self.control_pub.publish(msg)


def generate_ros2_launch_file(
    package_name: str = "drone_policy",
    node_name: str = "policy_node",
    model_path: str = "model.zip",
) -> str:
    """
    Generate a ROS2 launch file for the policy node.

    Args:
        package_name: ROS2 package name
        node_name: Node name
        model_path: Path to trained model

    Returns:
        Launch file content as string
    """
    return f'''"""
Auto-generated ROS2 launch file for drone policy deployment.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='{model_path}',
            description='Path to trained model file'
        ),
        DeclareLaunchArgument(
            'control_frequency',
            default_value='50.0',
            description='Control loop frequency in Hz'
        ),

        Node(
            package='{package_name}',
            executable='{node_name}',
            name='{node_name}',
            parameters=[{{
                'model_path': LaunchConfiguration('model_path'),
                'control_frequency': LaunchConfiguration('control_frequency'),
            }}],
            output='screen',
        ),
    ])
'''


def create_ros2_package_structure(
    output_dir: str,
    package_name: str = "drone_policy",
) -> Dict[str, str]:
    """
    Create ROS2 package structure for deployment.

    Args:
        output_dir: Output directory for package
        package_name: ROS2 package name

    Returns:
        Dictionary of created file paths
    """
    output_path = Path(output_dir) / package_name
    output_path.mkdir(parents=True, exist_ok=True)

    created_files = {}

    # package.xml
    package_xml = f'''<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.1.0</version>
  <description>Trained drone policy deployment package</description>
  <maintainer email="maintainer@example.com">Maintainer</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
'''
    (output_path / "package.xml").write_text(package_xml)
    created_files["package.xml"] = str(output_path / "package.xml")

    # setup.py
    setup_py = f'''from setuptools import setup

package_name = '{package_name}'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/policy_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maintainer',
    maintainer_email='maintainer@example.com',
    description='Trained drone policy deployment',
    license='MIT',
    entry_points={{
        'console_scripts': [
            'policy_node = {package_name}.policy_node:main',
        ],
    }},
)
'''
    (output_path / "setup.py").write_text(setup_py)
    created_files["setup.py"] = str(output_path / "setup.py")

    # Create subdirectories
    (output_path / package_name).mkdir(exist_ok=True)
    (output_path / "launch").mkdir(exist_ok=True)
    (output_path / "resource").mkdir(exist_ok=True)

    # Resource marker
    (output_path / "resource" / package_name).write_text("")

    # Launch file
    launch_content = generate_ros2_launch_file(package_name=package_name)
    (output_path / "launch" / "policy_launch.py").write_text(launch_content)
    created_files["launch"] = str(output_path / "launch" / "policy_launch.py")

    print(f"Created ROS2 package structure at: {output_path}")
    return created_files

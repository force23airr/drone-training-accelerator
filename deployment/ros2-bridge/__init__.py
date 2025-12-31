"""
ROS2 Bridge for deploying trained models.

Provides ROS2 nodes for:
- Publishing control commands
- Subscribing to sensor data
- Model inference integration
- Real-time telemetry
"""

from deployment.ros2_bridge.deploy_to_ros2 import ROS2DeploymentBridge

__all__ = ["ROS2DeploymentBridge"]

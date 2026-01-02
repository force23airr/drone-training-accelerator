"""
Deployment Module

Tools for deploying trained models to hardware and production environments.

Features:
- ONNX export for cross-platform deployment
- TorchScript export for C++/embedded systems
- ROS2 bridge for real-time control
- Hardware configuration profiles
- AirSim integration for visualization
"""

from deployment.model_export import (
    PolicyExporter,
    OnnxInferenceEngine,
    export_model_to_onnx,
    export_model_to_torchscript,
)

# ROS2 bridge (optional - requires rclpy)
try:
    from deployment.ros2_bridge.deploy_to_ros2 import (
        ROS2DeploymentBridge,
        create_ros2_package_structure,
        generate_ros2_launch_file,
    )
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# AirSim military environments
from deployment.airsim import (
    MilitaryEnvironmentType,
    MilitaryAirSimConfig,
    MILITARY_ENVIRONMENT_CONFIGS,
    get_military_environment,
    get_carrier_environment,
    get_airbase_environment,
    get_contested_airspace,
    get_urban_strike_zone,
)

__all__ = [
    # Model export
    "PolicyExporter",
    "OnnxInferenceEngine",
    "export_model_to_onnx",
    "export_model_to_torchscript",
    # ROS2 (conditional)
    "ROS2_AVAILABLE",
    # AirSim
    "MilitaryEnvironmentType",
    "MilitaryAirSimConfig",
    "MILITARY_ENVIRONMENT_CONFIGS",
    "get_military_environment",
    "get_carrier_environment",
    "get_airbase_environment",
    "get_contested_airspace",
    "get_urban_strike_zone",
]

if ROS2_AVAILABLE:
    __all__.extend([
        "ROS2DeploymentBridge",
        "create_ros2_package_structure",
        "generate_ros2_launch_file",
    ])

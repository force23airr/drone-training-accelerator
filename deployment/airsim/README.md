# AirSim Integration Guide

Deploy trained drone policies to photorealistic environments using Microsoft AirSim + Unreal Engine.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. TRAIN (Fast)              2. VISUALIZE (Realistic)    │
│   ┌─────────────────┐          ┌─────────────────────┐     │
│   │    PyBullet     │          │      AirSim         │     │
│   │   (Headless)    │ ──────▶  │  (Unreal Engine)    │     │
│   │                 │  Export  │                     │     │
│   │  ~5000 steps/s  │  Policy  │  Photorealistic     │     │
│   │  Blue boxes     │          │  Real environments  │     │
│   └─────────────────┘          └─────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Step 1: Install AirSim Python Client

```bash
pip install airsim
```

### Step 2: Download AirSim Simulator

Download pre-built binaries from:
https://github.com/microsoft/AirSim/releases

Available environments:
- **Blocks** - Simple test environment
- **CityEnviron** - Urban city streets
- **Neighborhood** - Suburban area
- **LandscapeMountains** - Mountain terrain
- **Africa** - Desert/savanna
- **Warehouse** - Indoor warehouse

### Step 3: Run AirSim

1. Extract the downloaded zip
2. Run the executable (e.g., `Blocks.exe` on Windows, `Blocks.app` on Mac)
3. AirSim will start and wait for API connections

## Usage

### Quick Start: Visualize Trained Policy

```python
from simulation.physics.airsim_integration import visualize_policy

# Visualize your trained hover policy in Florida streets
visualize_policy(
    model_path="trained_models/quadcopter_basic_hover_stability_final",
    environment="florida_street",
    max_steps=500
)
```

### Full Evaluation

```python
from simulation.physics.airsim_integration import deploy_to_airsim

results = deploy_to_airsim(
    model_path="trained_models/my_policy",
    environment="urban_sar",  # Urban search & rescue
    num_episodes=5,
    algorithm="PPO"
)

print(f"Mean reward: {results['mean_reward']}")
print(f"Collision rate: {results['collision_rate']*100}%")
```

### Record Video

```python
from simulation.physics.airsim_integration import record_video

record_video(
    model_path="trained_models/my_policy",
    output_path="demo_video.mp4",
    environment="mountain_recon",
    max_steps=500
)
```

### Command Line

```bash
# Basic visualization
python -m simulation.physics.airsim_integration.policy_bridge \
    --model trained_models/quadcopter_basic_hover_stability_final \
    --environment florida_street \
    --episodes 3

# Record video
python -m simulation.physics.airsim_integration.policy_bridge \
    --model trained_models/my_policy \
    --environment coastal_patrol \
    --record output.mp4
```

## Available Environments

| Name | Description | Use Case |
|------|-------------|----------|
| `florida_street` | Urban Miami streets | General testing |
| `mountain_recon` | High-altitude mountains | Reconnaissance |
| `urban_sar` | Dense city downtown | Search & rescue |
| `warehouse` | Indoor warehouse | Inspection |
| `coastal_patrol` | Beach/coastal area | Maritime patrol |

## Custom Environments

### Create Custom Configuration

```python
from simulation.physics.airsim_integration import RealisticEnvironmentConfig, EnvironmentType

my_env = RealisticEnvironmentConfig(
    name="My Test Zone",
    environment_type=EnvironmentType.CITY_DOWNTOWN,
    location="Los Angeles, CA",
    latitude=34.0522,
    longitude=-118.2437,
    weather_preset="cloudy",
    wind_speed=10.0,
    time_of_day="dusk",
    building_density="high",
    enable_lidar=True,
)
```

### Build Custom Unreal Environment

For truly custom environments (specific real-world locations):

1. Install Unreal Engine 4.27
2. Clone AirSim source: `git clone https://github.com/microsoft/AirSim`
3. Build AirSim plugin for Unreal
4. Create new Unreal project
5. Import terrain/building assets
6. Add AirSim plugin
7. Configure `settings.json`

See: https://microsoft.github.io/AirSim/build_windows/

## API Reference

### PolicyDeploymentBridge

```python
bridge = PolicyDeploymentBridge(
    model_path="path/to/model",
    environment_config=my_config,
    algorithm="PPO"  # or SAC, TD3
)

# Load model and connect to AirSim
bridge.load()

# Run single episode
results = bridge.run_episode(max_steps=1000)

# Run evaluation
summary = bridge.run_evaluation(num_episodes=10)

# Clean up
bridge.close()
```

### AirSimBackend (Low-Level)

```python
from simulation.physics.airsim_integration import AirSimBackend

backend = AirSimBackend()
backend.initialize(render_mode="human", environment="CityEnviron")

# Get drone state
pos = backend.get_position()
vel = backend.get_velocity()

# Control drone
backend.move_to_position((10, 20, 5), velocity=5.0)
backend.hover()

# Get camera images
rgb = backend.get_front_camera_image()
depth = backend.get_depth_image()
lidar = backend.get_lidar_data()

# Weather control
backend.set_weather(rain=0.5, fog=0.3)
backend.set_time_of_day(True, "2024-06-21 18:00:00")
backend.set_wind((10, 5, 0))

backend.shutdown()
```

## Troubleshooting

### "Failed to connect to AirSim"

- Make sure AirSim simulator is running before running Python code
- Check that no firewall is blocking localhost connections
- Try restarting AirSim

### "AIRSIM_AVAILABLE is False"

```bash
pip install airsim
```

### Poor performance / Low FPS

- AirSim is GPU-intensive
- Lower graphics settings in Unreal
- Use a smaller environment (Blocks)
- Reduce camera resolution

### Actions don't match training

- Ensure observation space matches between PyBullet and AirSim
- Check action space scaling
- The bridge handles most conversions automatically

## Architecture

```
simulation/physics/airsim_integration/
├── __init__.py              # Package exports
├── airsim_backend.py        # SimulatorBackend implementation
├── airsim_environment.py    # Gymnasium environment + configs
└── policy_bridge.py         # Policy deployment utilities

deployment/airsim/
└── README.md                # This file
```

## Next Steps

1. **Train in PyBullet** - Fast iteration on policy
2. **Test in AirSim** - Validate in realistic conditions
3. **Record demos** - Generate videos for stakeholders
4. **Deploy to hardware** - Use ROS2 bridge for real drones

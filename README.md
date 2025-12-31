# Autonomous Flight Training Platform

**Accelerating US Drone Development Through Simulation and Reinforcement Learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyBullet](https://img.shields.io/badge/Physics-PyBullet-green.svg)](https://pybullet.org/)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-red.svg)](https://gymnasium.farama.org/)

A production-ready, modular platform for training autonomous drone agents using reinforcement learning. Designed to accelerate drone development cycles for defense, law enforcement, and commercial applications.

---

## Why This Platform?

The US faces a critical gap in autonomous drone capabilities. Traditional development cycles are too slow:

| Traditional Approach | This Platform |
|---------------------|---------------|
| 6-12 months per iteration | Days to weeks |
| Hardware-dependent testing | 100% simulated training |
| Single-scenario training | Multi-environment robustness |
| Manual policy tuning | Automated RL optimization |

**This platform enables rapid prototyping of autonomous drone behaviors before any hardware is built.**

---

## Key Features

### Simulation Agnostic Architecture
```python
from simulation.physics import get_backend

# Swap physics engines without changing training code
backend = get_backend("pybullet")  # Fast, Python-native
# backend = get_backend("gazebo")  # ROS2 integration (future)
```

### Realistic Environmental Conditions
```python
from simulation import EnvironmentalConditions, WeatherType, TimeOfDay, TerrainType

# Train for real-world deployment scenarios
conditions = EnvironmentalConditions(
    weather=WeatherType.RAIN,
    time_of_day=TimeOfDay.NIGHT,
    terrain=TerrainType.URBAN,
    wind=WindModel(base_speed=12.0, gust_intensity=5.0)
)
```

### Mission-Ready Environments
- **UrbanNavigationEnv** - GPS-denied indoor/urban navigation
- **MaritimePatrolEnv** - Ship tracking and search patterns
- **SwarmCoordinationEnv** - Multi-agent formation flying

### 12+ Pre-Configured Drone Platforms
```python
from simulation import list_platforms, get_platform_config

print(list_platforms())
# ['quadcopter_basic', 'quadcopter_racing', 'quadcopter_heavy_lift',
#  'hexacopter_standard', 'octocopter_industrial', 'fixed_wing_trainer',
#  'fixed_wing_survey', 'vtol_tiltrotor', 'vtol_tailsitter',
#  'fpv_micro', 'inspection_drone']
```

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-org/drone-training-accelerator.git
cd drone-training-accelerator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Train Your First Agent (5 minutes)
```bash
python quickstart_train.py --platform quadcopter_basic --mission hover_stability --timesteps 50000
```

### Use Pre-Built Environments
```python
from simulation import BaseDroneEnv, get_platform_config, create_windy_conditions

# Create environment with challenging conditions
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_racing"),
    environmental_conditions=create_windy_conditions(wind_speed=10.0),
    render_mode="human"
)

# Standard Gymnasium interface
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## Architecture

```
drone-training-accelerator/
│
├── simulation/                 # Core simulation framework
│   ├── environments/          # Gymnasium-compatible environments
│   │   ├── base_drone_env.py         # Base environment class
│   │   ├── environmental_conditions.py # Weather, wind, lighting
│   │   ├── urban_nav_env.py          # Urban navigation mission
│   │   ├── maritime_patrol_env.py    # Maritime operations
│   │   └── swarm_coordination_env.py # Multi-agent coordination
│   │
│   ├── platforms/             # Drone configurations
│   │   └── platform_configs.py       # 12+ platform definitions
│   │
│   └── physics/               # Physics engine backends
│       ├── simulator_backend.py      # Abstract interface
│       ├── pybullet_backend.py       # PyBullet implementation
│       └── gazebo_backend.py         # Gazebo stub (future)
│
├── training/                  # RL training infrastructure
│   ├── algorithms/            # PPO, SAC, TD3 wrappers
│   ├── suites/               # Mission training configurations
│   │   └── mission_suites.py         # 9 mission types
│   ├── parallel/             # Multi-environment training
│   │   └── parallel_trainer.py       # Vectorized training
│   └── transfer/             # Sim-to-real transfer
│
├── evaluation/               # Benchmarking and metrics
├── deployment/               # Hardware deployment
│   └── ros2-bridge/          # ROS2 integration
│
└── infrastructure/           # Docker, CI/CD
    └── containers/
        ├── Dockerfile
        └── docker-compose.yml
```

---

## Environmental Conditions System

Train policies that work in the real world by simulating diverse conditions:

| Category | Options |
|----------|---------|
| **Weather** | Clear, Rain, Snow, Fog, Storm, Dust, Hail |
| **Time of Day** | Dawn, Day, Dusk, Night |
| **Terrain** | Open Field, Urban, Forest, Mountain, Coastal, Desert, Indoor |
| **Wind** | Base speed, Gusts, Turbulence, Direction |

```python
from simulation.environments import create_random_conditions

# Automatic domain randomization
conditions = create_random_conditions(difficulty="hard", seed=42)
# Randomizes weather, time, wind within difficulty parameters
```

### Sensor Noise Scaling
Conditions automatically affect sensor quality:
```python
# GPS degrades in urban environments
noise_scale = conditions.get_sensor_noise_scale("gps")  # Returns 1.0-5.0

# Cameras struggle at night
noise_scale = conditions.get_sensor_noise_scale("camera")  # Higher at night
```

---

## Mission Environments

### Urban Navigation
```python
from simulation import UrbanNavigationEnv, get_platform_config

env = UrbanNavigationEnv(
    platform_config=get_platform_config("inspection_drone"),
    scenario="indoor",           # 'outdoor_urban', 'indoor', 'parking_garage'
    num_waypoints=5,
    obstacle_density="high"
)
```
**Use Cases:** Law enforcement building clearing, search and rescue, warehouse inventory

### Maritime Patrol
```python
from simulation import MaritimePatrolEnv, get_platform_config

env = MaritimePatrolEnv(
    platform_config=get_platform_config("fixed_wing_survey"),
    mission_type="tracking",     # 'patrol', 'tracking', 'search_rescue'
    search_area_size=100.0,
    num_targets=3
)
```
**Use Cases:** Coast Guard operations, port security, naval ISR

### Swarm Coordination
```python
from simulation import SwarmCoordinationEnv, get_platform_config

env = SwarmCoordinationEnv(
    platform_config=get_platform_config("fpv_micro"),
    num_agents=4,
    formation="diamond",         # 'line', 'v_formation', 'diamond', 'grid'
    agent_id=0                   # This agent's role (0 = leader)
)
```
**Use Cases:** Tactical swarm operations, distributed sensing, coordinated search

---

## Training

### Basic Training
```bash
python quickstart_train.py \
    --platform quadcopter_basic \
    --mission hover_stability \
    --timesteps 100000 \
    --num-envs 4
```

### Advanced Training with Custom Conditions
```python
from training import ParallelTrainer, MissionSuite
from simulation import BaseDroneEnv, get_platform_config, create_adverse_conditions

# Create challenging environment
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_heavy_lift"),
    environmental_conditions=create_adverse_conditions()
)

# Configure mission
mission = MissionSuite("wind_disturbance")

# Train with parallel environments
trainer = ParallelTrainer(
    env=env,
    mission=mission,
    num_envs=8,
    algorithm="ppo"
)

trainer.train(total_timesteps=1_000_000)
trainer.save("./models/wind_resistant_policy")
```

### Available Missions
| Mission | Difficulty | Description |
|---------|------------|-------------|
| `hover_stability` | Beginner | Maintain stable hover |
| `takeoff_landing` | Beginner | Controlled vertical flight |
| `waypoint_navigation` | Intermediate | Navigate waypoint sequence |
| `trajectory_tracking` | Intermediate | Follow reference trajectory |
| `obstacle_avoidance` | Advanced | Navigate through obstacles |
| `wind_disturbance` | Advanced | Maintain position in wind |
| `aggressive_maneuvers` | Expert | Acrobatic flight |
| `multi_agent_coordination` | Expert | Formation flying |
| `autonomous_landing_moving` | Expert | Land on moving platform |

---

## Deployment

### Export to ROS2
```python
from deployment.ros2_bridge import ROS2DeploymentBridge, create_ros2_package_structure

# Generate ROS2 package
create_ros2_package_structure(
    output_dir="./ros2_ws/src",
    package_name="drone_policy"
)

# Deploy trained model
bridge = ROS2DeploymentBridge(model_path="./models/my_policy.zip")
bridge.start()
```

### Docker Deployment
```bash
# GPU training
docker-compose up training-gpu

# CPU training
docker-compose up training-cpu

# TensorBoard monitoring
docker-compose up tensorboard
# Visit http://localhost:6006
```

---

## Performance Benchmarks

| Environment | Steps/Second | GPU Memory | Notes |
|-------------|--------------|------------|-------|
| BaseDroneEnv (1 env) | ~2,000 | 200 MB | Single environment |
| BaseDroneEnv (8 env) | ~12,000 | 400 MB | Vectorized parallel |
| UrbanNavigationEnv | ~1,500 | 300 MB | With obstacles |
| SwarmCoordinationEnv | ~800 | 500 MB | 4 agents |

*Benchmarks on RTX 3080, AMD Ryzen 9 5900X*

---

## Roadmap

### Phase 1 (Current)
- [x] Core simulation framework
- [x] Environmental conditions system
- [x] Mission-specific environments
- [x] Parallel training infrastructure

### Phase 2 (Next)
- [ ] Gazebo/Isaac Sim backends
- [ ] ONNX model export
- [ ] PX4/ArduPilot SITL integration
- [ ] Real-time telemetry dashboard

### Phase 3 (Future)
- [ ] Sim-to-real transfer learning
- [ ] Hardware-in-the-loop testing
- [ ] Multi-agent policy optimization
- [ ] Cloud training orchestration

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- Additional physics backends (Gazebo, Isaac Sim)
- New mission environments
- Sim-to-real transfer techniques
- Documentation improvements

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [PyBullet](https://pybullet.org/) - Physics simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - Environment interface

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/your-org/drone-training-accelerator/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-org/drone-training-accelerator/discussions)

---

**Built to accelerate American drone innovation.**

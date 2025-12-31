# Quick Reference

Commands and code snippets cheat sheet for the Autonomous Flight Training Platform.

---

## CLI Commands

### Training
```bash
# Basic training
python quickstart_train.py --platform quadcopter_basic --mission hover_stability

# Full options
python quickstart_train.py \
    --platform quadcopter_racing \
    --mission waypoint_navigation \
    --timesteps 500000 \
    --num-envs 8 \
    --output-dir ./models \
    --render

# List available platforms
python quickstart_train.py --list-platforms
```

### Docker
```bash
# GPU training
docker-compose up training-gpu

# CPU training
docker-compose up training-cpu

# TensorBoard (port 6006)
docker-compose up tensorboard

# Jupyter notebook (port 8888)
docker-compose up jupyter

# All services
docker-compose up

# Interactive shell
docker exec -it drone-training-gpu bash
```

### TensorBoard
```bash
tensorboard --logdir ./trained_models/logs --port 6006
```

---

## Python Quick Reference

### Environment Creation

```python
from simulation import BaseDroneEnv, get_platform_config

# Basic environment
env = BaseDroneEnv(platform_config=get_platform_config("quadcopter_basic"))

# With visualization
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    render_mode="human"
)

# With environmental conditions
from simulation import create_windy_conditions
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    environmental_conditions=create_windy_conditions(wind_speed=10.0)
)
```

### Available Platforms
```python
from simulation import list_platforms
print(list_platforms())
```

| Platform | Type | Use Case |
|----------|------|----------|
| `quadcopter_basic` | Quadcopter | General training |
| `quadcopter_racing` | Quadcopter | High-speed maneuvers |
| `quadcopter_heavy_lift` | Quadcopter | Payload operations |
| `hexacopter_standard` | Hexacopter | Redundancy |
| `octocopter_industrial` | Octocopter | Heavy industrial |
| `fixed_wing_trainer` | Fixed Wing | Forward flight |
| `fixed_wing_survey` | Fixed Wing | Long endurance |
| `vtol_tiltrotor` | VTOL | Versatile |
| `vtol_tailsitter` | VTOL | Compact VTOL |
| `fpv_micro` | Quadcopter | Indoor/FPV |
| `inspection_drone` | Quadcopter | Inspection missions |

### Environmental Conditions

```python
from simulation.environments import (
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    TerrainType,
    WindModel,
    create_clear_day,
    create_windy_conditions,
    create_night_conditions,
    create_urban_conditions,
    create_indoor_conditions,
    create_adverse_conditions,
    create_random_conditions,
)

# Presets
conditions = create_clear_day()
conditions = create_windy_conditions(wind_speed=12.0)
conditions = create_night_conditions()
conditions = create_urban_conditions()
conditions = create_indoor_conditions()
conditions = create_adverse_conditions()
conditions = create_random_conditions(difficulty="hard")

# Custom conditions
conditions = EnvironmentalConditions(
    weather=WeatherType.RAIN,
    time_of_day=TimeOfDay.DUSK,
    terrain=TerrainType.URBAN,
    wind=WindModel(
        base_speed=10.0,
        base_direction=1.57,  # radians
        gust_intensity=5.0,
        gust_probability=0.1,
        turbulence_intensity=1.0
    )
)
```

### Mission Environments

```python
# Urban navigation
from simulation import UrbanNavigationEnv
env = UrbanNavigationEnv(
    platform_config=get_platform_config("inspection_drone"),
    scenario="indoor",  # "outdoor_urban", "indoor", "parking_garage"
    num_waypoints=5,
    obstacle_density="medium"  # "low", "medium", "high"
)

# Maritime patrol
from simulation import MaritimePatrolEnv
env = MaritimePatrolEnv(
    platform_config=get_platform_config("fixed_wing_survey"),
    mission_type="tracking",  # "patrol", "tracking", "search_rescue"
    search_area_size=100.0,
    num_targets=3,
    target_speed=5.0
)

# Swarm coordination
from simulation import SwarmCoordinationEnv
env = SwarmCoordinationEnv(
    platform_config=get_platform_config("fpv_micro"),
    num_agents=4,
    formation="diamond",  # "line", "v_formation", "diamond", "grid"
    agent_id=0,  # 0 = leader
    communication_range=20.0
)
```

### Training

```python
from training import ParallelTrainer, MissionSuite
from simulation import BaseDroneEnv, get_platform_config

# Setup
env = BaseDroneEnv(platform_config=get_platform_config("quadcopter_basic"))
mission = MissionSuite("hover_stability")

# Create trainer
trainer = ParallelTrainer(
    env=env,
    mission=mission,
    num_envs=4,
    algorithm="ppo",  # "ppo", "sac", "td3"
    output_dir="./models"
)

# Train
trainer.train(total_timesteps=100_000)

# Save
trainer.save("./models/my_model")

# Evaluate
metrics = trainer.evaluate(n_eval_episodes=10)
print(f"Mean reward: {metrics['mean_reward']}")
```

### Available Missions
```python
from training import list_missions
print(list_missions())
```

| Mission | Difficulty | Key Objectives |
|---------|------------|----------------|
| `hover_stability` | Beginner | Maintain position |
| `takeoff_landing` | Beginner | Vertical flight |
| `waypoint_navigation` | Intermediate | Navigate waypoints |
| `trajectory_tracking` | Intermediate | Follow trajectory |
| `obstacle_avoidance` | Advanced | Avoid obstacles |
| `wind_disturbance` | Advanced | Handle wind |
| `aggressive_maneuvers` | Expert | Acrobatics |
| `multi_agent_coordination` | Expert | Formation flying |
| `autonomous_landing_moving` | Expert | Land on moving target |

### Loading Trained Models

```python
from stable_baselines3 import PPO, SAC, TD3
from simulation import BaseDroneEnv, get_platform_config

env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    render_mode="human"
)

# Load model
model = PPO.load("./models/my_model")

# Run inference
obs, info = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Observation Space

Default observation (13 dimensions):
```
[0:3]   position (x, y, z)
[3:6]   velocity (vx, vy, vz)
[6:10]  orientation quaternion (qx, qy, qz, qw)
[10:13] angular velocity (wx, wy, wz)
```

Mission environments may extend this with additional features.

---

## Action Space

Motor commands normalized to [-1, 1]:
```
Quadcopter: [motor0, motor1, motor2, motor3]
Hexacopter: [motor0, motor1, motor2, motor3, motor4, motor5]
Fixed Wing: [throttle] or [throttle, aileron, elevator, rudder]
```

---

## Custom Platform Configuration

```python
from simulation.platforms import create_custom_platform, get_platform_config

# Create custom platform based on existing
platform_id = create_custom_platform(
    name="My Heavy Drone",
    base_platform="quadcopter_heavy_lift",
    overrides={
        "mass": 8.0,
        "max_thrust_per_motor": 30.0,
        "physics_params": {
            "drag_coefficient": 0.2
        }
    }
)

# Use it
config = get_platform_config(platform_id)
env = BaseDroneEnv(platform_config=config)
```

---

## Physics Backend

```python
from simulation.physics import get_backend, PyBulletBackend

# Get backend by name
backend = get_backend("pybullet")

# Or instantiate directly
backend = PyBulletBackend()
backend.initialize(render_mode="human")

# Use backend
object_id = backend.create_box(
    half_extents=(0.5, 0.5, 0.5),
    position=(0, 0, 1),
    mass=1.0
)
backend.apply_force(object_id, (0, 0, 10))
backend.step()

state = backend.get_state(object_id)
print(state["position"])

backend.shutdown()
```

---

## Deployment

### ROS2 Bridge
```python
from deployment.ros2_bridge import ROS2DeploymentBridge

bridge = ROS2DeploymentBridge(
    model_path="./models/my_model.zip",
    control_topic="/drone/control",
    pose_topic="/drone/pose"
)
bridge.start()
```

### Generate ROS2 Package
```python
from deployment.ros2_bridge import create_ros2_package_structure

create_ros2_package_structure(
    output_dir="./ros2_ws/src",
    package_name="drone_policy"
)
```

---

## Common Patterns

### Domain Randomization
```python
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    domain_randomization=True
)

# Or randomize conditions on reset
obs, info = env.reset(options={
    "randomize_conditions": True,
    "condition_difficulty": "hard"
})
```

### Add Obstacles
```python
env = BaseDroneEnv(platform_config=get_platform_config("quadcopter_basic"))
obs, info = env.reset()

# Add obstacles after reset
env.add_obstacle("box", position=(5, 0, 1), size=(0.5, 0.5, 1))
env.add_obstacle("sphere", position=(0, 5, 2), size=(0.3,))
env.add_obstacle("cylinder", position=(-3, 3, 0.5), size=(0.2, 1.0))
```

### Get Telemetry
```python
telemetry = env.get_telemetry()
print(f"Position: {telemetry['position']}")
print(f"Velocity: {telemetry['velocity']}")
print(f"Target distance: {telemetry['target_distance']}")
```

### Check Safety
```python
is_safe, reason = env.env_conditions.is_flight_safe()
if not is_safe:
    print(f"Unsafe: {reason}")
```

---

## Environment Info Dictionary

After `step()` or `reset()`:
```python
obs, reward, terminated, truncated, info = env.step(action)

# Available in info:
info["step"]              # Current step number
info["position"]          # Drone position
info["velocity"]          # Drone velocity
info["collisions"]        # Collision count
info["target_distance"]   # Distance to target

# On episode end:
info["episode_stats"]     # Cumulative statistics
```

---

## Debugging Tips

```python
# Print action/observation spaces
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Check environment state
print(f"Step: {env._step_count}")
print(f"Episode: {env._episode_count}")

# Get detailed telemetry
print(env.get_telemetry())

# Check conditions
print(env.env_conditions.to_dict())
```

---

## File Locations

```
trained_models/          # Saved models
trained_models/logs/     # TensorBoard logs
trained_models/checkpoints/  # Training checkpoints
```

# Getting Started Guide

This guide will walk you through training your first autonomous drone agent in under 30 minutes.

---

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU with CUDA support (optional but recommended)
- Basic familiarity with Python and command line

---

## Step 1: Installation (5 minutes)

### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-org/drone-training-accelerator.git
cd drone-training-accelerator

# Create virtual environment
python -m venv venv

# Activate (choose your OS)
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows PowerShell
venv\Scripts\activate.bat     # Windows CMD

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Verify Installation
```python
python -c "from simulation import BaseDroneEnv, list_platforms; print('Success!'); print(list_platforms())"
```

You should see a list of available drone platforms.

---

## Step 2: Understanding the Platform (10 minutes)

### Core Concepts

**1. Environments** - Gymnasium-compatible simulation environments
```python
from simulation import BaseDroneEnv

# Environments follow the standard Gymnasium interface:
# - env.reset() -> (observation, info)
# - env.step(action) -> (observation, reward, terminated, truncated, info)
# - env.close()
```

**2. Platform Configs** - Define drone physical properties
```python
from simulation import get_platform_config, list_platforms

# See all available platforms
print(list_platforms())

# Get specific platform configuration
config = get_platform_config("quadcopter_basic")
print(f"Mass: {config['mass']} kg")
print(f"Motors: {config['num_motors']}")
```

**3. Environmental Conditions** - Simulate real-world factors
```python
from simulation import EnvironmentalConditions, WeatherType, TimeOfDay

conditions = EnvironmentalConditions(
    weather=WeatherType.RAIN,
    time_of_day=TimeOfDay.DUSK
)
# Wind, visibility, sensor noise automatically adjusted
```

**4. Mission Suites** - Training curricula with objectives
```python
from training import MissionSuite

mission = MissionSuite("hover_stability")
print(f"Objectives: {mission.config.objectives}")
```

---

## Step 3: Your First Simulation (5 minutes)

### Manual Control Test
Create a file called `test_environment.py`:

```python
"""Test the simulation environment."""
from simulation import BaseDroneEnv, get_platform_config

# Create environment with visualization
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    render_mode="human"  # Opens PyBullet GUI
)

# Reset and run
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Run 500 steps with random actions
total_reward = 0
for step in range(500):
    # Random action (4 motor commands between -1 and 1)
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        print(f"Total reward: {total_reward:.2f}")
        obs, info = env.reset()
        total_reward = 0

env.close()
print("Done!")
```

Run it:
```bash
python test_environment.py
```

You should see a PyBullet window with a drone falling (random actions won't stabilize it - that's what training is for!).

---

## Step 4: Train Your First Agent (10 minutes)

### Quick Training
```bash
python quickstart_train.py \
    --platform quadcopter_basic \
    --mission hover_stability \
    --timesteps 50000 \
    --render
```

This will:
1. Create a quadcopter environment
2. Train with PPO algorithm
3. Save checkpoints to `./trained_models/`

### Watch Training Progress
In another terminal:
```bash
tensorboard --logdir ./trained_models/logs
```
Open http://localhost:6006 to see training curves.

---

## Step 5: Evaluate Your Agent

### Load and Test Trained Model
Create `evaluate_agent.py`:

```python
"""Evaluate a trained agent."""
from stable_baselines3 import PPO
from simulation import BaseDroneEnv, get_platform_config

# Create environment
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    render_mode="human"
)

# Load trained model
model = PPO.load("./trained_models/quadcopter_basic_hover_stability_final")

# Evaluate
obs, info = env.reset()
total_reward = 0
steps = 0

while True:
    # Use trained policy
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

    if terminated or truncated:
        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        break

env.close()
```

---

## Step 6: Try Different Scenarios

### Add Environmental Challenges
```python
from simulation import (
    BaseDroneEnv,
    get_platform_config,
    create_windy_conditions,
    create_night_conditions
)

# Train in windy conditions
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    environmental_conditions=create_windy_conditions(wind_speed=8.0)
)
```

### Try Different Platforms
```python
# Racing drone - fast and agile
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_racing")
)

# Heavy lift - stable but slow
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_heavy_lift")
)

# Fixed wing - forward flight
env = BaseDroneEnv(
    platform_config=get_platform_config("fixed_wing_trainer")
)
```

### Mission-Specific Environments
```python
from simulation import UrbanNavigationEnv, get_platform_config

# Indoor navigation with obstacles
env = UrbanNavigationEnv(
    platform_config=get_platform_config("inspection_drone"),
    scenario="indoor",
    num_waypoints=5,
    obstacle_density="medium"
)
```

---

## Step 7: Advanced Training

### Parallel Training (Faster!)
```python
from training import ParallelTrainer, MissionSuite
from simulation import BaseDroneEnv, get_platform_config

env = BaseDroneEnv(platform_config=get_platform_config("quadcopter_basic"))
mission = MissionSuite("waypoint_navigation")

trainer = ParallelTrainer(
    env=env,
    mission=mission,
    num_envs=8,           # 8 parallel environments
    algorithm="ppo",      # or "sac", "td3"
    output_dir="./models/waypoint_nav"
)

trainer.train(total_timesteps=500_000)
trainer.save("./models/waypoint_nav/final_model")
```

### Domain Randomization
```python
from simulation import BaseDroneEnv, get_platform_config, create_random_conditions

# Random conditions each episode for robust policies
env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_basic"),
    domain_randomization=True
)

# Or manually randomize
obs, info = env.reset(options={
    "randomize_conditions": True,
    "condition_difficulty": "hard"
})
```

### Curriculum Learning
Missions automatically progress through curriculum stages:
```python
from training import MissionSuite

mission = MissionSuite("hover_stability")

# Curriculum stages (automatic progression):
# Stage 1: Hover at z=0.5m, no wind
# Stage 2: Hover at z=1.0m, no wind
# Stage 3: Hover at z=1.5m, light wind
# Stage 4: Hover at z=2.0m, moderate wind
```

---

## Common Issues and Solutions

### "PyBullet GUI not opening"
```bash
# Install display dependencies (Linux)
sudo apt-get install libgl1-mesa-glx

# Or run headless
env = BaseDroneEnv(..., render_mode=None)
```

### "Out of memory"
```python
# Reduce parallel environments
trainer = ParallelTrainer(..., num_envs=2)

# Or use CPU
trainer = ParallelTrainer(..., device="cpu")
```

### "Training not converging"
```python
# Increase training time
trainer.train(total_timesteps=1_000_000)

# Or try different algorithm
trainer = ParallelTrainer(..., algorithm="sac")

# Or simplify the mission
mission = MissionSuite("hover_stability")  # Easier than waypoint_navigation
```

### "Drone crashes immediately"
```python
# Check platform config is appropriate
config = get_platform_config("quadcopter_basic")  # Good starting point

# Ensure action scaling is correct
# Actions should be in [-1, 1] range
action = np.clip(action, -1, 1)
```

---

## Next Steps

1. **Read the [QUICK_REFERENCE.md](QUICK_REFERENCE.md)** for a commands cheat sheet
2. **Explore mission environments** in `simulation/environments/`
3. **Customize platform configs** in `simulation/platforms/platform_configs.py`
4. **Deploy to hardware** with the ROS2 bridge (see deployment docs)

---

## Example Projects

### Project 1: Wind-Resistant Hover
Train a drone to hover stable in strong, gusty winds.
```python
from simulation import BaseDroneEnv, get_platform_config
from simulation.environments import EnvironmentalConditions, WindModel

conditions = EnvironmentalConditions(
    wind=WindModel(
        base_speed=15.0,
        gust_intensity=8.0,
        gust_probability=0.2,
        turbulence_intensity=2.0
    )
)

env = BaseDroneEnv(
    platform_config=get_platform_config("quadcopter_heavy_lift"),
    environmental_conditions=conditions
)
# Train with MissionSuite("wind_disturbance")
```

### Project 2: Indoor Search and Rescue
Navigate through a building to reach multiple points of interest.
```python
from simulation import UrbanNavigationEnv, get_platform_config

env = UrbanNavigationEnv(
    platform_config=get_platform_config("fpv_micro"),
    scenario="indoor",
    num_waypoints=10,
    obstacle_density="high"
)
# Train until waypoints_reached == 10
```

### Project 3: Swarm Formation
Coordinate 4 drones in diamond formation.
```python
from simulation import SwarmCoordinationEnv, get_platform_config

# Train each agent separately or use multi-agent RL
for agent_id in range(4):
    env = SwarmCoordinationEnv(
        platform_config=get_platform_config("fpv_micro"),
        num_agents=4,
        formation="diamond",
        agent_id=agent_id
    )
    # Train this agent
```

---

**Congratulations!** You've completed the getting started guide. You now know how to:
- Create and configure drone environments
- Train agents with different missions
- Evaluate and deploy trained policies
- Handle various environmental conditions

Happy flying!

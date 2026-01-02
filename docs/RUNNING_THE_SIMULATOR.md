# Running the Dogfight Simulator

## Quick Start

### 1. Start the Simulation (Terminal 1)
```bash
cd /Users/angelfernandez/drone-training-accelerator
source ~/drone-env/bin/activate  # or your conda env

python scripts/test_visualizer.py --red 1 --blue 1 --episodes 10 --port 5555
```

### 2. Start the Visualizer (Terminal 2)
```bash
cd /Users/angelfernandez/drone-training-accelerator
source ~/drone-env/bin/activate

python -m visualization.renderer.dogfight_viewer --connect localhost:5555
```

---

## Simulation Options

```bash
python scripts/test_visualizer.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--red` | 2 | Number of red team drones |
| `--blue` | 2 | Number of blue team drones |
| `--port` | 5555 | ZeroMQ streaming port |
| `--episodes` | 5 | Number of episodes to run |
| `--no-record` | false | Disable replay recording |
| `--mode` | demo | `demo` or `train` (requires stable-baselines3) |

### Examples

```bash
# 1v1 dogfight
python scripts/test_visualizer.py --red 1 --blue 1 --port 5555

# 2v2 team battle
python scripts/test_visualizer.py --red 2 --blue 2 --port 5555

# Large swarm battle (4v4)
python scripts/test_visualizer.py --red 4 --blue 4 --port 5555
```

---

## Visualizer Controls

### Camera Modes
| Key | Mode | Description |
|-----|------|-------------|
| `1` | Chase | Follow behind selected drone |
| `2` | Spectator | Free-fly camera (WASD to move) |
| `3` | Tactical | Top-down overview |
| `4` | Auto | AI-directed cinematic view |

### Other Controls
| Key | Action |
|-----|--------|
| `Tab` | Cycle to next drone |
| `Shift+Tab` | Cycle to previous drone |
| `V` | Toggle velocity vectors |
| `ESC` | Exit |

### Spectator Camera Movement
| Key | Direction |
|-----|-----------|
| `W` | Forward |
| `S` | Backward |
| `A` | Left |
| `D` | Right |
| `Space` | Up |
| `Shift` | Down |

---

## Visualizer Options

```bash
python -m visualization.renderer.dogfight_viewer [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--connect HOST:PORT` | Connect to live simulation |
| `--replay PATH` | Play back a recorded replay file |

### Examples

```bash
# Connect to simulation on port 5555
python -m visualization.renderer.dogfight_viewer --connect localhost:5555

# Play back a replay
python -m visualization.renderer.dogfight_viewer --replay dogfight_demo.dfrp
```

---

## HUD Display

The visualizer shows:

- **Top Center**: Score (RED vs BLUE) and match time
- **Top Right**: Current camera mode
- **Bottom Left**: Selected drone info
  - Team and ID
  - Health bar
  - Speed (m/s)
  - Altitude (m)
  - G-Force
  - Kill/Death ratio
  - Current maneuver

### Maneuver Names You'll See
| Maneuver | Meaning |
|----------|---------|
| PURSUIT | Chasing enemy |
| GUN RUN | Close attack approach |
| GUNS GUNS GUNS | Firing guns |
| FOX TWO | Firing missile |
| RTB | Returning to center (boundary avoidance) |
| PULL UP | Ground avoidance |
| RECOVER | Stall recovery |

---

## Combat Styles

Each team has a unique fighting style:

- **Red Team (AGGRESSIVE)**: Close combat, guns, hard turns
- **Blue Team (ACROBATIC)**: Stylish maneuvers, varied attacks

---

## Troubleshooting

### "Address already in use"
Another simulation is using that port. Either:
- Kill the old process: `pkill -f test_visualizer`
- Use a different port: `--port 5556`

### Visualizer shows blank/no drones
- Make sure simulation started first
- Check you're connecting to the correct port
- Wait a few seconds for connection

### Drones flying out of bounds
The boundary system should push them back. If they escape:
- This is a known issue being fixed
- Restart the simulation

---

## Running Both at Once (Single Command)

```bash
# Start simulation in background, then visualizer
python scripts/test_visualizer.py --red 1 --blue 1 --port 5555 &
sleep 3
python -m visualization.renderer.dogfight_viewer --connect localhost:5555
```

To stop everything:
```bash
pkill -f "test_visualizer|dogfight_viewer"
```

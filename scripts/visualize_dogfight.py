#!/usr/bin/env python3
"""
Dogfight Visualizer

Renders dogfight combat in real-time using matplotlib 3D animation.
Shows drone positions, trails, and combat events.

Usage:
    python scripts/visualize_dogfight.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from collections import deque
import sys

# Add project root to path
sys.path.insert(0, '.')

from simulation.environments.combat import DogfightEnv, DogfightConfig


class DogfightVisualizer:
    """Real-time 3D visualization of dogfight combat."""

    def __init__(self, env: DogfightEnv, trail_length: int = 100):
        self.env = env
        self.trail_length = trail_length

        # Track positions for trails
        self.trails = {drone_id: deque(maxlen=trail_length)
                      for drone_id in env.drones.keys()}

        # Combat event markers
        self.hit_markers = []  # (position, time)
        self.kill_markers = []

        # Setup figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Stats display
        self.stats_text = None

        self._setup_plot()

    def _setup_plot(self):
        """Setup the 3D plot."""
        arena = self.env.config.arena_size / 2

        self.ax.set_xlim(-arena, arena)
        self.ax.set_ylim(-arena, arena)
        self.ax.set_zlim(
            self.env.config.arena_height_min,
            self.env.config.arena_height_max
        )

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Altitude (m)')
        self.ax.set_title('🛩️ Fixed-Wing Dogfight Combat 🛩️', fontsize=14, fontweight='bold')

        # Draw arena boundaries
        self._draw_arena()

        # Legend
        red_patch = mpatches.Patch(color='red', label='Red Team')
        blue_patch = mpatches.Patch(color='blue', label='Blue Team')
        self.ax.legend(handles=[red_patch, blue_patch], loc='upper left')

    def _draw_arena(self):
        """Draw arena boundary box."""
        arena = self.env.config.arena_size / 2
        z_min = self.env.config.arena_height_min
        z_max = self.env.config.arena_height_max

        # Draw floor grid
        x = np.linspace(-arena, arena, 10)
        y = np.linspace(-arena, arena, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.ones_like(X) * z_min
        self.ax.plot_wireframe(X, Y, Z, alpha=0.1, color='gray')

    def update(self, frame):
        """Update animation frame."""
        # Clear old artists
        self.ax.cla()
        self._setup_plot()

        # Step environment with simple pursuit policy
        obs = self._get_pursuit_action()
        action = obs
        _, reward, terminated, truncated, info = self.env.step(action)

        # Update trails
        for drone_id, drone in self.env.drones.items():
            self.trails[drone_id].append(drone.position.copy())

        # Draw drones and trails
        for drone_id, drone in self.env.drones.items():
            color = 'red' if drone.team == 0 else 'blue'

            if drone.is_alive:
                # Draw drone as triangle/arrow
                self.ax.scatter(
                    drone.position[0],
                    drone.position[1],
                    drone.position[2],
                    c=color, s=200, marker='^',
                    edgecolors='white', linewidths=2
                )

                # Draw velocity vector
                vel_scale = 2.0
                self.ax.quiver(
                    drone.position[0], drone.position[1], drone.position[2],
                    drone.velocity[0] * vel_scale,
                    drone.velocity[1] * vel_scale,
                    drone.velocity[2] * vel_scale,
                    color=color, alpha=0.5, arrow_length_ratio=0.3
                )
            else:
                # Dead drone - show as X
                self.ax.scatter(
                    drone.position[0],
                    drone.position[1],
                    drone.position[2],
                    c=color, s=100, marker='x', alpha=0.5
                )

            # Draw trail
            if len(self.trails[drone_id]) > 1:
                trail = np.array(self.trails[drone_id])
                alphas = np.linspace(0.1, 0.6, len(trail))
                for i in range(1, len(trail)):
                    self.ax.plot3D(
                        trail[i-1:i+1, 0],
                        trail[i-1:i+1, 1],
                        trail[i-1:i+1, 2],
                        color=color, alpha=alphas[i], linewidth=1
                    )

        # Process combat events
        combat_events = info.get('combat_events', [])
        for event in combat_events:
            if event['type'] == 'hit':
                target = self.env.drones.get(event['target'])
                if target:
                    self.hit_markers.append((target.position.copy(), frame))
            elif event['type'] == 'kill':
                target = self.env.drones.get(event['target'])
                if target:
                    self.kill_markers.append((target.position.copy(), frame))

        # Draw hit markers (yellow bursts that fade)
        for pos, hit_frame in self.hit_markers[-20:]:
            age = frame - hit_frame
            if age < 30:
                alpha = 1 - age / 30
                self.ax.scatter(
                    pos[0], pos[1], pos[2],
                    c='yellow', s=100 * (1 - age/30),
                    marker='*', alpha=alpha
                )

        # Draw kill markers (explosion effect)
        for pos, kill_frame in self.kill_markers[-10:]:
            age = frame - kill_frame
            if age < 60:
                alpha = 1 - age / 60
                size = 50 + age * 5
                self.ax.scatter(
                    pos[0], pos[1], pos[2],
                    c='orange', s=size, marker='o', alpha=alpha * 0.5
                )

        # Stats overlay
        stats_text = (
            f"Time: {self.env.match_time:.1f}s\n"
            f"Red Kills: {self.env.red_kills}\n"
            f"Blue Kills: {self.env.blue_kills}\n"
            f"Frame: {frame}"
        )
        self.ax.text2D(
            0.02, 0.98, stats_text,
            transform=self.ax.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Check for match end
        if terminated or truncated:
            winner = "RED WINS!" if self.env.red_kills > self.env.blue_kills else \
                     "BLUE WINS!" if self.env.blue_kills > self.env.red_kills else \
                     "DRAW!"
            self.ax.text2D(
                0.5, 0.5, winner,
                transform=self.ax.transAxes,
                fontsize=24, fontweight='bold',
                ha='center', va='center',
                color='gold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8)
            )

        return []

    def _get_pursuit_action(self):
        """Generate pursuit action based on current state."""
        agent = self.env.drones[self.env.agent_drone_id]

        # Find nearest enemy
        enemies = [d for d in self.env.drones.values()
                   if d.team != agent.team and d.is_alive]

        if not enemies:
            return np.array([0, 0, 0.5, 0, 0, 0])

        nearest = min(enemies, key=lambda e: np.linalg.norm(e.position - agent.position))

        # Calculate pursuit
        to_enemy = nearest.position - agent.position
        distance = np.linalg.norm(to_enemy)
        to_enemy_norm = to_enemy / (distance + 1e-6)

        # Current heading
        yaw = agent.orientation[2]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])

        # Turn commands
        cross = np.cross(heading, to_enemy_norm)
        roll_cmd = np.clip(cross[2] * 3, -1, 1)

        alt_diff = nearest.position[2] - agent.position[2]
        pitch_cmd = np.clip(alt_diff / 200, -1, 1)

        # Fire if close and aligned
        alignment = np.dot(heading, to_enemy_norm)
        fire = 1.0 if distance < 400 and alignment > 0.8 else 0.0

        return np.array([roll_cmd, pitch_cmd, 0.75, 0, fire, 0])

    def run(self, frames: int = 2000, interval: int = 16):
        """Run the visualization."""
        print("Starting dogfight visualization...")
        print("Close the window to exit.")
        print()

        # Reset environment
        self.env.reset()

        # Clear trails
        for trail in self.trails.values():
            trail.clear()

        self.hit_markers = []
        self.kill_markers = []

        # Create animation
        ani = FuncAnimation(
            self.fig, self.update,
            frames=frames, interval=interval,
            blit=False, repeat=False
        )

        plt.tight_layout()
        plt.show()


def main():
    print("=" * 60)
    print("DOGFIGHT COMBAT VISUALIZER")
    print("=" * 60)
    print()

    # Create environment with exciting config
    config = DogfightConfig(
        num_red=1,
        num_blue=1,
        arena_size=800.0,
        arena_height_min=100.0,
        arena_height_max=800.0,
        respawn_enabled=True,
        kills_to_win=5,
        max_match_time=120.0,
        weapons_config=[
            {"type": "gun", "ammo": 1000, "range": 300, "damage": 30, "cooldown": 0.08},
        ],
    )

    env = DogfightEnv(config=config)

    print(f"Arena: {config.arena_size}m x {config.arena_size}m")
    print(f"Altitude: {config.arena_height_min}m - {config.arena_height_max}m")
    print(f"First to {config.kills_to_win} kills wins!")
    print()

    # Create and run visualizer
    viz = DogfightVisualizer(env)
    viz.run(frames=5000, interval=16)  # ~60 FPS

    env.close()
    print("Visualization complete!")


if __name__ == "__main__":
    main()

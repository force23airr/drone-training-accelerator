#!/usr/bin/env python3
"""
2D Top-Down Dogfight Visualizer

Simpler visualization showing combat from above.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow, FancyArrow
from collections import deque
import sys

sys.path.insert(0, '.')
from simulation.environments.combat import DogfightEnv, DogfightConfig


def run_2d_dogfight():
    print("=" * 50)
    print("2D DOGFIGHT VIEWER")
    print("=" * 50)

    # Config
    config = DogfightConfig(
        num_red=1,
        num_blue=1,
        arena_size=600.0,
        respawn_enabled=True,
        kills_to_win=5,
        max_match_time=90.0,
        weapons_config=[
            {"type": "gun", "ammo": 1000, "range": 250, "damage": 35, "cooldown": 0.08},
        ],
    )

    env = DogfightEnv(config=config)
    env.reset()

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10))
    arena = config.arena_size / 2
    ax.set_xlim(-arena, arena)
    ax.set_ylim(-arena, arena)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_title('Dogfight Combat - Top Down View', color='white', fontsize=14)
    fig.patch.set_facecolor('#0f0f1a')

    # Trails
    trails = {d_id: deque(maxlen=80) for d_id in env.drones.keys()}
    explosions = []

    def update(frame):
        ax.clear()
        ax.set_xlim(-arena, arena)
        ax.set_ylim(-arena, arena)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.2, color='white')

        # Get action
        agent = env.drones[env.agent_drone_id]
        enemies = [d for d in env.drones.values() if d.team != agent.team and d.is_alive]

        if enemies:
            nearest = min(enemies, key=lambda e: np.linalg.norm(e.position - agent.position))
            to_enemy = nearest.position - agent.position
            distance = np.linalg.norm(to_enemy)

            yaw = agent.orientation[2]
            heading = np.array([np.cos(yaw), np.sin(yaw), 0])
            cross = np.cross(heading, to_enemy / (distance + 1e-6))

            roll = np.clip(cross[2] * 3, -1, 1)
            pitch = np.clip((nearest.position[2] - agent.position[2]) / 200, -1, 1)
            fire = 1.0 if distance < 300 and np.dot(heading, to_enemy/(distance+1e-6)) > 0.85 else 0
            action = np.array([roll, pitch, 0.8, 0, fire, 0])
        else:
            action = np.array([0, 0, 0.5, 0, 0, 0])

        # Step
        _, _, terminated, truncated, info = env.step(action)

        # Draw arena border
        rect = plt.Rectangle((-arena, -arena), arena*2, arena*2,
                            fill=False, edgecolor='yellow', linewidth=2)
        ax.add_patch(rect)

        # Update trails and draw
        for d_id, drone in env.drones.items():
            trails[d_id].append(drone.position[:2].copy())
            color = '#ff4444' if drone.team == 0 else '#4444ff'
            light_color = '#ff8888' if drone.team == 0 else '#8888ff'

            # Draw trail
            if len(trails[d_id]) > 1:
                trail = np.array(trails[d_id])
                ax.plot(trail[:, 0], trail[:, 1], color=light_color,
                       alpha=0.4, linewidth=2)

            if drone.is_alive:
                # Draw drone body
                ax.scatter(drone.position[0], drone.position[1],
                          c=color, s=300, marker='o', zorder=5,
                          edgecolors='white', linewidths=2)

                # Draw heading arrow
                yaw = drone.orientation[2]
                dx = np.cos(yaw) * 40
                dy = np.sin(yaw) * 40
                ax.arrow(drone.position[0], drone.position[1], dx, dy,
                        head_width=15, head_length=10, fc=color, ec='white',
                        zorder=6, linewidth=1)

                # Health bar
                health_pct = drone.health / drone.max_health
                bar_width = 40
                bar_x = drone.position[0] - bar_width/2
                bar_y = drone.position[1] + 30
                ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, 6,
                            facecolor='#333', edgecolor='white', linewidth=1))
                ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * health_pct, 6,
                            facecolor='#44ff44' if health_pct > 0.5 else '#ffff44' if health_pct > 0.25 else '#ff4444'))
            else:
                # Dead - show X
                ax.scatter(drone.position[0], drone.position[1],
                          c='gray', s=200, marker='x', zorder=5)

        # Combat events
        for event in info.get('combat_events', []):
            if event['type'] in ['hit', 'kill']:
                target = env.drones.get(event['target'])
                if target:
                    explosions.append({
                        'pos': target.position[:2].copy(),
                        'frame': frame,
                        'type': event['type']
                    })

        # Draw explosions
        for exp in explosions[-15:]:
            age = frame - exp['frame']
            if age < 20:
                alpha = 1 - age/20
                size = 100 + age * 30
                color = '#ffff00' if exp['type'] == 'hit' else '#ff8800'
                ax.scatter(exp['pos'][0], exp['pos'][1],
                          c=color, s=size, marker='*', alpha=alpha, zorder=4)

        # Score display
        score_text = f"RED {env.red_kills} - {env.blue_kills} BLUE"
        ax.text(0, arena - 30, score_text, ha='center', va='top',
               fontsize=20, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))

        # Time
        ax.text(-arena + 20, arena - 20, f"Time: {env.match_time:.1f}s",
               fontsize=12, color='white')

        # Match end
        if terminated or truncated:
            if env.red_kills > env.blue_kills:
                result = "RED WINS!"
                result_color = '#ff4444'
            elif env.blue_kills > env.red_kills:
                result = "BLUE WINS!"
                result_color = '#4444ff'
            else:
                result = "DRAW!"
                result_color = '#ffffff'

            ax.text(0, 0, result, ha='center', va='center',
                   fontsize=36, fontweight='bold', color=result_color,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))

        ax.set_title(f'Dogfight - Frame {frame}', color='white', fontsize=14)

        return []

    print("Launching 2D visualization...")
    print("Red = You, Blue = Opponent")
    print()

    ani = FuncAnimation(fig, update, frames=4000, interval=16, blit=False)
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    run_2d_dogfight()

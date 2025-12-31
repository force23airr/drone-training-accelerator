"""
Swarm Coordination Environment

Multi-agent environment for drone swarm coordination and formation flying.
Features inter-agent communication, collision avoidance, and formation control.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from gymnasium import spaces
import pybullet as p

from simulation.environments.base_drone_env import BaseDroneEnv
from simulation.environments.environmental_conditions import (
    EnvironmentalConditions,
    WeatherType,
    TimeOfDay,
    TerrainType,
    WindModel,
    create_clear_day,
)


class SwarmCoordinationEnv(BaseDroneEnv):
    """
    Multi-agent swarm coordination environment.

    Mission objectives:
    - Maintain formation geometry
    - Avoid inter-agent collisions
    - Coordinate movement through waypoints
    - Handle agent failures gracefully

    Features:
    - Multiple drone instances
    - Formation patterns (line, V, diamond, grid)
    - Communication simulation
    - Leader-follower dynamics
    """

    # Pre-defined formation patterns
    FORMATIONS = {
        "line": lambda n: [(i * 2, 0, 0) for i in range(n)],
        "v_formation": lambda n: [
            (abs(i - n // 2) * 2, (i - n // 2) * 2, 0)
            for i in range(n)
        ],
        "diamond": lambda n: [
            (np.cos(2 * np.pi * i / n) * 3, np.sin(2 * np.pi * i / n) * 3, 0)
            for i in range(n)
        ],
        "grid": lambda n: [
            ((i % int(np.sqrt(n))) * 2, (i // int(np.sqrt(n))) * 2, 0)
            for i in range(n)
        ],
    }

    def __init__(
        self,
        platform_config: Dict[str, Any],
        num_agents: int = 4,
        formation: str = "diamond",
        agent_id: int = 0,
        communication_range: float = 20.0,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize swarm coordination environment.

        Args:
            platform_config: Drone platform configuration
            num_agents: Total number of agents in swarm
            formation: Formation type ('line', 'v_formation', 'diamond', 'grid')
            agent_id: This agent's ID (0 = leader)
            communication_range: Range for inter-agent communication
            render_mode: Rendering mode
        """
        conditions = create_clear_day()
        # Add some wind for challenge
        conditions.wind.base_speed = 3.0
        conditions.wind.turbulence_intensity = 0.5

        self.num_agents = num_agents
        self.formation_type = formation
        self.agent_id = agent_id
        self.communication_range = communication_range
        self.is_leader = (agent_id == 0)

        # Agent state tracking
        self._other_agents: List[Dict[str, Any]] = []
        self._formation_offsets: List[Tuple[float, float, float]] = []
        self._formation_center = np.array([0.0, 0.0, 5.0])

        # Formation error tracking
        self._formation_error = 0.0
        self._collision_with_agents = 0

        # Extend observation for swarm awareness
        original_obs_dim = platform_config.get("observation_dim", 13)
        # Add: formation center (3), nearest neighbor relative pos (3),
        # formation error (1), num neighbors in range (1)
        platform_config["observation_dim"] = original_obs_dim + 8

        super().__init__(
            platform_config=platform_config,
            environmental_conditions=conditions,
            render_mode=render_mode,
            **kwargs
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_obs_dim + 8,),
            dtype=np.float32
        )

    def _setup_environment(self):
        """Set up swarm environment."""
        self._calculate_formation_offsets()
        self._spawn_other_agents()
        self._update_formation_center()

    def _calculate_formation_offsets(self):
        """Calculate formation position offsets."""
        if self.formation_type in self.FORMATIONS:
            offsets = self.FORMATIONS[self.formation_type](self.num_agents)
            self._formation_offsets = [np.array(o) for o in offsets]
        else:
            # Default to diamond if unknown
            offsets = self.FORMATIONS["diamond"](self.num_agents)
            self._formation_offsets = [np.array(o) for o in offsets]

    def _spawn_other_agents(self):
        """Spawn simulated other agents."""
        self._other_agents = []

        for i in range(self.num_agents):
            if i == self.agent_id:
                continue

            # Calculate initial position based on formation
            offset = self._formation_offsets[i]
            position = self._formation_center + offset

            # Create visual for other agent
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.02],
                rgbaColor=[0.8, 0.2, 0.2, 0.8] if i == 0 else [0.2, 0.8, 0.2, 0.8],
                physicsClientId=self.physics_client
            )
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.02],
                physicsClientId=self.physics_client
            )

            body_id = p.createMultiBody(
                baseMass=0,  # Kinematic, not dynamic
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=position.tolist(),
                physicsClientId=self.physics_client
            )

            self._other_agents.append({
                "id": i,
                "body_id": body_id,
                "position": position.copy(),
                "velocity": np.zeros(3),
                "is_leader": i == 0,
            })
            self._obstacle_ids.append(body_id)

    def _update_other_agents(self):
        """Update simulated agent positions (simple formation following)."""
        # Move formation center (leader behavior)
        if self._step_count % 100 == 0 and self.is_leader:
            # Leader occasionally changes direction
            self._formation_center += np.random.uniform(-0.5, 0.5, 3)
            self._formation_center[2] = np.clip(self._formation_center[2], 2, 10)

        # Update each agent to follow formation
        for agent in self._other_agents:
            target = self._formation_center + self._formation_offsets[agent["id"]]

            # Simple proportional control to target
            error = target - agent["position"]
            agent["velocity"] = error * 0.5  # Simple P controller
            agent["position"] += agent["velocity"] / self.control_hz

            # Update visual
            p.resetBasePositionAndOrientation(
                agent["body_id"],
                agent["position"].tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.physics_client
            )

    def _update_formation_center(self):
        """Update formation center based on current positions."""
        if self.is_leader:
            # Leader defines formation center
            if self.drone_id is not None:
                pos, _ = p.getBasePositionAndOrientation(
                    self.drone_id, physicsClientId=self.physics_client
                )
                # Formation center is leader position minus leader's offset
                self._formation_center = np.array(pos) - self._formation_offsets[0]
        else:
            # Followers track leader's formation center
            leader = next((a for a in self._other_agents if a["is_leader"]), None)
            if leader:
                self._formation_center = leader["position"] - self._formation_offsets[0]

    def _calculate_formation_error(self, drone_pos: np.ndarray) -> float:
        """Calculate this agent's formation error."""
        target = self._formation_center + self._formation_offsets[self.agent_id]
        return float(np.linalg.norm(drone_pos - target))

    def _get_nearest_neighbor(self, drone_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get relative position to nearest neighbor."""
        min_dist = float('inf')
        nearest_rel = np.zeros(3)

        for agent in self._other_agents:
            rel = agent["position"] - drone_pos
            dist = np.linalg.norm(rel)
            if dist < min_dist:
                min_dist = dist
                nearest_rel = rel

        return nearest_rel, min_dist

    def _count_neighbors_in_range(self, drone_pos: np.ndarray) -> int:
        """Count agents within communication range."""
        count = 0
        for agent in self._other_agents:
            dist = np.linalg.norm(agent["position"] - drone_pos)
            if dist <= self.communication_range:
                count += 1
        return count

    def _check_agent_collision(self, drone_pos: np.ndarray) -> bool:
        """Check for collision with other agents."""
        collision_radius = 0.3

        for agent in self._other_agents:
            dist = np.linalg.norm(agent["position"] - drone_pos)
            if dist < collision_radius:
                return True
        return False

    def _get_observation(self) -> np.ndarray:
        """Get swarm-aware observation."""
        base_obs = super()._get_observation()
        drone_pos = base_obs[:3]

        # Update formation center
        self._update_formation_center()

        # Formation center relative position
        rel_center = self._formation_center - drone_pos

        # Nearest neighbor
        nearest_rel, nearest_dist = self._get_nearest_neighbor(drone_pos)

        # Formation error
        formation_error = self._calculate_formation_error(drone_pos)
        self._formation_error = formation_error

        # Neighbors in range
        neighbors_in_range = self._count_neighbors_in_range(drone_pos)

        return np.concatenate([
            base_obs,
            rel_center,
            nearest_rel,
            [formation_error],
            [float(neighbors_in_range)]
        ]).astype(np.float32)

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute swarm coordination reward."""
        pos = obs[:3]
        vel = obs[3:6]

        reward = 0.0

        # Update other agents
        self._update_other_agents()

        # Formation keeping reward
        formation_error = self._calculate_formation_error(pos)
        reward += 2.0 * np.exp(-formation_error)  # Exponential reward for closeness

        # Penalty for large formation error
        if formation_error > 1.0:
            reward -= formation_error

        # Collision avoidance
        if self._check_agent_collision(pos):
            reward -= 50.0
            self._collision_with_agents += 1

        # Maintain safe distance from neighbors
        _, nearest_dist = self._get_nearest_neighbor(pos)
        if nearest_dist < 0.5:
            reward -= 10.0 * (0.5 - nearest_dist)
        elif nearest_dist > 5.0:
            reward -= 0.5 * (nearest_dist - 5.0)  # Penalty for being too far

        # Communication maintenance
        neighbors = self._count_neighbors_in_range(pos)
        if neighbors < self.num_agents - 2:
            reward -= 2.0  # Penalty for losing communication

        # Velocity matching (simplified)
        target_vel = np.zeros(3)  # Formation should hover
        vel_error = np.linalg.norm(vel - target_vel)
        reward -= 0.1 * vel_error

        # Action smoothness
        reward -= 0.01 * np.sum(action**2)

        # Altitude maintenance
        target_alt = self._formation_center[2] + self._formation_offsets[self.agent_id][2]
        alt_error = abs(pos[2] - target_alt)
        reward -= 0.2 * alt_error

        # Leader bonus for good formation
        if self.is_leader:
            avg_formation_error = np.mean([
                np.linalg.norm(
                    a["position"] - (self._formation_center + self._formation_offsets[a["id"]])
                )
                for a in self._other_agents
            ])
            if avg_formation_error < 0.5:
                reward += 5.0

        return float(reward)

    def _check_termination(self, obs: np.ndarray) -> Tuple[bool, bool]:
        """Check termination conditions."""
        pos = obs[:3]
        terminated = False

        # Ground collision
        if pos[2] < 0.1:
            terminated = True

        # Out of bounds
        if abs(pos[0]) > 50 or abs(pos[1]) > 50 or pos[2] > 30:
            terminated = True

        # Agent collision
        if self._check_agent_collision(pos):
            terminated = True

        # Lost communication (too far from all agents)
        if self._count_neighbors_in_range(pos) == 0:
            terminated = True

        # Formation error too large for too long
        if self._formation_error > 10.0:
            terminated = True

        # Max steps
        max_steps = self.platform_config.get("max_episode_steps", 2000)
        truncated = self._step_count >= max_steps

        return terminated, truncated

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step."""
        obs, reward, terminated, truncated, info = super().step(action)

        info["formation_error"] = self._formation_error
        info["agent_collisions"] = self._collision_with_agents
        info["neighbors_in_range"] = self._count_neighbors_in_range(obs[:3])
        info["is_leader"] = self.is_leader
        info["formation_type"] = self.formation_type

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        self._collision_with_agents = 0
        self._formation_error = 0.0
        self._formation_center = np.array([0.0, 0.0, 5.0])

        obs, info = super().reset(seed=seed, options=options)

        info["num_agents"] = self.num_agents
        info["formation_type"] = self.formation_type
        info["agent_id"] = self.agent_id
        info["is_leader"] = self.is_leader

        return obs, info

    def get_formation_positions(self) -> Dict[int, np.ndarray]:
        """Get current positions of all agents."""
        positions = {}

        # This agent
        if self.drone_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self.drone_id, physicsClientId=self.physics_client
            )
            positions[self.agent_id] = np.array(pos)

        # Other agents
        for agent in self._other_agents:
            positions[agent["id"]] = agent["position"].copy()

        return positions

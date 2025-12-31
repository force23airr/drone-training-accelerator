"""
Mission Suites

Defines training mission configurations for different skill levels
and operational objectives.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional
import numpy as np


@dataclass
class MissionConfig:
    """Configuration for a training mission."""
    name: str
    description: str
    difficulty: str  # beginner, intermediate, advanced, expert
    objectives: List[str]
    success_criteria: Dict[str, float]
    reward_weights: Dict[str, float]
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    environment_params: Dict[str, Any] = field(default_factory=dict)


# Registry of mission suites
_MISSION_REGISTRY: Dict[str, MissionConfig] = {}


def register_mission(mission_id: str, config: MissionConfig):
    """Register a new mission configuration."""
    _MISSION_REGISTRY[mission_id] = config


def list_missions() -> List[str]:
    """List all available mission IDs."""
    return list(_MISSION_REGISTRY.keys())


def get_missions_by_difficulty(difficulty: str) -> List[str]:
    """Get all missions of a specific difficulty."""
    return [
        mid for mid, config in _MISSION_REGISTRY.items()
        if config.difficulty == difficulty
    ]


class MissionSuite:
    """
    Mission suite for structured drone training.

    Provides reward shaping, curriculum learning, and success evaluation
    for specific mission objectives.
    """

    def __init__(self, mission_id: str):
        """
        Initialize mission suite.

        Args:
            mission_id: Identifier for the mission type
        """
        if mission_id not in _MISSION_REGISTRY:
            available = ", ".join(_MISSION_REGISTRY.keys())
            raise ValueError(
                f"Unknown mission '{mission_id}'. Available: {available}"
            )

        self.config = _MISSION_REGISTRY[mission_id]
        self.name = self.config.name
        self.difficulty = self.config.difficulty

        # Curriculum tracking
        self._current_stage = 0
        self._stage_progress = 0.0

    def compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        Compute mission-specific reward.

        Args:
            obs: Current observation
            action: Action taken
            next_obs: Resulting observation
            info: Additional environment info

        Returns:
            Shaped reward value
        """
        reward = 0.0
        weights = self.config.reward_weights

        # Extract state components
        pos = next_obs[:3]
        vel = next_obs[3:6]
        orientation = next_obs[6:10]

        # Base survival reward
        if "alive" in weights:
            reward += weights["alive"]

        # Position-based rewards
        if "position" in weights:
            target_pos = self._get_target_position()
            pos_error = np.linalg.norm(pos - target_pos)
            reward -= weights["position"] * pos_error

        # Velocity-based rewards
        if "velocity" in weights:
            target_vel = self._get_target_velocity()
            vel_error = np.linalg.norm(vel - target_vel)
            reward -= weights["velocity"] * vel_error

        # Stability rewards
        if "stability" in weights:
            # Penalize excessive angular rates
            ang_vel = next_obs[10:13]
            reward -= weights["stability"] * np.linalg.norm(ang_vel)

        # Action smoothness
        if "action_smooth" in weights:
            reward -= weights["action_smooth"] * np.sum(action**2)

        # Energy efficiency
        if "energy" in weights:
            reward -= weights["energy"] * np.sum(np.abs(action))

        return float(reward)

    def check_success(
        self,
        obs: np.ndarray,
        episode_stats: Dict[str, Any]
    ) -> bool:
        """
        Check if mission success criteria are met.

        Args:
            obs: Current observation
            episode_stats: Cumulative episode statistics

        Returns:
            True if mission is successfully completed
        """
        criteria = self.config.success_criteria

        for criterion, threshold in criteria.items():
            if criterion == "min_hover_time":
                if episode_stats.get("hover_time", 0) < threshold:
                    return False
            elif criterion == "max_position_error":
                pos_error = episode_stats.get("avg_position_error", float("inf"))
                if pos_error > threshold:
                    return False
            elif criterion == "waypoints_reached":
                reached = episode_stats.get("waypoints_reached", 0)
                if reached < threshold:
                    return False
            elif criterion == "min_survival_time":
                if episode_stats.get("survival_time", 0) < threshold:
                    return False

        return True

    def get_curriculum_params(self) -> Dict[str, Any]:
        """Get current curriculum stage parameters."""
        if not self.config.curriculum_stages:
            return {}

        stage_idx = min(self._current_stage, len(self.config.curriculum_stages) - 1)
        return self.config.curriculum_stages[stage_idx]

    def advance_curriculum(self, success_rate: float):
        """
        Potentially advance to next curriculum stage.

        Args:
            success_rate: Recent success rate (0-1)
        """
        if not self.config.curriculum_stages:
            return

        # Advance if success rate exceeds threshold
        if success_rate > 0.8 and self._current_stage < len(self.config.curriculum_stages) - 1:
            self._current_stage += 1
            self._stage_progress = 0.0
            print(f"Curriculum advanced to stage {self._current_stage + 1}")

    def _get_target_position(self) -> np.ndarray:
        """Get current target position based on mission and curriculum."""
        params = self.get_curriculum_params()
        return np.array(params.get("target_position", [0, 0, 1.0]))

    def _get_target_velocity(self) -> np.ndarray:
        """Get current target velocity."""
        params = self.get_curriculum_params()
        return np.array(params.get("target_velocity", [0, 0, 0]))

    def get_mission_info(self) -> Dict[str, Any]:
        """Get mission information for logging."""
        return {
            "mission_name": self.name,
            "difficulty": self.difficulty,
            "objectives": self.config.objectives,
            "current_stage": self._current_stage,
            "stage_progress": self._stage_progress,
        }


# =============================================================================
# BEGINNER MISSIONS
# =============================================================================

register_mission(
    "hover_stability",
    MissionConfig(
        name="Hover Stability",
        description="Learn to maintain stable hover at a fixed position",
        difficulty="beginner",
        objectives=[
            "Maintain altitude within 0.1m of target",
            "Minimize horizontal drift",
            "Achieve stable hover for 10+ seconds",
        ],
        success_criteria={
            "min_hover_time": 10.0,
            "max_position_error": 0.2,
        },
        reward_weights={
            "alive": 1.0,
            "position": 0.5,
            "velocity": 0.3,
            "stability": 0.2,
            "action_smooth": 0.01,
        },
        curriculum_stages=[
            {"target_position": [0, 0, 0.5], "max_wind": 0},
            {"target_position": [0, 0, 1.0], "max_wind": 0},
            {"target_position": [0, 0, 1.5], "max_wind": 1.0},
            {"target_position": [0, 0, 2.0], "max_wind": 2.0},
        ]
    )
)

register_mission(
    "takeoff_landing",
    MissionConfig(
        name="Takeoff and Landing",
        description="Execute controlled takeoff and landing sequences",
        difficulty="beginner",
        objectives=[
            "Smooth vertical takeoff to target altitude",
            "Controlled descent and landing",
            "Minimize ground impact velocity",
        ],
        success_criteria={
            "max_landing_velocity": 0.5,
            "min_survival_time": 15.0,
        },
        reward_weights={
            "alive": 1.0,
            "position": 0.4,
            "velocity": 0.4,
            "stability": 0.2,
        },
        curriculum_stages=[
            {"target_altitude": 0.5, "landing_zone_radius": 0.5},
            {"target_altitude": 1.0, "landing_zone_radius": 0.3},
            {"target_altitude": 2.0, "landing_zone_radius": 0.2},
        ]
    )
)

# =============================================================================
# INTERMEDIATE MISSIONS
# =============================================================================

register_mission(
    "waypoint_navigation",
    MissionConfig(
        name="Waypoint Navigation",
        description="Navigate through a sequence of waypoints",
        difficulty="intermediate",
        objectives=[
            "Reach each waypoint within tolerance",
            "Maintain smooth flight path",
            "Complete course within time limit",
        ],
        success_criteria={
            "waypoints_reached": 5,
            "max_position_error": 0.5,
        },
        reward_weights={
            "alive": 0.5,
            "position": 1.0,
            "velocity": 0.2,
            "stability": 0.3,
            "energy": 0.1,
        },
        curriculum_stages=[
            {"num_waypoints": 3, "waypoint_radius": 0.5, "max_distance": 5},
            {"num_waypoints": 5, "waypoint_radius": 0.3, "max_distance": 10},
            {"num_waypoints": 8, "waypoint_radius": 0.2, "max_distance": 15},
        ]
    )
)

register_mission(
    "trajectory_tracking",
    MissionConfig(
        name="Trajectory Tracking",
        description="Follow a predefined trajectory precisely",
        difficulty="intermediate",
        objectives=[
            "Minimize deviation from reference trajectory",
            "Match velocity profile",
            "Handle trajectory transitions smoothly",
        ],
        success_criteria={
            "max_position_error": 0.3,
            "min_survival_time": 30.0,
        },
        reward_weights={
            "alive": 0.5,
            "position": 1.5,
            "velocity": 0.8,
            "stability": 0.3,
            "action_smooth": 0.05,
        }
    )
)

# =============================================================================
# ADVANCED MISSIONS
# =============================================================================

register_mission(
    "obstacle_avoidance",
    MissionConfig(
        name="Obstacle Avoidance",
        description="Navigate through environments with static and dynamic obstacles",
        difficulty="advanced",
        objectives=[
            "Avoid all obstacles",
            "Reach goal position",
            "Minimize flight time",
        ],
        success_criteria={
            "collisions": 0,
            "goal_reached": 1,
        },
        reward_weights={
            "alive": 1.0,
            "position": 0.8,
            "collision_penalty": -10.0,
            "goal_bonus": 10.0,
            "time_penalty": 0.01,
        },
        environment_params={
            "num_static_obstacles": 10,
            "num_dynamic_obstacles": 3,
        }
    )
)

register_mission(
    "wind_disturbance",
    MissionConfig(
        name="Wind Disturbance Rejection",
        description="Maintain stable flight under varying wind conditions",
        difficulty="advanced",
        objectives=[
            "Maintain position under wind gusts",
            "Recover from disturbances quickly",
            "Adapt to changing wind patterns",
        ],
        success_criteria={
            "max_position_error": 0.5,
            "min_survival_time": 60.0,
        },
        reward_weights={
            "alive": 1.0,
            "position": 1.0,
            "velocity": 0.3,
            "stability": 0.5,
        },
        environment_params={
            "wind_speed_range": [0, 10],
            "gust_probability": 0.1,
            "gust_duration": [0.5, 2.0],
        }
    )
)

# =============================================================================
# EXPERT MISSIONS
# =============================================================================

register_mission(
    "aggressive_maneuvers",
    MissionConfig(
        name="Aggressive Maneuvers",
        description="Execute high-speed acrobatic maneuvers",
        difficulty="expert",
        objectives=[
            "Complete flip/roll maneuvers",
            "High-speed cornering",
            "Rapid altitude changes",
        ],
        success_criteria={
            "maneuvers_completed": 3,
            "min_survival_time": 30.0,
        },
        reward_weights={
            "alive": 0.5,
            "maneuver_completion": 5.0,
            "stability": 0.1,
            "action_smooth": 0.01,
        }
    )
)

register_mission(
    "multi_agent_coordination",
    MissionConfig(
        name="Multi-Agent Coordination",
        description="Coordinate with other drones for formation flying",
        difficulty="expert",
        objectives=[
            "Maintain formation geometry",
            "Avoid inter-agent collisions",
            "Coordinate through waypoints",
        ],
        success_criteria={
            "formation_error": 0.5,
            "collisions": 0,
        },
        reward_weights={
            "alive": 1.0,
            "formation": 2.0,
            "collision_penalty": -20.0,
            "coordination_bonus": 1.0,
        },
        environment_params={
            "num_agents": 4,
            "formation_type": "diamond",
        }
    )
)

register_mission(
    "autonomous_landing_moving",
    MissionConfig(
        name="Autonomous Landing on Moving Platform",
        description="Land on a moving platform (vehicle, ship, etc.)",
        difficulty="expert",
        objectives=[
            "Track moving platform",
            "Execute precision landing",
            "Handle platform motion uncertainty",
        ],
        success_criteria={
            "landing_success": 1,
            "landing_accuracy": 0.2,
        },
        reward_weights={
            "alive": 1.0,
            "tracking": 1.5,
            "landing_precision": 5.0,
            "velocity_matching": 0.5,
        },
        environment_params={
            "platform_speed": [0, 5],
            "platform_motion": "sinusoidal",
        }
    )
)

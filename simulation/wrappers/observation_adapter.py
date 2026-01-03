"""
Observation Adapter Wrapper

Converts environment observations into the canonical observation schema.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.observation_schema import (
    CANONICAL_OBS_DIM,
    DEFAULT_SPEED_OF_SOUND_M_S,
    canonicalize_observation,
)


@dataclass
class ObservationAdapterConfig:
    """Configuration for observation-space adaptation."""
    quaternion_format: str = "xyzw"  # "xyzw" (PyBullet) or "wxyz"
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S
    energy_default: float = 1.0
    pass_through_if_canonical: bool = True
    store_raw_in_info: bool = False


class ObservationAdapterWrapper(gym.Wrapper):
    """
    Wraps an environment to expose the canonical observation vector.

    This should be applied OUTERMOST so inner wrappers (shield, action adapter)
    still receive the raw observation format they expect.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[ObservationAdapterConfig] = None,
    ):
        super().__init__(env)
        self.config = config or ObservationAdapterConfig()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(CANONICAL_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        canonical = self._to_canonical(obs)
        if self.config.store_raw_in_info:
            info = dict(info)
            info["raw_observation"] = np.array(obs, dtype=np.float32)
        return canonical, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        canonical = self._to_canonical(obs)
        if self.config.store_raw_in_info:
            info = dict(info)
            info["raw_observation"] = np.array(obs, dtype=np.float32)
        return canonical, reward, terminated, truncated, info

    def _to_canonical(self, obs: np.ndarray) -> np.ndarray:
        return canonicalize_observation(
            obs,
            quaternion_format=self.config.quaternion_format,
            speed_of_sound_m_s=self.config.speed_of_sound_m_s,
            energy_default=self.config.energy_default,
            pass_through_if_canonical=self.config.pass_through_if_canonical,
        )


def make_observation_adapted_env(
    env: gym.Env,
    config: Optional[ObservationAdapterConfig] = None,
) -> ObservationAdapterWrapper:
    """Convenience function to wrap an environment with observation adapter."""
    return ObservationAdapterWrapper(env, config=config)

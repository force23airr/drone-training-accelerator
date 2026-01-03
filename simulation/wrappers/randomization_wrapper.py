"""
Domain Randomization Gym Wrapper

Wraps drone environments to apply domain randomization for sim2real transfer.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import deque

from simulation.randomization.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
    SampledDomainParams,
    create_curriculum_configs,
)


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Gym wrapper that applies domain randomization on reset.

    Randomizes:
    - Physics: mass, inertia, CG offset, payload
    - Actuators: motor lag, gain, saturation, battery sag
    - Sensors: IMU bias/noise, GPS dropout
    - Environment: wind, gusts, turbulence
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[DomainRandomizationConfig] = None,
        seed: Optional[int] = None,
        randomize_on_reset: bool = True,
    ):
        """
        Args:
            env: Base environment to wrap
            config: Domain randomization configuration
            seed: Random seed
            randomize_on_reset: Whether to randomize on each reset
        """
        super().__init__(env)

        self.config = config or DomainRandomizationConfig()
        self.randomizer = DomainRandomizer(self.config, seed)
        self.randomize_on_reset = randomize_on_reset

        # State for actuator simulation
        self._motor_state = np.zeros(4)  # Low-pass filter state
        self._command_buffer: deque = deque(maxlen=10)  # For command delay
        self._battery_charge = 1.0  # Simulated battery level
        self._step_count = 0

        # Current domain params
        self._domain_params: Optional[SampledDomainParams] = None

        # Observation buffer for delay
        self._obs_buffer: deque = deque(maxlen=5)

    @property
    def domain_params(self) -> Optional[SampledDomainParams]:
        """Get current domain parameters."""
        return self._domain_params

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with new domain randomization.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            (observation, info)
        """
        # Sample new domain parameters
        if self.randomize_on_reset:
            self._domain_params = self.randomizer.sample()

        # Reset internal state
        self._motor_state = np.zeros(4)
        self._command_buffer.clear()
        self._obs_buffer.clear()
        self._battery_charge = 1.0
        self._step_count = 0

        # Apply physics randomization to environment if supported
        if self._domain_params is not None:
            self._apply_physics_randomization()

        # Reset underlying environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Add domain params to info
        if self._domain_params is not None:
            info['domain_params'] = self._domain_params.to_dict()

        # Apply sensor noise to initial observation
        obs = self._apply_sensor_randomization(obs)

        # Initialize observation buffer
        self._obs_buffer.append(obs)

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step with domain randomization effects.

        Args:
            action: Control action

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Apply actuator randomization to action
        processed_action = self._apply_actuator_randomization(action)

        # Step underlying environment
        obs, reward, terminated, truncated, info = self.env.step(processed_action)

        # Apply sensor randomization to observation
        obs = self._apply_sensor_randomization(obs)

        # Apply observation delay
        obs = self._apply_observation_delay(obs)

        # Add environment disturbances (wind, gusts)
        self._apply_environment_disturbances()

        # Simulate battery drain
        self._simulate_battery_drain()

        return obs, reward, terminated, truncated, info

    def _apply_physics_randomization(self):
        """Apply physics parameters to environment."""
        if self._domain_params is None:
            return

        params = self._domain_params

        # Try to set physics parameters on environment
        if hasattr(self.env, 'set_mass_scale'):
            self.env.set_mass_scale(params.mass_scale)

        if hasattr(self.env, 'set_inertia_scale'):
            self.env.set_inertia_scale(params.inertia_scale)

        if hasattr(self.env, 'set_cg_offset'):
            self.env.set_cg_offset(params.cg_offset)

        if hasattr(self.env, 'set_payload_mass'):
            self.env.set_payload_mass(params.payload_mass)

        if hasattr(self.env, 'set_drag_scale'):
            self.env.set_drag_scale(params.drag_scale)

        # Set wind conditions
        if hasattr(self.env, 'set_wind'):
            wind_vector = params.wind_direction * params.wind_speed
            self.env.set_wind(wind_vector)

        if hasattr(self.env, 'set_turbulence'):
            self.env.set_turbulence(params.turbulence_scale)

        # Alternative: use env_conditions if available
        if hasattr(self.env, 'env_conditions'):
            conditions = self.env.env_conditions
            if hasattr(conditions, 'wind_speed'):
                conditions.wind_speed = params.wind_speed
            if hasattr(conditions, 'wind_direction'):
                conditions.wind_direction = params.wind_direction
            if hasattr(conditions, 'turbulence_intensity'):
                conditions.turbulence_intensity = params.turbulence_scale
            if hasattr(conditions, 'air_density'):
                conditions.air_density = 1.225 * params.air_density_scale

    def _apply_actuator_randomization(self, action: np.ndarray) -> np.ndarray:
        """Apply actuator effects to action."""
        if self._domain_params is None:
            return action

        params = self._domain_params
        action = np.array(action, dtype=np.float32)

        # Apply motor bias
        if len(action) >= 4:
            action[:4] = action[:4] + params.motor_biases

        # Apply gain scaling
        action = action * params.motor_gain_scale

        # Apply saturation
        action = np.clip(action, -params.saturation_level, params.saturation_level)

        # Apply battery voltage sag
        effective_voltage = 1.0 - params.battery_sag_factor * (1.0 - self._battery_charge)
        action = action * effective_voltage

        # Apply motor lag (first-order filter)
        if len(action) >= 4:
            dt = getattr(self.env, 'dt', 0.01)
            tau = max(params.motor_time_constant, 0.001)
            alpha = dt / (tau + dt)

            for i in range(4):
                self._motor_state[i] = (1 - alpha) * self._motor_state[i] + alpha * action[i]
                action[i] = self._motor_state[i]

        # Apply command delay
        if params.command_delay > 0:
            self._command_buffer.append(action.copy())
            delay_steps = int(params.command_delay / getattr(self.env, 'dt', 0.01))

            if len(self._command_buffer) > delay_steps:
                action = self._command_buffer[0]

        return action

    def _apply_sensor_randomization(self, obs: np.ndarray) -> np.ndarray:
        """Apply sensor noise and bias to observation."""
        if self._domain_params is None:
            return obs

        params = self._domain_params
        obs = np.array(obs, dtype=np.float32)

        # Assume observation structure:
        # - Legacy: [position(3), velocity(3), orientation(4), angular_velocity(3), ...]
        # - Canonical: [position(3), velocity(3), rpy(3), angular_velocity(3), ...]
        if len(obs) >= 21:
            pos_slice = slice(0, 3)
            vel_slice = slice(3, 6)
            ang_slice = slice(9, 12)
        elif len(obs) >= 13:
            pos_slice = slice(0, 3)
            vel_slice = slice(3, 6)
            ang_slice = slice(10, 13)
        else:
            return obs

        # Apply GPS noise to position
        gps_noise = np.random.randn(3) * 0.1 * params.gps_noise_scale

        # Simulate GPS dropout
        if np.random.rand() < params.gps_dropout_prob:
            # Hold last position (simulate dropout)
            if len(self._obs_buffer) > 0:
                obs[pos_slice] = self._obs_buffer[-1][pos_slice]
        else:
            obs[pos_slice] = obs[pos_slice] + gps_noise

        # Apply IMU noise to velocity
        accel_noise = params.imu_accel_bias + np.random.randn(3) * 0.1 * params.imu_noise_scale
        obs[vel_slice] = obs[vel_slice] + accel_noise * getattr(self.env, 'dt', 0.01)

        # Apply gyro bias to angular velocity
        obs[ang_slice] = obs[ang_slice] + params.imu_gyro_bias
        gyro_noise = np.random.randn(3) * 0.01 * params.imu_noise_scale
        obs[ang_slice] = obs[ang_slice] + gyro_noise

        return obs

    def _apply_observation_delay(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation delay."""
        if self._domain_params is None or self._domain_params.observation_delay <= 0:
            self._obs_buffer.append(obs)
            return obs

        self._obs_buffer.append(obs)

        delay_steps = int(self._domain_params.observation_delay / getattr(self.env, 'dt', 0.01))

        if len(self._obs_buffer) > delay_steps:
            return self._obs_buffer[-delay_steps - 1]

        return obs

    def _apply_environment_disturbances(self):
        """Apply wind gusts and turbulence."""
        if self._domain_params is None:
            return

        params = self._domain_params

        # Generate gust if intensity > 0
        if params.gust_intensity > 0 and hasattr(self.env, 'apply_force'):
            # Random gust with exponential decay
            if np.random.rand() < 0.01:  # 1% chance per step
                gust_direction = np.random.randn(3)
                gust_direction = gust_direction / (np.linalg.norm(gust_direction) + 1e-8)
                gust_magnitude = np.random.exponential(params.gust_intensity)
                gust_force = gust_direction * gust_magnitude
                self.env.apply_force(gust_force)

    def _simulate_battery_drain(self):
        """Simulate battery charge depletion."""
        if self._domain_params is None:
            return

        # Simple linear drain model
        drain_rate = 0.0001  # Per step
        self._battery_charge = max(0.0, self._battery_charge - drain_rate)

    def set_curriculum_stage(self, stage: int):
        """Set curriculum stage."""
        self.randomizer.set_curriculum_stage(stage)

    def advance_curriculum(self):
        """Advance to next curriculum stage."""
        self.randomizer.advance_curriculum()


class CurriculumRandomizationWrapper(DomainRandomizationWrapper):
    """
    Domain randomization with automatic curriculum progression.

    Advances difficulty based on training progress.
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_configs: Optional[List[DomainRandomizationConfig]] = None,
        episodes_per_stage: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            env: Base environment
            curriculum_configs: List of configs for each stage (or auto-generated)
            episodes_per_stage: Episodes before advancing
            seed: Random seed
        """
        configs = curriculum_configs or create_curriculum_configs()

        # Start with first stage
        super().__init__(env, configs[0], seed)

        self.curriculum_configs = configs
        self.episodes_per_stage = episodes_per_stage
        self._episode_count = 0
        self._current_stage = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with curriculum tracking."""
        self._episode_count += 1

        # Check for curriculum advancement
        if self._episode_count >= self.episodes_per_stage:
            self._advance_stage()
            self._episode_count = 0

        obs, info = super().reset(**kwargs)
        info['curriculum_stage'] = self._current_stage
        return obs, info

    def _advance_stage(self):
        """Advance to next curriculum stage."""
        if self._current_stage < len(self.curriculum_configs) - 1:
            self._current_stage += 1
            self.config = self.curriculum_configs[self._current_stage]
            self.randomizer = DomainRandomizer(self.config)
            print(f"Advanced to curriculum stage {self._current_stage}")

    def set_stage(self, stage: int):
        """Manually set curriculum stage."""
        stage = max(0, min(stage, len(self.curriculum_configs) - 1))
        self._current_stage = stage
        self.config = self.curriculum_configs[stage]
        self.randomizer = DomainRandomizer(self.config)


def make_randomized_env(
    env: gym.Env,
    config: Optional[DomainRandomizationConfig] = None,
    curriculum: bool = False,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Convenience function to wrap environment with domain randomization.

    Args:
        env: Base environment
        config: Randomization config (uses defaults if None)
        curriculum: Whether to use curriculum learning
        seed: Random seed

    Returns:
        Wrapped environment
    """
    if curriculum:
        return CurriculumRandomizationWrapper(env, seed=seed)
    else:
        return DomainRandomizationWrapper(env, config, seed)

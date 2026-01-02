"""
Domain Randomization Configuration

Main configuration and coordination for sim2real transfer via
physics, sensor, and actuator randomization.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum


class DistributionType(Enum):
    """Distribution types for randomization."""
    UNIFORM = "uniform"          # Uniform between min and max
    NORMAL = "normal"            # Gaussian with mean at center
    LOG_UNIFORM = "log_uniform"  # Log-uniform for scale parameters
    CONSTANT = "constant"        # No randomization


@dataclass
class RandomizationRange:
    """
    Range specification for a randomizable parameter.

    Supports multiple distribution types for different parameter
    characteristics.
    """
    min_value: float
    max_value: float
    distribution: DistributionType = DistributionType.UNIFORM
    mean_value: Optional[float] = None  # For normal distribution
    std_value: Optional[float] = None   # For normal distribution

    def __post_init__(self):
        if self.distribution == DistributionType.NORMAL:
            if self.mean_value is None:
                self.mean_value = (self.min_value + self.max_value) / 2
            if self.std_value is None:
                self.std_value = (self.max_value - self.min_value) / 6  # 3-sigma

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample a value from this range.

        Args:
            rng: Optional random generator for reproducibility

        Returns:
            Sampled value
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == DistributionType.CONSTANT:
            return self.mean_value if self.mean_value is not None else self.min_value

        if self.distribution == DistributionType.UNIFORM:
            return rng.uniform(self.min_value, self.max_value)

        if self.distribution == DistributionType.NORMAL:
            value = rng.normal(self.mean_value, self.std_value)
            return np.clip(value, self.min_value, self.max_value)

        if self.distribution == DistributionType.LOG_UNIFORM:
            log_min = np.log(max(self.min_value, 1e-10))
            log_max = np.log(max(self.max_value, 1e-10))
            return np.exp(rng.uniform(log_min, log_max))

        return self.min_value

    def scale(self, factor: float) -> 'RandomizationRange':
        """
        Create scaled version of this range.

        Useful for curriculum learning where we gradually increase
        randomization difficulty.
        """
        center = (self.min_value + self.max_value) / 2
        half_range = (self.max_value - self.min_value) / 2 * factor

        return RandomizationRange(
            min_value=center - half_range,
            max_value=center + half_range,
            distribution=self.distribution,
            mean_value=self.mean_value,
            std_value=self.std_value * factor if self.std_value else None,
        )


@dataclass
class PhysicsRandomizationConfig:
    """Configuration for physics randomization."""
    # Mass randomization (fraction of nominal)
    mass_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )

    # Inertia randomization (fraction of nominal)
    inertia_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )

    # Center of gravity offset (meters)
    cg_offset_x: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.02, 0.02)
    )
    cg_offset_y: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.02, 0.02)
    )
    cg_offset_z: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.02, 0.02)
    )

    # Payload mass (kg)
    payload_mass: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 0.5)
    )

    # Drag coefficient scale
    drag_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )


@dataclass
class ActuatorRandomizationConfig:
    """Configuration for actuator randomization."""
    # Motor time constant (seconds)
    motor_time_constant: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.01, 0.1)
    )

    # Motor gain scale (fraction of nominal)
    motor_gain_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.9, 1.1)
    )

    # Saturation level (fraction of max)
    saturation_level: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.85, 1.0)
    )

    # Battery voltage sag (fraction reduction)
    battery_sag_factor: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 0.15)
    )

    # Command delay (seconds)
    command_delay: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 0.05)
    )

    # Motor bias (per-motor offset)
    motor_bias: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.05, 0.05)
    )


@dataclass
class SensorRandomizationConfig:
    """Configuration for sensor randomization."""
    # IMU gyroscope bias (rad/s)
    imu_gyro_bias: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.05, 0.05)
    )

    # IMU accelerometer bias (m/s^2)
    imu_accel_bias: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.5, 0.5)
    )

    # IMU noise scale (multiplier on nominal noise)
    imu_noise_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.5, 2.0)
    )

    # GPS noise scale (multiplier on nominal noise)
    gps_noise_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.5, 3.0)
    )

    # GPS dropout probability (per-timestep)
    gps_dropout_prob: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 0.1)
    )

    # Magnetometer bias (normalized)
    mag_bias: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.1, 0.1)
    )

    # Observation delay (seconds)
    observation_delay: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 0.03)
    )


@dataclass
class EnvironmentRandomizationConfig:
    """Configuration for environment randomization."""
    # Wind speed (m/s)
    wind_speed: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 10.0)
    )

    # Gust intensity (m/s)
    gust_intensity: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 5.0)
    )

    # Turbulence scale (multiplier)
    turbulence_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0, 2.0)
    )

    # Air density variation (fraction)
    air_density_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.95, 1.05)
    )

    # Ground effect scale
    ground_effect_scale: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )


@dataclass
class DomainRandomizationConfig:
    """
    Complete configuration for domain randomization.

    Combines physics, actuator, sensor, and environment randomization.
    """
    physics: PhysicsRandomizationConfig = field(
        default_factory=PhysicsRandomizationConfig
    )
    actuator: ActuatorRandomizationConfig = field(
        default_factory=ActuatorRandomizationConfig
    )
    sensor: SensorRandomizationConfig = field(
        default_factory=SensorRandomizationConfig
    )
    environment: EnvironmentRandomizationConfig = field(
        default_factory=EnvironmentRandomizationConfig
    )

    # Global settings
    enabled: bool = True
    random_seed: Optional[int] = None

    # Curriculum settings
    curriculum_enabled: bool = False
    curriculum_stage: int = 0  # 0-3 for increasing difficulty

    def scale_all(self, factor: float) -> 'DomainRandomizationConfig':
        """
        Create scaled version of all randomization ranges.

        Args:
            factor: Scale factor (0 = no randomization, 1 = full)

        Returns:
            Scaled configuration
        """
        # This is a simplified implementation - in practice you'd
        # recursively scale all RandomizationRange fields
        return DomainRandomizationConfig(
            physics=self.physics,
            actuator=self.actuator,
            sensor=self.sensor,
            environment=self.environment,
            enabled=self.enabled,
            random_seed=self.random_seed,
            curriculum_enabled=self.curriculum_enabled,
            curriculum_stage=self.curriculum_stage,
        )


@dataclass
class SampledDomainParams:
    """
    Sampled domain parameters for a single episode.

    Contains the actual values sampled from the configuration ranges.
    """
    # Physics
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    cg_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    payload_mass: float = 0.0
    drag_scale: float = 1.0

    # Actuator
    motor_time_constant: float = 0.02
    motor_gain_scale: float = 1.0
    saturation_level: float = 1.0
    battery_sag_factor: float = 0.0
    command_delay: float = 0.0
    motor_biases: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Sensor
    imu_gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_noise_scale: float = 1.0
    gps_noise_scale: float = 1.0
    gps_dropout_prob: float = 0.0
    mag_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    observation_delay: float = 0.0

    # Environment
    wind_speed: float = 0.0
    wind_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    gust_intensity: float = 0.0
    turbulence_scale: float = 0.0
    air_density_scale: float = 1.0
    ground_effect_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'physics': {
                'mass_scale': self.mass_scale,
                'inertia_scale': self.inertia_scale,
                'cg_offset': self.cg_offset.tolist(),
                'payload_mass': self.payload_mass,
                'drag_scale': self.drag_scale,
            },
            'actuator': {
                'motor_time_constant': self.motor_time_constant,
                'motor_gain_scale': self.motor_gain_scale,
                'saturation_level': self.saturation_level,
                'battery_sag_factor': self.battery_sag_factor,
                'command_delay': self.command_delay,
                'motor_biases': self.motor_biases.tolist(),
            },
            'sensor': {
                'imu_gyro_bias': self.imu_gyro_bias.tolist(),
                'imu_accel_bias': self.imu_accel_bias.tolist(),
                'imu_noise_scale': self.imu_noise_scale,
                'gps_noise_scale': self.gps_noise_scale,
                'gps_dropout_prob': self.gps_dropout_prob,
                'mag_bias': self.mag_bias.tolist(),
                'observation_delay': self.observation_delay,
            },
            'environment': {
                'wind_speed': self.wind_speed,
                'wind_direction': self.wind_direction.tolist(),
                'gust_intensity': self.gust_intensity,
                'turbulence_scale': self.turbulence_scale,
                'air_density_scale': self.air_density_scale,
                'ground_effect_scale': self.ground_effect_scale,
            },
        }


class DomainRandomizer:
    """
    Main domain randomizer that samples parameters and applies them.

    Coordinates physics, actuator, and sensor randomization.
    """

    def __init__(
        self,
        config: Optional[DomainRandomizationConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            config: Randomization configuration
            seed: Random seed for reproducibility
        """
        self.config = config or DomainRandomizationConfig()
        self.rng = np.random.default_rng(seed or self.config.random_seed)
        self._current_params: Optional[SampledDomainParams] = None

    @property
    def current_params(self) -> Optional[SampledDomainParams]:
        """Get current sampled parameters."""
        return self._current_params

    def sample(self) -> SampledDomainParams:
        """
        Sample new domain parameters.

        Returns:
            SampledDomainParams with all sampled values
        """
        if not self.config.enabled:
            return SampledDomainParams()

        # Apply curriculum scaling if enabled
        if self.config.curriculum_enabled:
            scale = self._get_curriculum_scale()
        else:
            scale = 1.0

        cfg = self.config

        # Sample physics
        mass_scale = self._sample_scaled(cfg.physics.mass_scale, scale)
        inertia_scale = self._sample_scaled(cfg.physics.inertia_scale, scale)
        cg_offset = np.array([
            self._sample_scaled(cfg.physics.cg_offset_x, scale),
            self._sample_scaled(cfg.physics.cg_offset_y, scale),
            self._sample_scaled(cfg.physics.cg_offset_z, scale),
        ])
        payload_mass = self._sample_scaled(cfg.physics.payload_mass, scale)
        drag_scale = self._sample_scaled(cfg.physics.drag_scale, scale)

        # Sample actuator
        motor_tc = self._sample_scaled(cfg.actuator.motor_time_constant, scale)
        motor_gain = self._sample_scaled(cfg.actuator.motor_gain_scale, scale)
        saturation = self._sample_scaled(cfg.actuator.saturation_level, scale)
        battery_sag = self._sample_scaled(cfg.actuator.battery_sag_factor, scale)
        cmd_delay = self._sample_scaled(cfg.actuator.command_delay, scale)
        motor_biases = np.array([
            self._sample_scaled(cfg.actuator.motor_bias, scale)
            for _ in range(4)
        ])

        # Sample sensor
        gyro_bias = np.array([
            self._sample_scaled(cfg.sensor.imu_gyro_bias, scale)
            for _ in range(3)
        ])
        accel_bias = np.array([
            self._sample_scaled(cfg.sensor.imu_accel_bias, scale)
            for _ in range(3)
        ])
        imu_noise = self._sample_scaled(cfg.sensor.imu_noise_scale, scale)
        gps_noise = self._sample_scaled(cfg.sensor.gps_noise_scale, scale)
        gps_dropout = self._sample_scaled(cfg.sensor.gps_dropout_prob, scale)
        mag_bias = np.array([
            self._sample_scaled(cfg.sensor.mag_bias, scale)
            for _ in range(3)
        ])
        obs_delay = self._sample_scaled(cfg.sensor.observation_delay, scale)

        # Sample environment
        wind_speed = self._sample_scaled(cfg.environment.wind_speed, scale)
        wind_angle = self.rng.uniform(0, 2 * np.pi)
        wind_direction = np.array([np.cos(wind_angle), np.sin(wind_angle), 0.0])
        gust = self._sample_scaled(cfg.environment.gust_intensity, scale)
        turbulence = self._sample_scaled(cfg.environment.turbulence_scale, scale)
        air_density = self._sample_scaled(cfg.environment.air_density_scale, scale)
        ground_effect = self._sample_scaled(cfg.environment.ground_effect_scale, scale)

        self._current_params = SampledDomainParams(
            mass_scale=mass_scale,
            inertia_scale=inertia_scale,
            cg_offset=cg_offset,
            payload_mass=payload_mass,
            drag_scale=drag_scale,
            motor_time_constant=motor_tc,
            motor_gain_scale=motor_gain,
            saturation_level=saturation,
            battery_sag_factor=battery_sag,
            command_delay=cmd_delay,
            motor_biases=motor_biases,
            imu_gyro_bias=gyro_bias,
            imu_accel_bias=accel_bias,
            imu_noise_scale=imu_noise,
            gps_noise_scale=gps_noise,
            gps_dropout_prob=gps_dropout,
            mag_bias=mag_bias,
            observation_delay=obs_delay,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            gust_intensity=gust,
            turbulence_scale=turbulence,
            air_density_scale=air_density,
            ground_effect_scale=ground_effect,
        )

        return self._current_params

    def _sample_scaled(
        self,
        range_spec: RandomizationRange,
        scale: float,
    ) -> float:
        """Sample from range with curriculum scaling."""
        if scale >= 1.0:
            return range_spec.sample(self.rng)

        scaled = range_spec.scale(scale)
        return scaled.sample(self.rng)

    def _get_curriculum_scale(self) -> float:
        """Get curriculum scale based on current stage."""
        # Stage 0: 25%, Stage 1: 50%, Stage 2: 75%, Stage 3: 100%
        scales = [0.25, 0.5, 0.75, 1.0]
        stage = min(self.config.curriculum_stage, len(scales) - 1)
        return scales[stage]

    def set_curriculum_stage(self, stage: int):
        """Set curriculum stage (0-3)."""
        self.config.curriculum_stage = max(0, min(stage, 3))

    def advance_curriculum(self):
        """Advance to next curriculum stage."""
        if self.config.curriculum_stage < 3:
            self.config.curriculum_stage += 1


def create_curriculum_configs() -> List[DomainRandomizationConfig]:
    """
    Create curriculum of domain randomization configs.

    Returns 4 configs with increasing difficulty:
    - Stage 0 (Easy): Minimal randomization
    - Stage 1 (Medium): Moderate randomization
    - Stage 2 (Hard): Full physics/actuator, partial sensor
    - Stage 3 (Extreme): Full randomization

    Returns:
        List of 4 DomainRandomizationConfig
    """
    configs = []

    # Stage 0: Easy
    easy = DomainRandomizationConfig(
        physics=PhysicsRandomizationConfig(
            mass_scale=RandomizationRange(0.95, 1.05),
            inertia_scale=RandomizationRange(0.95, 1.05),
            cg_offset_x=RandomizationRange(-0.005, 0.005),
            cg_offset_y=RandomizationRange(-0.005, 0.005),
            cg_offset_z=RandomizationRange(-0.005, 0.005),
        ),
        actuator=ActuatorRandomizationConfig(
            motor_time_constant=RandomizationRange(0.02, 0.03),
            motor_gain_scale=RandomizationRange(0.98, 1.02),
        ),
        sensor=SensorRandomizationConfig(
            imu_gyro_bias=RandomizationRange(-0.01, 0.01),
            imu_accel_bias=RandomizationRange(-0.1, 0.1),
            imu_noise_scale=RandomizationRange(0.9, 1.1),
            gps_dropout_prob=RandomizationRange(0.0, 0.0),
        ),
        environment=EnvironmentRandomizationConfig(
            wind_speed=RandomizationRange(0.0, 2.0),
            gust_intensity=RandomizationRange(0.0, 1.0),
        ),
        curriculum_stage=0,
    )
    configs.append(easy)

    # Stage 1: Medium
    medium = DomainRandomizationConfig(
        physics=PhysicsRandomizationConfig(
            mass_scale=RandomizationRange(0.9, 1.1),
            inertia_scale=RandomizationRange(0.9, 1.1),
            cg_offset_x=RandomizationRange(-0.01, 0.01),
            cg_offset_y=RandomizationRange(-0.01, 0.01),
            cg_offset_z=RandomizationRange(-0.01, 0.01),
            payload_mass=RandomizationRange(0.0, 0.2),
        ),
        actuator=ActuatorRandomizationConfig(
            motor_time_constant=RandomizationRange(0.015, 0.05),
            motor_gain_scale=RandomizationRange(0.95, 1.05),
            battery_sag_factor=RandomizationRange(0.0, 0.05),
        ),
        sensor=SensorRandomizationConfig(
            imu_gyro_bias=RandomizationRange(-0.02, 0.02),
            imu_accel_bias=RandomizationRange(-0.25, 0.25),
            imu_noise_scale=RandomizationRange(0.7, 1.3),
            gps_noise_scale=RandomizationRange(0.8, 1.5),
            gps_dropout_prob=RandomizationRange(0.0, 0.02),
        ),
        environment=EnvironmentRandomizationConfig(
            wind_speed=RandomizationRange(0.0, 5.0),
            gust_intensity=RandomizationRange(0.0, 2.0),
            turbulence_scale=RandomizationRange(0.0, 0.5),
        ),
        curriculum_stage=1,
    )
    configs.append(medium)

    # Stage 2: Hard
    hard = DomainRandomizationConfig(
        physics=PhysicsRandomizationConfig(
            mass_scale=RandomizationRange(0.85, 1.15),
            inertia_scale=RandomizationRange(0.85, 1.15),
            cg_offset_x=RandomizationRange(-0.015, 0.015),
            cg_offset_y=RandomizationRange(-0.015, 0.015),
            cg_offset_z=RandomizationRange(-0.015, 0.015),
            payload_mass=RandomizationRange(0.0, 0.35),
        ),
        actuator=ActuatorRandomizationConfig(
            motor_time_constant=RandomizationRange(0.01, 0.07),
            motor_gain_scale=RandomizationRange(0.92, 1.08),
            saturation_level=RandomizationRange(0.9, 1.0),
            battery_sag_factor=RandomizationRange(0.0, 0.1),
            command_delay=RandomizationRange(0.0, 0.03),
        ),
        sensor=SensorRandomizationConfig(
            imu_gyro_bias=RandomizationRange(-0.035, 0.035),
            imu_accel_bias=RandomizationRange(-0.35, 0.35),
            imu_noise_scale=RandomizationRange(0.6, 1.6),
            gps_noise_scale=RandomizationRange(0.6, 2.0),
            gps_dropout_prob=RandomizationRange(0.0, 0.05),
        ),
        environment=EnvironmentRandomizationConfig(
            wind_speed=RandomizationRange(0.0, 8.0),
            gust_intensity=RandomizationRange(0.0, 3.5),
            turbulence_scale=RandomizationRange(0.0, 1.0),
        ),
        curriculum_stage=2,
    )
    configs.append(hard)

    # Stage 3: Extreme (full randomization)
    extreme = DomainRandomizationConfig(
        curriculum_stage=3,
    )  # Uses all defaults which are the extreme ranges
    configs.append(extreme)

    return configs

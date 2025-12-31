"""
Environmental Conditions System

Simulates weather, wind, lighting, and interference that affects drone flight.
Provides realistic environmental disturbances for robust policy training.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np


class WeatherType(Enum):
    """Weather condition types."""
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"
    DUST = "dust"
    HAIL = "hail"


class TimeOfDay(Enum):
    """Time of day affecting lighting and visibility."""
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


class TerrainType(Enum):
    """Terrain type affecting GPS and wind patterns."""
    OPEN_FIELD = "open_field"
    URBAN = "urban"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    COASTAL = "coastal"
    DESERT = "desert"
    INDOOR = "indoor"


@dataclass
class WindModel:
    """
    Wind model with base wind, gusts, and turbulence.

    Attributes:
        base_speed: Steady-state wind speed (m/s)
        base_direction: Wind direction in radians (0 = North, pi/2 = East)
        gust_intensity: Maximum gust speed addition (m/s)
        gust_probability: Probability of gust per timestep
        gust_duration_range: Min/max gust duration in seconds
        turbulence_intensity: Turbulence noise scale
    """
    base_speed: float = 0.0
    base_direction: float = 0.0
    gust_intensity: float = 0.0
    gust_probability: float = 0.0
    gust_duration_range: Tuple[float, float] = (0.5, 2.0)
    turbulence_intensity: float = 0.0

    # Internal state for gust tracking
    _current_gust: float = field(default=0.0, repr=False)
    _gust_remaining: float = field(default=0.0, repr=False)

    def get_wind_vector(self, dt: float = 0.01) -> np.ndarray:
        """
        Get current wind force vector with gusts and turbulence.

        Args:
            dt: Timestep for gust decay

        Returns:
            3D wind force vector [x, y, z]
        """
        # Update gust state
        if self._gust_remaining > 0:
            self._gust_remaining -= dt
        else:
            # Check for new gust
            if np.random.random() < self.gust_probability * dt:
                self._current_gust = np.random.uniform(0, self.gust_intensity)
                self._gust_remaining = np.random.uniform(*self.gust_duration_range)
            else:
                self._current_gust = 0.0

        # Total wind speed
        total_speed = self.base_speed + self._current_gust

        # Add turbulence
        turbulence = np.random.normal(0, self.turbulence_intensity, 3)

        # Convert to vector
        wind_vector = np.array([
            total_speed * np.cos(self.base_direction) + turbulence[0],
            total_speed * np.sin(self.base_direction) + turbulence[1],
            turbulence[2]  # Vertical turbulence only
        ])

        return wind_vector

    def reset(self):
        """Reset gust state."""
        self._current_gust = 0.0
        self._gust_remaining = 0.0


@dataclass
class EnvironmentalConditions:
    """
    Environmental parameters affecting drone flight.

    This class encapsulates all environmental factors that influence
    drone behavior, sensor quality, and mission success.
    """
    # Weather and visibility
    weather: WeatherType = WeatherType.CLEAR
    time_of_day: TimeOfDay = TimeOfDay.DAY
    terrain: TerrainType = TerrainType.OPEN_FIELD

    # Atmospheric conditions
    temperature: float = 20.0  # Celsius
    pressure: float = 101325.0  # Pascals (sea level)
    humidity: float = 50.0  # Percentage
    air_density: float = 1.225  # kg/m³

    # Visibility
    visibility: float = 10000.0  # meters
    cloud_ceiling: float = 3000.0  # meters

    # Wind model
    wind: WindModel = field(default_factory=WindModel)

    # Interference and noise
    rf_interference: float = 0.0  # 0-1 scale
    magnetic_interference: float = 0.0  # 0-1 scale
    gps_degradation: float = 0.0  # 0-1 scale (0 = perfect, 1 = denied)

    # Lighting (affects camera sensors)
    ambient_light: float = 1.0  # 0-1 scale
    sun_angle: float = 45.0  # degrees from horizon

    def __post_init__(self):
        """Apply weather presets after initialization."""
        self._apply_weather_effects()
        self._apply_time_effects()
        self._apply_terrain_effects()

    def _apply_weather_effects(self):
        """Modify parameters based on weather type."""
        if self.weather == WeatherType.CLEAR:
            pass  # Default values

        elif self.weather == WeatherType.RAIN:
            self.visibility = min(self.visibility, 5000.0)
            self.humidity = max(self.humidity, 80.0)
            if self.wind.turbulence_intensity < 0.5:
                self.wind.turbulence_intensity = 0.5

        elif self.weather == WeatherType.SNOW:
            self.visibility = min(self.visibility, 2000.0)
            self.temperature = min(self.temperature, 0.0)
            self.air_density = 1.3  # Cold air is denser

        elif self.weather == WeatherType.FOG:
            self.visibility = min(self.visibility, 500.0)
            self.humidity = 100.0

        elif self.weather == WeatherType.STORM:
            self.visibility = min(self.visibility, 1000.0)
            self.wind.base_speed = max(self.wind.base_speed, 15.0)
            self.wind.gust_intensity = max(self.wind.gust_intensity, 10.0)
            self.wind.gust_probability = 0.3
            self.wind.turbulence_intensity = 2.0
            self.rf_interference = max(self.rf_interference, 0.3)

        elif self.weather == WeatherType.DUST:
            self.visibility = min(self.visibility, 1000.0)
            self.wind.base_speed = max(self.wind.base_speed, 8.0)

    def _apply_time_effects(self):
        """Modify parameters based on time of day."""
        if self.time_of_day == TimeOfDay.DAWN:
            self.ambient_light = 0.4
            self.sun_angle = 10.0

        elif self.time_of_day == TimeOfDay.DAY:
            self.ambient_light = 1.0
            self.sun_angle = 45.0

        elif self.time_of_day == TimeOfDay.DUSK:
            self.ambient_light = 0.3
            self.sun_angle = 5.0

        elif self.time_of_day == TimeOfDay.NIGHT:
            self.ambient_light = 0.05
            self.sun_angle = -30.0

    def _apply_terrain_effects(self):
        """Modify parameters based on terrain type."""
        if self.terrain == TerrainType.URBAN:
            self.rf_interference = max(self.rf_interference, 0.4)
            self.magnetic_interference = max(self.magnetic_interference, 0.3)
            self.gps_degradation = max(self.gps_degradation, 0.2)
            self.wind.turbulence_intensity += 0.5  # Building-induced turbulence

        elif self.terrain == TerrainType.INDOOR:
            self.gps_degradation = 1.0  # GPS denied
            self.wind.base_speed = 0.0
            self.wind.turbulence_intensity = 0.1  # Light HVAC currents
            self.visibility = min(self.visibility, 100.0)
            self.rf_interference = max(self.rf_interference, 0.5)

        elif self.terrain == TerrainType.MOUNTAIN:
            self.wind.base_speed *= 1.5
            self.wind.turbulence_intensity += 1.0
            self.pressure *= 0.85  # Higher altitude
            self.air_density *= 0.85

        elif self.terrain == TerrainType.COASTAL:
            self.wind.base_speed = max(self.wind.base_speed, 5.0)
            self.humidity = max(self.humidity, 70.0)

        elif self.terrain == TerrainType.FOREST:
            self.gps_degradation = max(self.gps_degradation, 0.3)
            self.visibility = min(self.visibility, 200.0)

    def get_wind_vector(self, dt: float = 0.01) -> np.ndarray:
        """Get current wind force vector."""
        return self.wind.get_wind_vector(dt)

    def get_sensor_noise_scale(self, sensor_type: str) -> float:
        """
        Get noise scale factor for a sensor based on current conditions.

        Args:
            sensor_type: One of 'camera', 'gps', 'imu', 'barometer', 'lidar', 'magnetometer'

        Returns:
            Noise multiplier (1.0 = nominal, higher = more noise)
        """
        base_noise = 1.0

        # Weather effects
        if self.weather == WeatherType.RAIN:
            if sensor_type == "camera":
                base_noise *= 2.0
            elif sensor_type == "gps":
                base_noise *= 1.5
            elif sensor_type == "lidar":
                base_noise *= 1.3

        elif self.weather == WeatherType.FOG:
            if sensor_type == "camera":
                base_noise *= 3.0
            elif sensor_type == "lidar":
                base_noise *= 2.0

        elif self.weather == WeatherType.SNOW:
            if sensor_type == "camera":
                base_noise *= 2.5
            elif sensor_type == "gps":
                base_noise *= 1.3

        elif self.weather == WeatherType.STORM:
            if sensor_type == "gps":
                base_noise *= 2.0
            elif sensor_type == "magnetometer":
                base_noise *= 1.5

        # Time of day effects
        if self.time_of_day == TimeOfDay.NIGHT:
            if sensor_type == "camera":
                base_noise *= 3.0
        elif self.time_of_day in (TimeOfDay.DAWN, TimeOfDay.DUSK):
            if sensor_type == "camera":
                base_noise *= 1.5

        # RF interference effects
        if sensor_type == "gps":
            base_noise *= (1.0 + self.rf_interference * 2.0)
            base_noise *= (1.0 + self.gps_degradation * 5.0)

        # Magnetic interference effects
        if sensor_type == "magnetometer":
            base_noise *= (1.0 + self.magnetic_interference * 3.0)

        # Temperature effects on barometer
        if sensor_type == "barometer":
            temp_deviation = abs(self.temperature - 20.0)
            base_noise *= (1.0 + temp_deviation * 0.01)

        return base_noise

    def get_drag_coefficient_modifier(self) -> float:
        """
        Get drag coefficient modifier based on weather.

        Returns:
            Multiplier for base drag coefficient
        """
        modifier = 1.0

        if self.weather == WeatherType.RAIN:
            modifier *= 1.1
        elif self.weather == WeatherType.SNOW:
            modifier *= 1.15
        elif self.weather == WeatherType.DUST:
            modifier *= 1.05

        # Air density affects drag
        modifier *= (self.air_density / 1.225)

        return modifier

    def get_motor_efficiency_modifier(self) -> float:
        """
        Get motor efficiency modifier based on conditions.

        Returns:
            Multiplier for motor efficiency (< 1.0 means reduced efficiency)
        """
        efficiency = 1.0

        # Temperature effects (optimal around 20-25°C)
        if self.temperature < 0:
            efficiency *= 0.9
        elif self.temperature > 40:
            efficiency *= 0.85

        # Altitude effects (lower air density = less lift per RPM)
        efficiency *= (self.air_density / 1.225)

        return efficiency

    def is_flight_safe(self) -> Tuple[bool, str]:
        """
        Check if conditions are safe for flight.

        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        if self.weather == WeatherType.STORM:
            return False, "Storm conditions - flight not recommended"

        if self.wind.base_speed > 20.0:
            return False, f"Wind speed {self.wind.base_speed:.1f} m/s exceeds safe limit"

        if self.visibility < 100.0:
            return False, f"Visibility {self.visibility:.0f}m below minimum"

        return True, ""

    def reset(self):
        """Reset dynamic state (e.g., wind gusts)."""
        self.wind.reset()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "weather": self.weather.value,
            "time_of_day": self.time_of_day.value,
            "terrain": self.terrain.value,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "humidity": self.humidity,
            "air_density": self.air_density,
            "visibility": self.visibility,
            "wind_speed": self.wind.base_speed,
            "wind_direction": self.wind.base_direction,
            "rf_interference": self.rf_interference,
            "gps_degradation": self.gps_degradation,
            "ambient_light": self.ambient_light,
        }


# =============================================================================
# PRESET CONDITIONS
# =============================================================================

def create_clear_day() -> EnvironmentalConditions:
    """Create ideal clear day conditions."""
    return EnvironmentalConditions(
        weather=WeatherType.CLEAR,
        time_of_day=TimeOfDay.DAY,
        terrain=TerrainType.OPEN_FIELD,
        wind=WindModel(base_speed=2.0, turbulence_intensity=0.1)
    )


def create_windy_conditions(wind_speed: float = 10.0) -> EnvironmentalConditions:
    """Create windy but otherwise clear conditions."""
    return EnvironmentalConditions(
        weather=WeatherType.CLEAR,
        time_of_day=TimeOfDay.DAY,
        wind=WindModel(
            base_speed=wind_speed,
            gust_intensity=wind_speed * 0.3,
            gust_probability=0.1,
            turbulence_intensity=wind_speed * 0.1
        )
    )


def create_night_conditions() -> EnvironmentalConditions:
    """Create night flight conditions."""
    return EnvironmentalConditions(
        weather=WeatherType.CLEAR,
        time_of_day=TimeOfDay.NIGHT,
        wind=WindModel(base_speed=3.0, turbulence_intensity=0.2)
    )


def create_urban_conditions() -> EnvironmentalConditions:
    """Create urban environment conditions."""
    return EnvironmentalConditions(
        weather=WeatherType.CLEAR,
        time_of_day=TimeOfDay.DAY,
        terrain=TerrainType.URBAN,
        wind=WindModel(base_speed=3.0, turbulence_intensity=1.0)
    )


def create_indoor_conditions() -> EnvironmentalConditions:
    """Create indoor/GPS-denied conditions."""
    return EnvironmentalConditions(
        weather=WeatherType.CLEAR,
        time_of_day=TimeOfDay.DAY,
        terrain=TerrainType.INDOOR,
    )


def create_adverse_conditions() -> EnvironmentalConditions:
    """Create challenging adverse weather conditions."""
    return EnvironmentalConditions(
        weather=WeatherType.RAIN,
        time_of_day=TimeOfDay.DUSK,
        terrain=TerrainType.COASTAL,
        wind=WindModel(
            base_speed=12.0,
            gust_intensity=5.0,
            gust_probability=0.2,
            turbulence_intensity=1.5
        )
    )


def create_random_conditions(
    difficulty: str = "medium",
    seed: Optional[int] = None
) -> EnvironmentalConditions:
    """
    Create randomized environmental conditions.

    Args:
        difficulty: 'easy', 'medium', 'hard', or 'extreme'
        seed: Random seed for reproducibility

    Returns:
        Randomized EnvironmentalConditions
    """
    if seed is not None:
        np.random.seed(seed)

    # Difficulty-based parameter ranges
    ranges = {
        "easy": {
            "weather": [WeatherType.CLEAR],
            "time": [TimeOfDay.DAY],
            "wind_speed": (0, 3),
            "turbulence": (0, 0.3),
        },
        "medium": {
            "weather": [WeatherType.CLEAR, WeatherType.RAIN],
            "time": [TimeOfDay.DAY, TimeOfDay.DAWN, TimeOfDay.DUSK],
            "wind_speed": (2, 8),
            "turbulence": (0.2, 0.8),
        },
        "hard": {
            "weather": [WeatherType.CLEAR, WeatherType.RAIN, WeatherType.FOG],
            "time": list(TimeOfDay),
            "wind_speed": (5, 15),
            "turbulence": (0.5, 1.5),
        },
        "extreme": {
            "weather": list(WeatherType),
            "time": list(TimeOfDay),
            "wind_speed": (10, 20),
            "turbulence": (1.0, 3.0),
        },
    }

    params = ranges.get(difficulty, ranges["medium"])

    weather = np.random.choice(params["weather"])
    time = np.random.choice(params["time"])
    wind_speed = np.random.uniform(*params["wind_speed"])
    turbulence = np.random.uniform(*params["turbulence"])

    return EnvironmentalConditions(
        weather=weather,
        time_of_day=time,
        wind=WindModel(
            base_speed=wind_speed,
            base_direction=np.random.uniform(0, 2 * np.pi),
            gust_intensity=wind_speed * 0.3,
            gust_probability=0.1,
            turbulence_intensity=turbulence
        )
    )

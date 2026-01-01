"""
Military AirSim Environment Configurations.

Defines environment presets for visualizing military UAV operations in AirSim:
- Aircraft carrier for X-47B operations
- Military airbase with runway operations
- Contested airspace with threat systems
- Urban strike zone for CAS missions

These environments are designed for policy visualization and testing,
not training (training uses PyBullet for speed).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json


class MilitaryEnvironmentType(Enum):
    """Military environment types for AirSim."""
    AIRCRAFT_CARRIER = "AircraftCarrier"
    MILITARY_AIRBASE = "MilitaryAirbase"
    CONTESTED_AIRSPACE = "ContestedAirspace"
    URBAN_STRIKE_ZONE = "UrbanStrikeZone"
    MARITIME_PATROL = "MaritimePatrol"
    DESERT_STRIKE = "DesertStrike"
    MOUNTAIN_TERRAIN = "MountainTerrain"


class WeatherCondition(Enum):
    """Weather conditions for environment."""
    CLEAR = "clear"
    OVERCAST = "overcast"
    RAIN = "rain"
    FOG = "fog"
    STORM = "storm"


class TimeOfDay(Enum):
    """Time of day settings."""
    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"


@dataclass
class CarrierConfig:
    """Aircraft carrier configuration."""
    ship_type: str = "Nimitz"           # Nimitz, Ford, Queen_Elizabeth
    heading: float = 0.0                # Ship heading [degrees]
    speed: float = 30.0                 # Ship speed [knots]
    deck_position: Tuple[float, float, float] = (0.0, 0.0, 20.0)

    # Deck layout
    angled_deck_angle: float = 9.0      # Angled deck offset [degrees]
    catapult_count: int = 4             # Number of catapults
    wire_count: int = 4                 # Arresting wires

    # Motion (sea state dependent)
    sea_state: int = 3                  # 1-6 sea state
    pitch_amplitude: float = 2.0        # Max pitch [degrees]
    roll_amplitude: float = 3.0         # Max roll [degrees]
    heave_amplitude: float = 2.0        # Max heave [meters]


@dataclass
class AirbaseConfig:
    """Military airbase configuration."""
    runway_length: float = 3000.0       # Runway length [meters]
    runway_width: float = 45.0          # Runway width [meters]
    runway_heading: float = 270.0       # Runway heading [degrees]
    elevation: float = 100.0            # Field elevation [meters]

    # Facilities
    num_hangars: int = 4
    num_parking_spots: int = 20
    has_control_tower: bool = True
    has_ils: bool = True                # Instrument landing system

    # Taxiways
    taxiway_layout: str = "standard"    # standard, complex, simple


@dataclass
class ThreatSystemConfig:
    """Air defense threat system configuration."""
    system_type: str                    # SAM_LONG, SAM_MEDIUM, SAM_SHORT, AAA
    position: Tuple[float, float, float]
    detection_range: float              # [meters]
    engagement_range: float             # [meters]
    active: bool = True

    # Visual representation
    show_detection_ring: bool = True
    show_engagement_ring: bool = True
    ring_color: Tuple[int, int, int] = (255, 100, 100)


@dataclass
class TargetConfig:
    """Ground target configuration."""
    target_type: str                    # vehicle, structure, radar, bunker
    position: Tuple[float, float, float]
    heading: float = 0.0
    mobile: bool = False

    # Visual
    model: str = "default"
    scale: float = 1.0


@dataclass
class MilitaryAirSimConfig:
    """Complete military AirSim environment configuration."""
    environment_type: MilitaryEnvironmentType
    name: str
    description: str

    # World settings
    world_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_scale: float = 1.0

    # Weather and time
    weather: WeatherCondition = WeatherCondition.CLEAR
    time_of_day: TimeOfDay = TimeOfDay.NOON
    cloud_density: float = 0.0          # 0-1
    wind_speed: float = 0.0             # [m/s]
    wind_direction: float = 0.0         # [degrees]
    visibility: float = 10000.0         # [meters]

    # Environment-specific configs
    carrier: Optional[CarrierConfig] = None
    airbase: Optional[AirbaseConfig] = None
    threats: List[ThreatSystemConfig] = field(default_factory=list)
    targets: List[TargetConfig] = field(default_factory=list)

    # Camera settings
    camera_positions: List[Dict[str, Any]] = field(default_factory=list)
    follow_camera_distance: float = 50.0
    follow_camera_height: float = 10.0

    # Physics settings (for AirSim)
    enable_collision: bool = True
    enable_traces: bool = False         # Show flight path
    trace_color: Tuple[int, int, int] = (0, 255, 0)

    def to_settings_json(self) -> Dict[str, Any]:
        """Convert to AirSim settings.json format."""
        settings = {
            "SettingsVersion": 1.2,
            "SimMode": "Multirotor",  # Or "ComputerVision" for viz only

            "ClockSpeed": 1.0,
            "ViewMode": "FlyWithMe",

            "SubWindows": [
                {"WindowID": 0, "CameraName": "0", "ImageType": 0, "Visible": True}
            ],

            "CameraDefaults": {
                "CaptureSettings": [
                    {
                        "ImageType": 0,
                        "Width": 1280,
                        "Height": 720,
                        "FOV_Degrees": 90,
                    }
                ]
            },

            "Recording": {
                "RecordOnMove": False,
                "RecordInterval": 0.05,
            },

            "EnvironmentType": self.environment_type.value,
            "WeatherEnabled": self.weather != WeatherCondition.CLEAR,
            "TimeOfDay": {
                "Enabled": True,
                "StartTime": self._time_to_start_time(),
                "TimeSpeed": 0,  # Frozen time
            },
        }

        # Add weather settings
        if self.weather != WeatherCondition.CLEAR:
            settings["Weather"] = self._get_weather_settings()

        # Add wind
        if self.wind_speed > 0:
            settings["Wind"] = {
                "X": self.wind_speed * np.cos(np.radians(self.wind_direction)),
                "Y": self.wind_speed * np.sin(np.radians(self.wind_direction)),
                "Z": 0,
            }

        return settings

    def _time_to_start_time(self) -> str:
        """Convert TimeOfDay to AirSim start time."""
        times = {
            TimeOfDay.DAWN: "2024-01-15 06:00:00",
            TimeOfDay.MORNING: "2024-01-15 09:00:00",
            TimeOfDay.NOON: "2024-01-15 12:00:00",
            TimeOfDay.AFTERNOON: "2024-01-15 15:00:00",
            TimeOfDay.DUSK: "2024-01-15 18:00:00",
            TimeOfDay.NIGHT: "2024-01-15 22:00:00",
        }
        return times.get(self.time_of_day, times[TimeOfDay.NOON])

    def _get_weather_settings(self) -> Dict[str, Any]:
        """Get AirSim weather configuration."""
        weather_configs = {
            WeatherCondition.OVERCAST: {"Fog": 0.0, "Rain": 0.0, "CloudDensity": 0.7},
            WeatherCondition.RAIN: {"Fog": 0.1, "Rain": 0.5, "CloudDensity": 0.8},
            WeatherCondition.FOG: {"Fog": 0.6, "Rain": 0.0, "CloudDensity": 0.3},
            WeatherCondition.STORM: {"Fog": 0.2, "Rain": 0.8, "CloudDensity": 0.9},
        }
        return weather_configs.get(self.weather, {"Fog": 0.0, "Rain": 0.0})


# =============================================================================
# PRE-DEFINED ENVIRONMENT CONFIGURATIONS
# =============================================================================

def _create_carrier_environment() -> MilitaryAirSimConfig:
    """Create aircraft carrier environment configuration."""
    return MilitaryAirSimConfig(
        environment_type=MilitaryEnvironmentType.AIRCRAFT_CARRIER,
        name="USS Gerald Ford Carrier Operations",
        description="Nuclear aircraft carrier environment for X-47B training",

        world_origin=(0.0, 0.0, 0.0),
        weather=WeatherCondition.CLEAR,
        time_of_day=TimeOfDay.MORNING,
        wind_speed=15.0,  # 30 knot wind over deck
        wind_direction=0.0,  # Into ship heading
        visibility=15000.0,

        carrier=CarrierConfig(
            ship_type="Ford",
            heading=0.0,
            speed=30.0,
            deck_position=(0.0, 0.0, 20.0),
            angled_deck_angle=9.0,
            catapult_count=4,
            wire_count=4,
            sea_state=3,
            pitch_amplitude=2.0,
            roll_amplitude=3.0,
            heave_amplitude=2.0,
        ),

        camera_positions=[
            {"name": "LSO_Platform", "position": (-50, 30, 25), "rotation": (0, 0, 45)},
            {"name": "Island", "position": (50, 50, 60), "rotation": (-10, 0, -135)},
            {"name": "Bow_Camera", "position": (150, 0, 30), "rotation": (0, 0, 180)},
            {"name": "Chase", "position": (-100, 0, 30), "rotation": (0, 0, 0)},
        ],

        follow_camera_distance=75.0,
        follow_camera_height=15.0,
        enable_traces=True,
        trace_color=(0, 200, 255),
    )


def _create_airbase_environment() -> MilitaryAirSimConfig:
    """Create military airbase environment configuration."""
    return MilitaryAirSimConfig(
        environment_type=MilitaryEnvironmentType.MILITARY_AIRBASE,
        name="Edwards AFB Test Range",
        description="Military airbase with long runway for testing",

        world_origin=(0.0, 0.0, 700.0),  # High desert elevation
        weather=WeatherCondition.CLEAR,
        time_of_day=TimeOfDay.NOON,
        wind_speed=5.0,
        wind_direction=270.0,
        visibility=50000.0,  # Excellent desert visibility

        airbase=AirbaseConfig(
            runway_length=4500.0,
            runway_width=60.0,
            runway_heading=270.0,
            elevation=700.0,
            num_hangars=6,
            num_parking_spots=30,
            has_control_tower=True,
            has_ils=True,
            taxiway_layout="complex",
        ),

        camera_positions=[
            {"name": "Tower", "position": (500, 200, 30), "rotation": (-5, 0, -45)},
            {"name": "Runway_Start", "position": (-2200, 0, 5), "rotation": (0, 0, 90)},
            {"name": "Runway_End", "position": (2200, 0, 5), "rotation": (0, 0, -90)},
            {"name": "Hangar", "position": (0, 300, 10), "rotation": (0, 0, -90)},
        ],

        follow_camera_distance=100.0,
        follow_camera_height=20.0,
    )


def _create_contested_airspace() -> MilitaryAirSimConfig:
    """Create contested airspace environment with threats."""
    threats = [
        # Long-range SAM (S-300 style)
        ThreatSystemConfig(
            system_type="SAM_LONG",
            position=(15000.0, 5000.0, 0.0),
            detection_range=150000.0,
            engagement_range=100000.0,
            active=True,
            show_detection_ring=True,
            ring_color=(255, 50, 50),
        ),
        # Medium-range SAM battery 1
        ThreatSystemConfig(
            system_type="SAM_MEDIUM",
            position=(8000.0, -3000.0, 0.0),
            detection_range=80000.0,
            engagement_range=40000.0,
            active=True,
            ring_color=(255, 150, 50),
        ),
        # Medium-range SAM battery 2
        ThreatSystemConfig(
            system_type="SAM_MEDIUM",
            position=(12000.0, 8000.0, 0.0),
            detection_range=80000.0,
            engagement_range=40000.0,
            active=True,
            ring_color=(255, 150, 50),
        ),
        # Short-range SAM
        ThreatSystemConfig(
            system_type="SAM_SHORT",
            position=(10000.0, 0.0, 0.0),
            detection_range=30000.0,
            engagement_range=12000.0,
            active=True,
            ring_color=(255, 200, 50),
        ),
        # AAA sites
        ThreatSystemConfig(
            system_type="AAA",
            position=(9000.0, -1000.0, 0.0),
            detection_range=8000.0,
            engagement_range=4000.0,
            active=True,
            ring_color=(200, 200, 50),
        ),
        ThreatSystemConfig(
            system_type="AAA",
            position=(11000.0, 2000.0, 0.0),
            detection_range=8000.0,
            engagement_range=4000.0,
            active=True,
            ring_color=(200, 200, 50),
        ),
    ]

    targets = [
        TargetConfig(
            target_type="radar",
            position=(15000.0, 5000.0, 0.0),
            model="radar_station",
        ),
        TargetConfig(
            target_type="command",
            position=(14000.0, 4500.0, 0.0),
            model="command_bunker",
        ),
        TargetConfig(
            target_type="vehicle",
            position=(10500.0, 500.0, 0.0),
            mobile=True,
            model="sam_launcher",
        ),
    ]

    return MilitaryAirSimConfig(
        environment_type=MilitaryEnvironmentType.CONTESTED_AIRSPACE,
        name="IADS Penetration Zone",
        description="Integrated Air Defense System environment for SEAD training",

        world_origin=(0.0, 0.0, 0.0),
        weather=WeatherCondition.OVERCAST,
        time_of_day=TimeOfDay.DUSK,
        cloud_density=0.6,
        wind_speed=8.0,
        wind_direction=45.0,
        visibility=8000.0,

        threats=threats,
        targets=targets,

        camera_positions=[
            {"name": "Overview", "position": (5000, 0, 5000), "rotation": (-45, 0, 90)},
            {"name": "Target_Area", "position": (15000, 5000, 500), "rotation": (-10, 0, 0)},
            {"name": "Ingress_Route", "position": (0, 0, 1000), "rotation": (0, 0, 90)},
        ],

        follow_camera_distance=150.0,
        follow_camera_height=30.0,
        enable_traces=True,
        trace_color=(100, 255, 100),
    )


def _create_urban_strike_zone() -> MilitaryAirSimConfig:
    """Create urban environment for CAS missions."""
    targets = [
        TargetConfig(
            target_type="vehicle",
            position=(500.0, 200.0, 0.0),
            mobile=True,
            model="technical",
        ),
        TargetConfig(
            target_type="vehicle",
            position=(600.0, -100.0, 0.0),
            mobile=True,
            model="apc",
        ),
        TargetConfig(
            target_type="structure",
            position=(800.0, 0.0, 0.0),
            model="warehouse",
        ),
    ]

    threats = [
        ThreatSystemConfig(
            system_type="MANPADS",
            position=(400.0, 100.0, 0.0),
            detection_range=5000.0,
            engagement_range=4000.0,
            active=True,
            ring_color=(255, 100, 100),
        ),
        ThreatSystemConfig(
            system_type="AAA",
            position=(700.0, -200.0, 0.0),
            detection_range=6000.0,
            engagement_range=3000.0,
            active=True,
            ring_color=(255, 150, 50),
        ),
    ]

    return MilitaryAirSimConfig(
        environment_type=MilitaryEnvironmentType.URBAN_STRIKE_ZONE,
        name="Urban CAS Environment",
        description="Urban environment for close air support training",

        world_origin=(0.0, 0.0, 50.0),
        weather=WeatherCondition.CLEAR,
        time_of_day=TimeOfDay.AFTERNOON,
        wind_speed=3.0,
        wind_direction=180.0,
        visibility=10000.0,

        threats=threats,
        targets=targets,

        camera_positions=[
            {"name": "Overhead", "position": (500, 0, 500), "rotation": (-90, 0, 0)},
            {"name": "Street_Level", "position": (0, 0, 5), "rotation": (0, 0, 90)},
            {"name": "Rooftop", "position": (300, 100, 30), "rotation": (-10, 0, 60)},
        ],

        follow_camera_distance=80.0,
        follow_camera_height=20.0,
        enable_traces=True,
        trace_color=(255, 200, 0),
    )


# =============================================================================
# ENVIRONMENT REGISTRY
# =============================================================================

MILITARY_ENVIRONMENT_CONFIGS: Dict[MilitaryEnvironmentType, MilitaryAirSimConfig] = {
    MilitaryEnvironmentType.AIRCRAFT_CARRIER: _create_carrier_environment(),
    MilitaryEnvironmentType.MILITARY_AIRBASE: _create_airbase_environment(),
    MilitaryEnvironmentType.CONTESTED_AIRSPACE: _create_contested_airspace(),
    MilitaryEnvironmentType.URBAN_STRIKE_ZONE: _create_urban_strike_zone(),
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_military_environment(
    env_type: MilitaryEnvironmentType
) -> MilitaryAirSimConfig:
    """
    Get military environment configuration.

    Args:
        env_type: Type of military environment

    Returns:
        Environment configuration
    """
    if env_type not in MILITARY_ENVIRONMENT_CONFIGS:
        available = ", ".join(e.value for e in MILITARY_ENVIRONMENT_CONFIGS.keys())
        raise ValueError(f"Unknown environment type. Available: {available}")

    return MILITARY_ENVIRONMENT_CONFIGS[env_type]


def get_carrier_environment(
    sea_state: int = 3,
    time_of_day: TimeOfDay = TimeOfDay.MORNING,
    wind_over_deck: float = 30.0,
) -> MilitaryAirSimConfig:
    """
    Get carrier environment with custom settings.

    Args:
        sea_state: Sea state 1-6
        time_of_day: Time of day setting
        wind_over_deck: Wind speed over deck [knots]

    Returns:
        Configured carrier environment
    """
    config = _create_carrier_environment()
    config.time_of_day = time_of_day
    config.wind_speed = wind_over_deck * 0.514  # knots to m/s

    if config.carrier:
        config.carrier.sea_state = sea_state
        # Adjust motion based on sea state
        config.carrier.pitch_amplitude = sea_state * 0.7
        config.carrier.roll_amplitude = sea_state * 1.0
        config.carrier.heave_amplitude = sea_state * 0.7

    return config


def get_airbase_environment(
    runway_heading: float = 270.0,
    visibility: float = 10000.0,
    weather: WeatherCondition = WeatherCondition.CLEAR,
) -> MilitaryAirSimConfig:
    """
    Get airbase environment with custom settings.

    Args:
        runway_heading: Runway heading [degrees]
        visibility: Visibility [meters]
        weather: Weather condition

    Returns:
        Configured airbase environment
    """
    config = _create_airbase_environment()
    config.weather = weather
    config.visibility = visibility

    if config.airbase:
        config.airbase.runway_heading = runway_heading
        # Adjust wind to be favorable for runway
        config.wind_direction = runway_heading

    return config


def get_contested_airspace(
    threat_density: str = "medium",
    time_of_day: TimeOfDay = TimeOfDay.DUSK,
) -> MilitaryAirSimConfig:
    """
    Get contested airspace with configurable threat density.

    Args:
        threat_density: "low", "medium", or "high"
        time_of_day: Time setting

    Returns:
        Configured contested airspace
    """
    config = _create_contested_airspace()
    config.time_of_day = time_of_day

    # Adjust threat count based on density
    if threat_density == "low":
        config.threats = config.threats[:2]  # Only long-range SAMs
    elif threat_density == "high":
        # Add more threats
        config.threats.extend([
            ThreatSystemConfig(
                system_type="SAM_SHORT",
                position=(7000.0, 4000.0, 0.0),
                detection_range=30000.0,
                engagement_range=12000.0,
                active=True,
            ),
            ThreatSystemConfig(
                system_type="MANPADS",
                position=(10000.0, 1000.0, 0.0),
                detection_range=5000.0,
                engagement_range=4000.0,
                active=True,
            ),
        ])

    return config


def get_urban_strike_zone(
    num_targets: int = 3,
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON,
) -> MilitaryAirSimConfig:
    """
    Get urban strike zone with configurable targets.

    Args:
        num_targets: Number of targets to include
        time_of_day: Time setting

    Returns:
        Configured urban environment
    """
    config = _create_urban_strike_zone()
    config.time_of_day = time_of_day
    config.targets = config.targets[:num_targets]

    return config


def export_environment_settings(
    config: MilitaryAirSimConfig,
    output_path: str
) -> str:
    """
    Export environment configuration to AirSim settings.json.

    Args:
        config: Environment configuration
        output_path: Path for output file

    Returns:
        Path to exported file
    """
    settings = config.to_settings_json()

    with open(output_path, 'w') as f:
        json.dump(settings, f, indent=2)

    print(f"Exported AirSim settings to: {output_path}")
    return output_path


class MilitaryAirSimBridge:
    """
    Bridge for running trained policies in military AirSim environments.

    Handles:
    - Environment configuration
    - Policy deployment
    - Visualization settings
    - Recording and replay
    """

    def __init__(
        self,
        environment: MilitaryAirSimConfig,
        policy_path: Optional[str] = None,
    ):
        """
        Initialize AirSim bridge.

        Args:
            environment: Environment configuration
            policy_path: Path to trained policy (ONNX or TorchScript)
        """
        self.environment = environment
        self.policy_path = policy_path
        self.client = None

    def connect(self, ip: str = "127.0.0.1", port: int = 41451):
        """
        Connect to AirSim instance.

        Args:
            ip: AirSim IP address
            port: AirSim port
        """
        try:
            import airsim
            self.client = airsim.MultirotorClient(ip=ip, port=port)
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            print(f"Connected to AirSim at {ip}:{port}")
        except ImportError:
            raise ImportError("Install airsim: pip install airsim")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to AirSim: {e}")

    def setup_environment(self):
        """Apply environment configuration to AirSim."""
        if self.client is None:
            raise RuntimeError("Not connected to AirSim")

        # Set weather if supported
        if hasattr(self.client, 'simSetWeatherParameter'):
            weather_map = {
                WeatherCondition.RAIN: ("Rain", 0.5),
                WeatherCondition.FOG: ("Fog", 0.6),
                WeatherCondition.STORM: ("Rain", 0.8),
            }
            if self.environment.weather in weather_map:
                param, value = weather_map[self.environment.weather]
                self.client.simSetWeatherParameter(param, value)

        print(f"Environment configured: {self.environment.name}")

    def run_policy_visualization(
        self,
        duration: float = 60.0,
        record: bool = False,
    ):
        """
        Run policy in AirSim for visualization.

        Args:
            duration: Duration to run [seconds]
            record: Whether to record the session
        """
        if self.client is None:
            raise RuntimeError("Not connected to AirSim")

        if self.policy_path is None:
            raise ValueError("No policy path specified")

        # Load policy
        if self.policy_path.endswith('.onnx'):
            from deployment.model_export import OnnxInferenceEngine
            policy = OnnxInferenceEngine(self.policy_path)
        else:
            raise ValueError(f"Unsupported policy format: {self.policy_path}")

        # Start recording if requested
        if record:
            self.client.startRecording()

        print(f"Running policy visualization for {duration}s...")

        # Main loop would go here
        # (Actual implementation depends on AirSim API version)

        if record:
            self.client.stopRecording()
            print("Recording saved")

    def set_camera_view(self, camera_name: str):
        """
        Set camera to a predefined position.

        Args:
            camera_name: Name of camera position from config
        """
        for cam in self.environment.camera_positions:
            if cam.get("name") == camera_name:
                pos = cam.get("position", (0, 0, 0))
                rot = cam.get("rotation", (0, 0, 0))

                if self.client and hasattr(self.client, 'simSetCameraPose'):
                    import airsim
                    pose = airsim.Pose(
                        airsim.Vector3r(pos[0], pos[1], pos[2]),
                        airsim.to_quaternion(rot[0], rot[1], rot[2])
                    )
                    self.client.simSetCameraPose("0", pose)

                print(f"Camera set to: {camera_name}")
                return

        print(f"Camera position not found: {camera_name}")

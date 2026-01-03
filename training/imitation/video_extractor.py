"""
Video-to-Trajectory Extraction

Extract drone trajectories from video footage for imitation learning.

This is the key to learning from:
- YouTube videos of expert pilots
- Cockpit footage from real flights
- FPV drone racing videos
- Military flight recordings

Pipeline:
1. Video → Frame extraction
2. Frames → Object detection (drone localization)
3. Detection → Pose estimation (6-DOF)
4. Poses → State trajectory
5. State trajectory → Demonstration object

Supported methods:
- Visual odometry (camera-only)
- Marker-based tracking (ArUco, AprilTag)
- Deep learning pose estimation
- Optical flow velocity estimation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime

from training.imitation.demonstration import (
    Demonstration,
    DemonstrationStep,
    DemonstrationDataset,
)
from simulation.observation_schema import (
    build_canonical_observation,
    DEFAULT_SPEED_OF_SOUND_M_S,
)


@dataclass
class VideoMetadata:
    """Metadata about a video file."""
    filepath: str
    width: int = 0
    height: int = 0
    fps: float = 30.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    codec: str = ""

    # Source info
    source: str = "unknown"  # youtube, cockpit, fpv, simulation
    aircraft_type: str = "unknown"
    pilot_info: str = ""

    # Camera info (if known)
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    fov_degrees: float = 90.0


@dataclass
class ExtractedPose:
    """Extracted pose from a single frame."""
    frame_idx: int
    timestamp: float

    # Position (in camera/world frame)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z]

    # Orientation (quaternion or euler)
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # [w, x, y, z]
    euler_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [roll, pitch, yaw]

    # Velocity (if estimable)
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    angular_velocity: Optional[np.ndarray] = None  # [p, q, r]

    # Confidence/quality metrics
    confidence: float = 1.0
    detection_score: float = 1.0
    tracking_quality: float = 1.0

    # Raw detection info
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    keypoints: Optional[np.ndarray] = None


@dataclass
class ExtractionConfig:
    """Configuration for trajectory extraction."""
    # Frame extraction
    sample_rate_hz: float = 30.0  # Target extraction rate
    start_time: float = 0.0
    end_time: Optional[float] = None

    # Detection method
    detection_method: str = "yolo"  # yolo, ssd, marker, manual
    pose_method: str = "pnp"  # pnp, deep, visual_odometry

    # Pose estimation
    use_smoothing: bool = True
    smoothing_window: int = 5
    outlier_rejection: bool = True
    outlier_threshold: float = 3.0  # Standard deviations

    # Velocity estimation
    estimate_velocity: bool = True
    velocity_method: str = "finite_diff"  # finite_diff, optical_flow, kalman

    # Quality filtering
    min_confidence: float = 0.5
    min_tracking_quality: float = 0.3

    # Output
    output_coordinate_frame: str = "ned"  # ned, enu, camera
    observation_schema: str = "canonical"  # canonical, legacy
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S


class VideoTrajectoryExtractor:
    """
    Main class for extracting trajectories from video.

    Workflow:
    1. Load video and detect drone in each frame
    2. Estimate 6-DOF pose from detections
    3. Apply smoothing and filtering
    4. Convert to demonstration format
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or ExtractionConfig()
        self.device = device

        # Initialize detection/pose models (lazy loading)
        self._detector = None
        self._pose_estimator = None
        self._tracker = None

        # Extraction state
        self.video_metadata: Optional[VideoMetadata] = None
        self.extracted_poses: List[ExtractedPose] = []

    def load_video(self, filepath: str) -> VideoMetadata:
        """
        Load video file and extract metadata.

        Args:
            filepath: Path to video file

        Returns:
            VideoMetadata object
        """
        try:
            import cv2

            cap = cv2.VideoCapture(filepath)

            if not cap.isOpened():
                raise ValueError(f"Could not open video: {filepath}")

            metadata = VideoMetadata(
                filepath=filepath,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS),
                total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                codec=str(int(cap.get(cv2.CAP_PROP_FOURCC))),
            )
            metadata.duration_seconds = metadata.total_frames / metadata.fps

            cap.release()

            self.video_metadata = metadata
            print(f"Loaded video: {metadata.width}x{metadata.height} @ {metadata.fps}fps, "
                  f"{metadata.duration_seconds:.1f}s")

            return metadata

        except ImportError:
            print("Warning: OpenCV not installed. Install with: pip install opencv-python")
            raise

    def _init_detector(self):
        """Initialize object detector."""
        if self._detector is not None:
            return

        method = self.config.detection_method

        if method == "yolo":
            self._detector = self._create_yolo_detector()
        elif method == "marker":
            self._detector = self._create_marker_detector()
        else:
            # Fallback to simple blob detection
            self._detector = self._create_blob_detector()

    def _create_yolo_detector(self):
        """Create YOLO-based drone detector."""
        try:
            # Try ultralytics YOLO
            from ultralytics import YOLO

            # Load pre-trained model (would need drone-specific fine-tuning)
            model = YOLO('yolov8n.pt')
            return model

        except ImportError:
            print("Warning: ultralytics not installed. Using fallback detector.")
            return None

    def _create_marker_detector(self):
        """Create ArUco/AprilTag marker detector."""
        try:
            import cv2
            import cv2.aruco as aruco

            # Create ArUco detector
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()

            return {
                'dictionary': aruco_dict,
                'parameters': parameters,
                'type': 'aruco',
            }

        except ImportError:
            print("Warning: OpenCV ArUco not available.")
            return None

    def _create_blob_detector(self):
        """Create simple blob detector as fallback."""
        try:
            import cv2

            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 100
            params.maxArea = 50000
            params.filterByCircularity = False

            return cv2.SimpleBlobDetector_create(params)

        except ImportError:
            return None

    def detect_drone(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect drone in a single frame.

        Args:
            frame: BGR image array

        Returns:
            Detection dict with bounding box, confidence, etc.
        """
        self._init_detector()

        if self._detector is None:
            return None

        try:
            import cv2

            method = self.config.detection_method

            if method == "yolo" and hasattr(self._detector, 'predict'):
                results = self._detector.predict(frame, verbose=False)
                # Find drone class (or use generic object)
                for r in results:
                    for box in r.boxes:
                        return {
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'confidence': float(box.conf),
                            'class': int(box.cls),
                        }

            elif method == "marker":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self._detector['dictionary'],
                    parameters=self._detector['parameters'],
                )
                if ids is not None and len(ids) > 0:
                    # Use first marker
                    marker_corners = corners[0][0]
                    center = np.mean(marker_corners, axis=0)
                    return {
                        'bbox': cv2.boundingRect(marker_corners.astype(np.int32)),
                        'corners': marker_corners,
                        'marker_id': int(ids[0]),
                        'confidence': 1.0,
                    }

            else:
                # Blob detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints = self._detector.detect(gray)
                if keypoints:
                    kp = keypoints[0]
                    return {
                        'center': (kp.pt[0], kp.pt[1]),
                        'size': kp.size,
                        'confidence': 0.5,
                    }

        except Exception as e:
            print(f"Detection error: {e}")

        return None

    def estimate_pose(
        self,
        detection: Dict[str, Any],
        camera_matrix: Optional[np.ndarray] = None,
    ) -> Optional[ExtractedPose]:
        """
        Estimate 6-DOF pose from detection.

        Args:
            detection: Detection result from detect_drone()
            camera_matrix: Camera intrinsics (for PnP)

        Returns:
            ExtractedPose or None
        """
        try:
            import cv2

            pose = ExtractedPose(frame_idx=0, timestamp=0.0)

            if 'corners' in detection:
                # Marker-based pose estimation
                marker_size = 0.1  # Marker size in meters

                obj_points = np.array([
                    [-marker_size / 2, marker_size / 2, 0],
                    [marker_size / 2, marker_size / 2, 0],
                    [marker_size / 2, -marker_size / 2, 0],
                    [-marker_size / 2, -marker_size / 2, 0],
                ], dtype=np.float32)

                if camera_matrix is None:
                    # Default camera matrix (rough estimate)
                    fx = fy = 800
                    cx, cy = 320, 240
                    camera_matrix = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ], dtype=np.float32)

                dist_coeffs = np.zeros(5)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    detection['corners'].astype(np.float32),
                    camera_matrix,
                    dist_coeffs,
                )

                if success:
                    pose.position = tvec.flatten()
                    # Convert rotation vector to euler angles
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    pose.euler_angles = self._rotation_matrix_to_euler(rot_mat)
                    pose.confidence = detection.get('confidence', 1.0)

            elif 'bbox' in detection:
                # Bounding box-based position estimate (depth from size)
                bbox = detection['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    size = max(x2 - x1, y2 - y1)

                    # Rough depth estimate (assumes known drone size)
                    reference_size = 100  # pixels at 1 meter
                    estimated_depth = reference_size / max(size, 1)

                    # Convert to 3D (assuming centered camera)
                    pose.position = np.array([
                        (center_x - 320) * estimated_depth / 800,
                        (center_y - 240) * estimated_depth / 800,
                        estimated_depth,
                    ])
                    pose.confidence = detection.get('confidence', 0.5)

            return pose

        except Exception as e:
            print(f"Pose estimation error: {e}")
            return None

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    def extract_trajectory(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ExtractedPose]:
        """
        Extract full trajectory from video.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            List of extracted poses
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV required. Install with: pip install opencv-python")
            return []

        # Load video
        self.load_video(video_path)
        cap = cv2.VideoCapture(video_path)

        # Calculate frame sampling
        video_fps = self.video_metadata.fps
        target_fps = self.config.sample_rate_hz
        frame_skip = max(1, int(video_fps / target_fps))

        # Extraction loop
        self.extracted_poses = []
        frame_idx = 0
        extracted_count = 0

        start_frame = int(self.config.start_time * video_fps)
        end_frame = (
            int(self.config.end_time * video_fps)
            if self.config.end_time
            else self.video_metadata.total_frames
        )

        print(f"Extracting trajectory from frames {start_frame} to {end_frame}...")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx < start_frame:
                frame_idx += 1
                continue

            if frame_idx >= end_frame:
                break

            # Process every nth frame
            if (frame_idx - start_frame) % frame_skip == 0:
                # Detect drone
                detection = self.detect_drone(frame)

                if detection is not None:
                    # Estimate pose
                    pose = self.estimate_pose(detection)

                    if pose is not None:
                        pose.frame_idx = frame_idx
                        pose.timestamp = frame_idx / video_fps

                        if pose.confidence >= self.config.min_confidence:
                            self.extracted_poses.append(pose)
                            extracted_count += 1

                if progress_callback:
                    progress_callback(frame_idx, end_frame)

            frame_idx += 1

        cap.release()

        print(f"Extracted {len(self.extracted_poses)} poses from {frame_idx} frames")

        # Post-processing
        if self.config.use_smoothing:
            self._smooth_trajectory()

        if self.config.estimate_velocity:
            self._estimate_velocities()

        return self.extracted_poses

    def _smooth_trajectory(self):
        """Apply smoothing to extracted trajectory."""
        if len(self.extracted_poses) < self.config.smoothing_window:
            return

        window = self.config.smoothing_window
        half_window = window // 2

        # Smooth positions
        positions = np.array([p.position for p in self.extracted_poses])

        smoothed_positions = np.zeros_like(positions)
        for i in range(len(positions)):
            start = max(0, i - half_window)
            end = min(len(positions), i + half_window + 1)
            smoothed_positions[i] = np.mean(positions[start:end], axis=0)

        # Update poses
        for i, pose in enumerate(self.extracted_poses):
            pose.position = smoothed_positions[i]

        print(f"Applied smoothing with window size {window}")

    def _estimate_velocities(self):
        """Estimate velocities from position trajectory."""
        if len(self.extracted_poses) < 2:
            return

        for i in range(len(self.extracted_poses)):
            if i == 0:
                # Forward difference
                dt = self.extracted_poses[1].timestamp - self.extracted_poses[0].timestamp
                if dt > 0:
                    vel = (self.extracted_poses[1].position - self.extracted_poses[0].position) / dt
                    self.extracted_poses[0].velocity = vel
            elif i == len(self.extracted_poses) - 1:
                # Backward difference
                dt = self.extracted_poses[i].timestamp - self.extracted_poses[i - 1].timestamp
                if dt > 0:
                    vel = (self.extracted_poses[i].position - self.extracted_poses[i - 1].position) / dt
                    self.extracted_poses[i].velocity = vel
            else:
                # Central difference
                dt = self.extracted_poses[i + 1].timestamp - self.extracted_poses[i - 1].timestamp
                if dt > 0:
                    vel = (self.extracted_poses[i + 1].position - self.extracted_poses[i - 1].position) / dt
                    self.extracted_poses[i].velocity = vel

        print("Estimated velocities from position trajectory")

    def to_demonstration(
        self,
        pilot_id: str = "video_extraction",
        task_type: str = "extracted",
        observation_builder: Optional[Callable[[ExtractedPose], np.ndarray]] = None,
        action_builder: Optional[Callable[[ExtractedPose, ExtractedPose], np.ndarray]] = None,
    ) -> Demonstration:
        """
        Convert extracted trajectory to Demonstration format.

        Args:
            pilot_id: ID for the pilot/source
            task_type: Type of task
            observation_builder: Custom function to build observations from poses
            action_builder: Custom function to build actions from pose transitions

        Returns:
            Demonstration object
        """
        if not self.extracted_poses:
            raise ValueError("No poses extracted. Run extract_trajectory() first.")

        demo = Demonstration(
            pilot_id=pilot_id,
            source="video",
            task_type=task_type,
            sample_rate_hz=self.config.sample_rate_hz,
            aircraft_type=self.video_metadata.aircraft_type if self.video_metadata else "unknown",
        )

        # Default observation builder: [position, velocity, orientation]
        def default_obs_builder(pose: ExtractedPose) -> np.ndarray:
            obs = np.concatenate([
                pose.position,
                pose.velocity if pose.velocity is not None else np.zeros(3),
                pose.euler_angles,
            ])
            return obs

        # Default action builder: change in orientation (approximate control)
        def default_action_builder(pose: ExtractedPose, next_pose: ExtractedPose) -> np.ndarray:
            dt = next_pose.timestamp - pose.timestamp
            if dt > 0 and pose.velocity is not None and next_pose.velocity is not None:
                # Approximate thrust from vertical acceleration
                accel = (next_pose.velocity - pose.velocity) / dt
                thrust_approx = accel[2] + 9.81  # Compensate gravity

                # Angular rates from orientation change
                angle_diff = next_pose.euler_angles - pose.euler_angles
                angular_rates = angle_diff / dt

                return np.array([
                    angular_rates[0],  # Roll rate
                    angular_rates[1],  # Pitch rate
                    angular_rates[2],  # Yaw rate
                    np.clip(thrust_approx / 20.0, 0, 1),  # Normalized thrust
                ])
            return np.zeros(4)

        obs_builder = observation_builder or default_obs_builder
        act_builder = action_builder or default_action_builder

        schema = (self.config.observation_schema or "legacy").lower()
        use_canonical = schema == "canonical" and observation_builder is None

        def build_canonical_obs(pose: ExtractedPose, next_pose: ExtractedPose) -> np.ndarray:
            dt = next_pose.timestamp - pose.timestamp
            angular_vel = None
            if dt > 0:
                angular_vel = (next_pose.euler_angles - pose.euler_angles) / dt
            return build_canonical_observation(
                position=pose.position,
                velocity=pose.velocity,
                attitude=pose.euler_angles,
                angular_velocity=angular_vel,
                speed_of_sound_m_s=self.config.speed_of_sound_m_s,
            )

        # Build demonstration steps
        for i in range(len(self.extracted_poses) - 1):
            pose = self.extracted_poses[i]
            next_pose = self.extracted_poses[i + 1]

            if use_canonical:
                observation = build_canonical_obs(pose, next_pose)
            else:
                observation = obs_builder(pose)
            action = act_builder(pose, next_pose)

            step = DemonstrationStep(
                observation=observation,
                action=action,
                timestamp=pose.timestamp,
                position=pose.position.copy(),
                velocity=pose.velocity.copy() if pose.velocity is not None else None,
                orientation=pose.euler_angles.copy(),
                info={
                    'frame_idx': pose.frame_idx,
                    'confidence': pose.confidence,
                },
            )

            demo.add_step(step)

        demo.quality_score = np.mean([p.confidence for p in self.extracted_poses])

        print(f"Created demonstration with {demo.num_steps} steps")
        return demo


def extract_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    config: Optional[ExtractionConfig] = None,
    pilot_id: str = "video",
    verbose: bool = True,
) -> Demonstration:
    """
    Convenience function to extract demonstration from video.

    Args:
        video_path: Path to video file
        output_path: Optional path to save demonstration
        config: Extraction configuration
        pilot_id: Pilot ID for demonstration
        verbose: Print progress

    Returns:
        Extracted Demonstration
    """
    extractor = VideoTrajectoryExtractor(config=config)

    if verbose:
        print(f"Extracting trajectory from: {video_path}")

    extractor.extract_trajectory(video_path)
    demo = extractor.to_demonstration(pilot_id=pilot_id)

    if output_path:
        demo.save(output_path)
        if verbose:
            print(f"Saved demonstration to: {output_path}")

    return demo


def extract_from_flight_log(
    log_path: str,
    log_format: str = "px4",
    output_path: Optional[str] = None,
    observation_schema: str = "canonical",
) -> Demonstration:
    """
    Extract demonstration from flight controller logs.

    Supports:
    - PX4 ULog format
    - ArduPilot DataFlash logs
    - Betaflight blackbox logs
    - Generic CSV format

    Args:
        log_path: Path to log file
        log_format: Format of log file
        output_path: Optional path to save demonstration

    Returns:
        Extracted Demonstration
    """
    demo = Demonstration(
        source="telemetry",
        task_type="flight_log",
    )

    if log_format == "px4":
        demo = _parse_px4_ulog(log_path, observation_schema=observation_schema)
    elif log_format == "ardupilot":
        demo = _parse_ardupilot_log(log_path)
    elif log_format == "betaflight":
        demo = _parse_betaflight_blackbox(log_path, observation_schema=observation_schema)
    elif log_format == "csv":
        demo = _parse_csv_log(log_path, observation_schema=observation_schema)
    else:
        raise ValueError(f"Unsupported log format: {log_format}")

    if output_path:
        demo.save(output_path)

    return demo


def _parse_px4_ulog(log_path: str, observation_schema: str = "canonical") -> Demonstration:
    """Parse PX4 ULog format."""
    try:
        from pyulog import ULog

        ulog = ULog(log_path)

        demo = Demonstration(
            source="px4_telemetry",
            pilot_id="px4_pilot",
            aircraft_type="multicopter",
        )

        # Extract vehicle_local_position
        try:
            pos_data = ulog.get_dataset('vehicle_local_position')

            timestamps = pos_data.data['timestamp'] / 1e6  # Convert to seconds
            x = pos_data.data['x']
            y = pos_data.data['y']
            z = pos_data.data['z']
            vx = pos_data.data['vx']
            vy = pos_data.data['vy']
            vz = pos_data.data['vz']

            # Extract attitude
            att_data = ulog.get_dataset('vehicle_attitude')
            roll = att_data.data['roll']
            pitch = att_data.data['pitch']
            yaw = att_data.data['yaw']

            # Extract actuator outputs (control inputs)
            actuator_data = ulog.get_dataset('actuator_outputs')
            outputs = actuator_data.data['output']

            # Build demonstration steps
            n_samples = min(len(timestamps), len(roll))

            use_canonical = (observation_schema or "legacy").lower() == "canonical"

            for i in range(n_samples - 1):
                if use_canonical:
                    angular_vel = None
                    dt = timestamps[i + 1] - timestamps[i]
                    if dt > 0:
                        angular_vel = np.array([
                            (roll[i + 1] - roll[i]) / dt,
                            (pitch[i + 1] - pitch[i]) / dt,
                            (yaw[i + 1] - yaw[i]) / dt,
                        ], dtype=np.float32)

                    observation = build_canonical_observation(
                        position=[x[i], y[i], z[i]],
                        velocity=[vx[i], vy[i], vz[i]],
                        attitude=[roll[i], pitch[i], yaw[i]],
                        angular_velocity=angular_vel,
                    )
                else:
                    observation = np.array([
                        x[i], y[i], z[i],
                        vx[i], vy[i], vz[i],
                        roll[i], pitch[i], yaw[i],
                    ])

                # Action: motor outputs (normalized)
                if i < len(outputs):
                    action = np.clip(outputs[i][:4] / 2000.0, 0, 1)
                else:
                    action = np.zeros(4)

                step = DemonstrationStep(
                    observation=observation,
                    action=action,
                    timestamp=timestamps[i],
                    position=np.array([x[i], y[i], z[i]]),
                    velocity=np.array([vx[i], vy[i], vz[i]]),
                    orientation=np.array([roll[i], pitch[i], yaw[i]]),
                )

                demo.add_step(step)

        except Exception as e:
            print(f"Error parsing ULog topics: {e}")

        return demo

    except ImportError:
        print("pyulog required. Install with: pip install pyulog")
        return Demonstration()


def _parse_ardupilot_log(log_path: str) -> Demonstration:
    """Parse ArduPilot DataFlash log."""
    # Simplified implementation - full version would use pymavlink
    demo = Demonstration(
        source="ardupilot_telemetry",
        aircraft_type="multicopter",
    )

    print("ArduPilot log parsing requires pymavlink")
    return demo


def _parse_betaflight_blackbox(log_path: str, observation_schema: str = "canonical") -> Demonstration:
    """Parse Betaflight blackbox log."""
    demo = Demonstration(
        source="betaflight_blackbox",
        aircraft_type="quadcopter",
    )

    try:
        # Betaflight logs are CSV after decoding with blackbox_decode
        with open(log_path, 'r') as f:
            import csv
            reader = csv.DictReader(f)

            use_canonical = (observation_schema or "legacy").lower() == "canonical"

            for i, row in enumerate(reader):
                try:
                    # Common Betaflight blackbox fields
                    gyro = np.array([
                        float(row.get('gyroADC[0]', 0)) / 1000.0,
                        float(row.get('gyroADC[1]', 0)) / 1000.0,
                        float(row.get('gyroADC[2]', 0)) / 1000.0,
                    ], dtype=np.float32)

                    acc = np.array([
                        float(row.get('accSmooth[0]', 0)) / 1000.0,
                        float(row.get('accSmooth[1]', 0)) / 1000.0,
                        float(row.get('accSmooth[2]', 0)) / 1000.0,
                    ], dtype=np.float32)

                    if use_canonical:
                        observation = build_canonical_observation(
                            position=None,
                            velocity=None,
                            attitude=None,
                            angular_velocity=gyro,
                        )
                    else:
                        observation = np.concatenate([gyro, acc])

                    # RC commands as actions
                    action = np.array([
                        float(row.get('rcCommand[0]', 1500)) / 500.0 - 3.0,  # Roll
                        float(row.get('rcCommand[1]', 1500)) / 500.0 - 3.0,  # Pitch
                        float(row.get('rcCommand[2]', 1500)) / 500.0 - 3.0,  # Yaw
                        float(row.get('rcCommand[3]', 1000)) / 1000.0,  # Throttle
                    ])

                    timestamp = float(row.get('time (us)', i * 1000)) / 1e6

                    step = DemonstrationStep(
                        observation=observation,
                        action=action,
                        timestamp=timestamp,
                    )

                    demo.add_step(step)

                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print(f"Error parsing Betaflight log: {e}")

    return demo


def _parse_csv_log(log_path: str, observation_schema: str = "canonical") -> Demonstration:
    """Parse generic CSV flight log."""
    demo = Demonstration(
        source="csv_telemetry",
    )

    try:
        import csv

        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)

            use_canonical = (observation_schema or "legacy").lower() == "canonical"

            for i, row in enumerate(reader):
                # Try to extract common fields
                try:
                    # Position
                    x = float(row.get('x', row.get('pos_x', row.get('position_x', 0))))
                    y = float(row.get('y', row.get('pos_y', row.get('position_y', 0))))
                    z = float(row.get('z', row.get('pos_z', row.get('position_z', row.get('altitude', 0)))))

                    # Velocity
                    vx = float(row.get('vx', row.get('vel_x', row.get('velocity_x', 0))))
                    vy = float(row.get('vy', row.get('vel_y', row.get('velocity_y', 0))))
                    vz = float(row.get('vz', row.get('vel_z', row.get('velocity_z', 0))))

                    # Attitude
                    roll = float(row.get('roll', row.get('phi', 0)))
                    pitch = float(row.get('pitch', row.get('theta', 0)))
                    yaw = float(row.get('yaw', row.get('psi', 0)))

                    if use_canonical:
                        observation = build_canonical_observation(
                            position=[x, y, z],
                            velocity=[vx, vy, vz],
                            attitude=[roll, pitch, yaw],
                        )
                    else:
                        observation = np.array([x, y, z, vx, vy, vz, roll, pitch, yaw])

                    # Actions (control inputs)
                    thrust = float(row.get('thrust', row.get('throttle', 0.5)))
                    roll_cmd = float(row.get('roll_cmd', row.get('roll_rate', 0)))
                    pitch_cmd = float(row.get('pitch_cmd', row.get('pitch_rate', 0)))
                    yaw_cmd = float(row.get('yaw_cmd', row.get('yaw_rate', 0)))

                    action = np.array([roll_cmd, pitch_cmd, yaw_cmd, thrust])

                    timestamp = float(row.get('time', row.get('timestamp', row.get('t', i * 0.01))))

                    step = DemonstrationStep(
                        observation=observation,
                        action=action,
                        timestamp=timestamp,
                        position=np.array([x, y, z]),
                        velocity=np.array([vx, vy, vz]),
                        orientation=np.array([roll, pitch, yaw]),
                    )

                    demo.add_step(step)

                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print(f"Error parsing CSV log: {e}")

    return demo

"""
Dataset Fingerprinting and Provenance

Tracks dataset provenance for reproducibility:
- Content hash (SHA256)
- Schema version
- Extraction configuration
- Processing history
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from training.imitation.demonstration import Demonstration, DemonstrationDataset


# Schema version - increment when breaking changes occur
SCHEMA_VERSION = "1.0.0"


@dataclass
class ProcessingStep:
    """Record of a processing step applied to data."""
    step_name: str                    # e.g., "quality_filter", "normalize"
    step_version: str = "1.0.0"       # Version of the processing code
    timestamp: str = ""               # When applied
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""              # Hash before processing
    output_hash: str = ""             # Hash after processing
    notes: str = ""                   # Additional notes

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_name': self.step_name,
            'step_version': self.step_version,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingStep':
        return cls(
            step_name=data['step_name'],
            step_version=data.get('step_version', '1.0.0'),
            timestamp=data.get('timestamp', ''),
            parameters=data.get('parameters', {}),
            input_hash=data.get('input_hash', ''),
            output_hash=data.get('output_hash', ''),
            notes=data.get('notes', ''),
        )


@dataclass
class FeatureSchema:
    """
    Schema of features in the dataset.

    CRITICAL: Even small changes to feature schema (e.g., observation vector
    order) can be catastrophic. Include this in the fingerprint hash.
    """
    # Dimensions
    observation_dim: int = 0
    action_dim: int = 0

    # Feature details
    observation_features: List[str] = field(default_factory=list)  # e.g., ["pos_x", "pos_y", "pos_z", ...]
    action_features: List[str] = field(default_factory=list)       # e.g., ["roll", "pitch", "yaw_rate", "thrust"]

    # Sensor channels included
    sensor_channels: List[str] = field(default_factory=list)  # e.g., ["imu", "gps", "barometer"]

    # Normalization applied
    normalization_type: Optional[str] = None  # "min_max", "z_score", None
    normalization_params: Dict[str, Any] = field(default_factory=dict)

    # Data types
    observation_dtype: str = "float32"
    action_dtype: str = "float32"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'observation_features': self.observation_features,
            'action_features': self.action_features,
            'sensor_channels': self.sensor_channels,
            'normalization_type': self.normalization_type,
            'normalization_params': self.normalization_params,
            'observation_dtype': self.observation_dtype,
            'action_dtype': self.action_dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSchema':
        return cls(
            observation_dim=data.get('observation_dim', 0),
            action_dim=data.get('action_dim', 0),
            observation_features=data.get('observation_features', []),
            action_features=data.get('action_features', []),
            sensor_channels=data.get('sensor_channels', []),
            normalization_type=data.get('normalization_type'),
            normalization_params=data.get('normalization_params', {}),
            observation_dtype=data.get('observation_dtype', 'float32'),
            action_dtype=data.get('action_dtype', 'float32'),
        )

    @classmethod
    def from_dataset(cls, dataset: 'DemonstrationDataset') -> 'FeatureSchema':
        """Auto-generate schema from a dataset."""
        if not dataset.demonstrations:
            return cls()

        demo = dataset.demonstrations[0]
        return cls(
            observation_dim=demo.observation_dim,
            action_dim=demo.action_dim,
            observation_dtype=str(demo.observations.dtype),
            action_dtype=str(demo.actions.dtype),
        )


@dataclass
class FilterConfig:
    """Configuration for quality filtering applied to data."""
    filter_version: str = "1.0.0"

    # Thresholds
    max_pose_jitter_m: float = 0.1
    max_missing_frame_ratio: float = 0.05
    max_timestamp_variance_pct: float = 50.0
    min_quality_score: float = 0.5

    # Applied filters
    applied_filters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'filter_version': self.filter_version,
            'max_pose_jitter_m': self.max_pose_jitter_m,
            'max_missing_frame_ratio': self.max_missing_frame_ratio,
            'max_timestamp_variance_pct': self.max_timestamp_variance_pct,
            'min_quality_score': self.min_quality_score,
            'applied_filters': self.applied_filters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterConfig':
        return cls(
            filter_version=data.get('filter_version', '1.0.0'),
            max_pose_jitter_m=data.get('max_pose_jitter_m', 0.1),
            max_missing_frame_ratio=data.get('max_missing_frame_ratio', 0.05),
            max_timestamp_variance_pct=data.get('max_timestamp_variance_pct', 50.0),
            min_quality_score=data.get('min_quality_score', 0.5),
            applied_filters=data.get('applied_filters', []),
        )


@dataclass
class SplitConfig:
    """Configuration for train/val/test splitting."""
    split_version: str = "1.0.0"

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    stratify_by_pilot: bool = True
    stratify_by_task: bool = True
    prevent_pilot_overlap: bool = True
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            'split_version': self.split_version,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'stratify_by_pilot': self.stratify_by_pilot,
            'stratify_by_task': self.stratify_by_task,
            'prevent_pilot_overlap': self.prevent_pilot_overlap,
            'random_seed': self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitConfig':
        return cls(
            split_version=data.get('split_version', '1.0.0'),
            train_ratio=data.get('train_ratio', 0.70),
            val_ratio=data.get('val_ratio', 0.15),
            test_ratio=data.get('test_ratio', 0.15),
            stratify_by_pilot=data.get('stratify_by_pilot', True),
            stratify_by_task=data.get('stratify_by_task', True),
            prevent_pilot_overlap=data.get('prevent_pilot_overlap', True),
            random_seed=data.get('random_seed', 42),
        )


@dataclass
class ExtractionConfig:
    """Configuration used during data extraction."""
    extractor_name: str = "unknown"      # e.g., "telemetry_parser", "video_processor"
    extractor_version: str = "1.0.0"

    # Source information
    source_type: str = "unknown"          # "telemetry", "video", "simulator"
    source_format: str = "unknown"        # "px4_log", "ardupilot_bin", "airsim_json"

    # Camera parameters (if from video)
    camera_intrinsics: Optional[Dict[str, Any]] = None
    camera_extrinsics: Optional[Dict[str, Any]] = None

    # Model versions (if using ML)
    pose_model: Optional[str] = None      # e.g., "mediapipe_v0.10"
    detection_model: Optional[str] = None

    # Processing parameters
    sample_rate_hz: float = 100.0
    coordinate_frame: str = "NED"         # NED, ENU, etc.
    normalization: Optional[str] = None   # "min_max", "z_score", etc.

    # Smoothing parameters
    smoothing_method: Optional[str] = None  # "kalman", "moving_avg", None
    smoothing_window: int = 0

    # Additional config
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'extractor_name': self.extractor_name,
            'extractor_version': self.extractor_version,
            'source_type': self.source_type,
            'source_format': self.source_format,
            'camera_intrinsics': self.camera_intrinsics,
            'camera_extrinsics': self.camera_extrinsics,
            'pose_model': self.pose_model,
            'detection_model': self.detection_model,
            'sample_rate_hz': self.sample_rate_hz,
            'coordinate_frame': self.coordinate_frame,
            'normalization': self.normalization,
            'smoothing_method': self.smoothing_method,
            'smoothing_window': self.smoothing_window,
            'extra': self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConfig':
        return cls(
            extractor_name=data.get('extractor_name', 'unknown'),
            extractor_version=data.get('extractor_version', '1.0.0'),
            source_type=data.get('source_type', 'unknown'),
            source_format=data.get('source_format', 'unknown'),
            camera_intrinsics=data.get('camera_intrinsics'),
            camera_extrinsics=data.get('camera_extrinsics'),
            pose_model=data.get('pose_model'),
            detection_model=data.get('detection_model'),
            sample_rate_hz=data.get('sample_rate_hz', 100.0),
            coordinate_frame=data.get('coordinate_frame', 'NED'),
            normalization=data.get('normalization'),
            smoothing_method=data.get('smoothing_method'),
            smoothing_window=data.get('smoothing_window', 0),
            extra=data.get('extra', {}),
        )


@dataclass
class DatasetFingerprint:
    """
    Complete fingerprint for dataset provenance tracking.

    Enables:
    - Reproducibility: Know exact data version used
    - Debugging: Track processing history
    - Compliance: Audit trail for data lineage

    IMPORTANT: The content_hash includes:
    - Raw data (observations, actions)
    - Extraction config (model versions, camera intrinsics, etc.)
    - Filter config (thresholds applied)
    - Split config (if applicable)

    This ensures that "same hash" means truly identical processing.
    """
    # Identity
    dataset_id: str = ""                  # Unique dataset identifier
    content_hash: str = ""                # SHA256 of content + configs

    # Schema
    schema_version: str = SCHEMA_VERSION

    # Dataset statistics
    num_demonstrations: int = 0
    total_steps: int = 0
    total_transitions: int = 0

    # Configs (all included in hash)
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)
    filter_config: Optional[FilterConfig] = None
    split_config: Optional[SplitConfig] = None
    feature_schema: Optional[FeatureSchema] = None

    # Processing history
    processing_history: List[ProcessingStep] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    last_modified_at: str = ""

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_modified_at:
            self.last_modified_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'content_hash': self.content_hash,
            'schema_version': self.schema_version,
            'num_demonstrations': self.num_demonstrations,
            'total_steps': self.total_steps,
            'total_transitions': self.total_transitions,
            'extraction_config': self.extraction_config.to_dict(),
            'filter_config': self.filter_config.to_dict() if self.filter_config else None,
            'split_config': self.split_config.to_dict() if self.split_config else None,
            'feature_schema': self.feature_schema.to_dict() if self.feature_schema else None,
            'processing_history': [p.to_dict() for p in self.processing_history],
            'created_at': self.created_at,
            'last_modified_at': self.last_modified_at,
            'description': self.description,
            'tags': self.tags,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetFingerprint':
        extraction = ExtractionConfig.from_dict(data.get('extraction_config', {}))
        filter_cfg = FilterConfig.from_dict(data['filter_config']) if data.get('filter_config') else None
        split_cfg = SplitConfig.from_dict(data['split_config']) if data.get('split_config') else None
        schema = FeatureSchema.from_dict(data['feature_schema']) if data.get('feature_schema') else None
        history = [ProcessingStep.from_dict(p) for p in data.get('processing_history', [])]

        return cls(
            dataset_id=data.get('dataset_id', ''),
            content_hash=data.get('content_hash', ''),
            schema_version=data.get('schema_version', SCHEMA_VERSION),
            num_demonstrations=data.get('num_demonstrations', 0),
            total_steps=data.get('total_steps', 0),
            total_transitions=data.get('total_transitions', 0),
            extraction_config=extraction,
            filter_config=filter_cfg,
            split_config=split_cfg,
            feature_schema=schema,
            processing_history=history,
            created_at=data.get('created_at', ''),
            last_modified_at=data.get('last_modified_at', ''),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
        )

    def add_processing_step(
        self,
        step_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ):
        """Add a processing step to history."""
        step = ProcessingStep(
            step_name=step_name,
            parameters=parameters or {},
            input_hash=self.content_hash,
            notes=notes,
        )
        self.processing_history.append(step)
        self.last_modified_at = datetime.now().isoformat()

    def save(self, filepath: str):
        """Save fingerprint to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'DatasetFingerprint':
        """Load fingerprint from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class DatasetFingerprintGenerator:
    """Generates fingerprints for datasets."""

    def __init__(
        self,
        extraction_config: Optional[ExtractionConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        split_config: Optional[SplitConfig] = None,
        feature_schema: Optional[FeatureSchema] = None,
    ):
        """
        Args:
            extraction_config: Extraction configuration to record
            filter_config: Filter configuration to record
            split_config: Split configuration to record
            feature_schema: Feature schema (auto-generated if None)
        """
        self.extraction_config = extraction_config or ExtractionConfig()
        self.filter_config = filter_config
        self.split_config = split_config
        self.feature_schema = feature_schema

    def generate(
        self,
        dataset: 'DemonstrationDataset',
        dataset_id: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> DatasetFingerprint:
        """
        Generate fingerprint for a dataset.

        The hash includes:
        - All observation/action data
        - Feature schema (dimensions, dtypes, feature names)
        - Extraction config (model versions, camera params, etc.)
        - Filter config (thresholds)
        - Split config (if applicable)

        Args:
            dataset: Dataset to fingerprint
            dataset_id: Optional ID (auto-generated if not provided)
            description: Dataset description
            tags: Dataset tags

        Returns:
            DatasetFingerprint
        """
        # Auto-generate feature schema if not provided
        schema = self.feature_schema or FeatureSchema.from_dataset(dataset)

        # Compute content hash (includes configs + schema)
        content_hash = self._compute_content_hash(dataset, schema)

        # Generate ID if not provided
        if dataset_id is None:
            dataset_id = f"dataset_{content_hash[:12]}"

        return DatasetFingerprint(
            dataset_id=dataset_id,
            content_hash=content_hash,
            num_demonstrations=len(dataset.demonstrations),
            total_steps=dataset.total_steps,
            total_transitions=dataset.total_transitions,
            extraction_config=self.extraction_config,
            filter_config=self.filter_config,
            split_config=self.split_config,
            feature_schema=schema,
            description=description,
            tags=tags or [],
        )

    def _compute_content_hash(
        self,
        dataset: 'DemonstrationDataset',
        schema: FeatureSchema,
    ) -> str:
        """
        Compute SHA256 hash of dataset content + configs + schema.

        IMPORTANT: This includes:
        - Feature schema (obs/action dims, feature names, dtypes)
        - Extraction config (model versions, camera params)
        - Filter config (thresholds)
        - Split config (ratios, stratification)
        - Raw data (observations, actions)

        A change to ANY of these changes the hash.
        """
        hasher = hashlib.sha256()

        # Hash feature schema (CRITICAL for detecting format changes)
        schema_str = json.dumps(schema.to_dict(), sort_keys=True)
        hasher.update(f"schema:{schema_str}".encode())

        # Hash extraction config
        extraction_str = json.dumps(self.extraction_config.to_dict(), sort_keys=True)
        hasher.update(f"extraction:{extraction_str}".encode())

        # Hash filter config (if present)
        if self.filter_config:
            filter_str = json.dumps(self.filter_config.to_dict(), sort_keys=True)
            hasher.update(f"filter:{filter_str}".encode())

        # Hash split config (if present)
        if self.split_config:
            split_str = json.dumps(self.split_config.to_dict(), sort_keys=True)
            hasher.update(f"split:{split_str}".encode())

        # Hash dataset content
        for demo in dataset.demonstrations:
            # Hash demo metadata
            meta = f"{demo.demo_id}|{demo.pilot_id}|{demo.task_type}|{demo.num_steps}"
            hasher.update(meta.encode())

            # Hash observations (as bytes)
            obs_bytes = demo.observations.tobytes()
            hasher.update(obs_bytes)

            # Hash actions
            action_bytes = demo.actions.tobytes()
            hasher.update(action_bytes)

        return hasher.hexdigest()


def generate_fingerprint(
    dataset: 'DemonstrationDataset',
    extraction_config: Optional[ExtractionConfig] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> DatasetFingerprint:
    """
    Convenience function to generate a dataset fingerprint.

    Args:
        dataset: Dataset to fingerprint
        extraction_config: Optional extraction configuration
        description: Dataset description
        tags: Dataset tags

    Returns:
        DatasetFingerprint
    """
    generator = DatasetFingerprintGenerator(extraction_config)
    return generator.generate(dataset, description=description, tags=tags)


def verify_fingerprint(
    dataset: 'DemonstrationDataset',
    fingerprint: DatasetFingerprint,
) -> Dict[str, Any]:
    """
    Verify that a dataset matches its fingerprint.

    Args:
        dataset: Dataset to verify
        fingerprint: Expected fingerprint

    Returns:
        Dict with verification results
    """
    generator = DatasetFingerprintGenerator()
    current_hash = generator._compute_content_hash(dataset)

    results = {
        'valid': True,
        'hash_match': current_hash == fingerprint.content_hash,
        'expected_hash': fingerprint.content_hash,
        'actual_hash': current_hash,
        'num_demos_match': len(dataset.demonstrations) == fingerprint.num_demonstrations,
        'steps_match': dataset.total_steps == fingerprint.total_steps,
    }

    if not results['hash_match']:
        results['valid'] = False

    return results


def load_with_verification(
    dataset_path: str,
    fingerprint_path: str,
) -> 'DemonstrationDataset':
    """
    Load a dataset and verify it matches its fingerprint.

    Args:
        dataset_path: Path to dataset directory
        fingerprint_path: Path to fingerprint JSON

    Returns:
        Verified DemonstrationDataset

    Raises:
        ValueError: If verification fails
    """
    from training.imitation.demonstration import DemonstrationDataset

    dataset = DemonstrationDataset.load(dataset_path)
    fingerprint = DatasetFingerprint.load(fingerprint_path)

    verification = verify_fingerprint(dataset, fingerprint)

    if not verification['valid']:
        raise ValueError(
            f"Dataset verification failed:\n"
            f"  Hash match: {verification['hash_match']}\n"
            f"  Expected: {verification['expected_hash']}\n"
            f"  Actual: {verification['actual_hash']}"
        )

    return dataset

"""
Demonstration Data Structures and Collection

Core data structures for storing and managing human pilot demonstrations.
These form the foundation of the imitation learning pipeline.

A demonstration consists of:
- Observations (what the pilot saw)
- Actions (what the pilot did)
- Metadata (pilot info, aircraft type, conditions)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
from datetime import datetime
import hashlib
from collections import defaultdict


@dataclass
class DemonstrationStep:
    """
    A single timestep in a demonstration.

    Captures the state-action pair at one moment in time.
    """
    # Core data
    observation: np.ndarray  # What the pilot observed (state)
    action: np.ndarray       # What the pilot did (control input)

    # Timing
    timestamp: float = 0.0   # Time since start of demonstration
    dt: float = 0.01         # Time delta from previous step

    # Optional enrichments
    next_observation: Optional[np.ndarray] = None  # For transition modeling
    reward: Optional[float] = None                  # If available (for IRL)
    done: bool = False                              # Episode termination
    info: Dict[str, Any] = field(default_factory=dict)

    # Flight data (if available from telemetry)
    position: Optional[np.ndarray] = None      # [x, y, z] in meters
    velocity: Optional[np.ndarray] = None      # [vx, vy, vz] in m/s
    orientation: Optional[np.ndarray] = None   # [roll, pitch, yaw] in radians
    angular_velocity: Optional[np.ndarray] = None  # [p, q, r] in rad/s

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'observation': self.observation.tolist(),
            'action': self.action.tolist(),
            'timestamp': self.timestamp,
            'dt': self.dt,
            'next_observation': self.next_observation.tolist() if self.next_observation is not None else None,
            'reward': self.reward,
            'done': self.done,
            'info': self.info,
            'position': self.position.tolist() if self.position is not None else None,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'orientation': self.orientation.tolist() if self.orientation is not None else None,
            'angular_velocity': self.angular_velocity.tolist() if self.angular_velocity is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DemonstrationStep':
        """Create from dictionary."""
        return cls(
            observation=np.array(data['observation']),
            action=np.array(data['action']),
            timestamp=data.get('timestamp', 0.0),
            dt=data.get('dt', 0.01),
            next_observation=np.array(data['next_observation']) if data.get('next_observation') else None,
            reward=data.get('reward'),
            done=data.get('done', False),
            info=data.get('info', {}),
            position=np.array(data['position']) if data.get('position') else None,
            velocity=np.array(data['velocity']) if data.get('velocity') else None,
            orientation=np.array(data['orientation']) if data.get('orientation') else None,
            angular_velocity=np.array(data['angular_velocity']) if data.get('angular_velocity') else None,
        )


@dataclass
class Demonstration:
    """
    A complete demonstration trajectory from a human pilot.

    This could come from:
    - Flight simulator recordings
    - Real flight telemetry logs
    - Video analysis + pose estimation
    - Manual flight controllers with logging
    """
    # Trajectory data
    steps: List[DemonstrationStep] = field(default_factory=list)

    # Metadata
    demo_id: str = ""
    pilot_id: str = "unknown"
    pilot_skill_level: str = "unknown"  # novice, intermediate, expert, professional

    # Aircraft info
    aircraft_type: str = "quadcopter"
    aircraft_name: str = ""
    aircraft_mass_kg: float = 1.0

    # Recording info
    source: str = "unknown"  # telemetry, video, simulator, manual
    recorded_at: str = ""
    duration_seconds: float = 0.0
    sample_rate_hz: float = 100.0

    # Task info
    task_type: str = "freeform"  # hover, waypoint, acrobatic, racing, inspection
    task_description: str = ""
    success: bool = True

    # Environment conditions
    environment: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    quality_score: float = 1.0  # 0-1, higher is better
    verified: bool = False

    def __post_init__(self):
        if not self.demo_id:
            self.demo_id = self._generate_id()
        if not self.recorded_at:
            self.recorded_at = datetime.now().isoformat()

    def _generate_id(self) -> str:
        """Generate unique demonstration ID."""
        content = f"{self.pilot_id}_{self.recorded_at}_{len(self.steps)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def observations(self) -> np.ndarray:
        """Get all observations as array."""
        return np.array([step.observation for step in self.steps])

    @property
    def actions(self) -> np.ndarray:
        """Get all actions as array."""
        return np.array([step.action for step in self.steps])

    @property
    def positions(self) -> Optional[np.ndarray]:
        """Get all positions as array (if available)."""
        if self.steps and self.steps[0].position is not None:
            return np.array([step.position for step in self.steps])
        return None

    @property
    def observation_dim(self) -> int:
        """Dimension of observation space."""
        return self.steps[0].observation.shape[0] if self.steps else 0

    @property
    def action_dim(self) -> int:
        """Dimension of action space."""
        return self.steps[0].action.shape[0] if self.steps else 0

    def add_step(self, step: DemonstrationStep):
        """Add a step to the demonstration."""
        self.steps.append(step)
        self.duration_seconds = step.timestamp

    def get_transitions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get (s, a, s', done) transitions for training.

        Returns:
            observations: (N-1, obs_dim)
            actions: (N-1, action_dim)
            next_observations: (N-1, obs_dim)
            dones: (N-1,)
        """
        obs = []
        acts = []
        next_obs = []
        dones = []

        for i in range(len(self.steps) - 1):
            obs.append(self.steps[i].observation)
            acts.append(self.steps[i].action)
            next_obs.append(self.steps[i + 1].observation)
            dones.append(self.steps[i].done)

        return (
            np.array(obs),
            np.array(acts),
            np.array(next_obs),
            np.array(dones),
        )

    def subsample(self, factor: int) -> 'Demonstration':
        """Create subsampled version of demonstration."""
        new_demo = Demonstration(
            demo_id=f"{self.demo_id}_sub{factor}",
            pilot_id=self.pilot_id,
            pilot_skill_level=self.pilot_skill_level,
            aircraft_type=self.aircraft_type,
            aircraft_name=self.aircraft_name,
            aircraft_mass_kg=self.aircraft_mass_kg,
            source=self.source,
            recorded_at=self.recorded_at,
            sample_rate_hz=self.sample_rate_hz / factor,
            task_type=self.task_type,
            task_description=self.task_description,
            success=self.success,
            environment=self.environment.copy(),
            quality_score=self.quality_score,
        )

        new_demo.steps = self.steps[::factor]
        new_demo.duration_seconds = self.duration_seconds

        return new_demo

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'demo_id': self.demo_id,
            'pilot_id': self.pilot_id,
            'pilot_skill_level': self.pilot_skill_level,
            'aircraft_type': self.aircraft_type,
            'aircraft_name': self.aircraft_name,
            'aircraft_mass_kg': self.aircraft_mass_kg,
            'source': self.source,
            'recorded_at': self.recorded_at,
            'duration_seconds': self.duration_seconds,
            'sample_rate_hz': self.sample_rate_hz,
            'task_type': self.task_type,
            'task_description': self.task_description,
            'success': self.success,
            'environment': self.environment,
            'quality_score': self.quality_score,
            'verified': self.verified,
            'steps': [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Demonstration':
        """Create from dictionary."""
        steps = [DemonstrationStep.from_dict(s) for s in data.get('steps', [])]

        return cls(
            steps=steps,
            demo_id=data.get('demo_id', ''),
            pilot_id=data.get('pilot_id', 'unknown'),
            pilot_skill_level=data.get('pilot_skill_level', 'unknown'),
            aircraft_type=data.get('aircraft_type', 'quadcopter'),
            aircraft_name=data.get('aircraft_name', ''),
            aircraft_mass_kg=data.get('aircraft_mass_kg', 1.0),
            source=data.get('source', 'unknown'),
            recorded_at=data.get('recorded_at', ''),
            duration_seconds=data.get('duration_seconds', 0.0),
            sample_rate_hz=data.get('sample_rate_hz', 100.0),
            task_type=data.get('task_type', 'freeform'),
            task_description=data.get('task_description', ''),
            success=data.get('success', True),
            environment=data.get('environment', {}),
            quality_score=data.get('quality_score', 1.0),
            verified=data.get('verified', False),
        )

    def save(self, filepath: str):
        """Save demonstration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'Demonstration':
        """Load demonstration from file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)


class DemonstrationDataset:
    """
    Collection of demonstrations for training.

    Handles:
    - Loading/saving multiple demonstrations
    - Filtering by criteria (pilot, task, quality)
    - Batching for training
    - Statistics and analysis
    """

    def __init__(self, demonstrations: Optional[List[Demonstration]] = None):
        self.demonstrations: List[Demonstration] = demonstrations or []
        self._index = 0

    def __len__(self) -> int:
        return len(self.demonstrations)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> Demonstration:
        if self._index < len(self.demonstrations):
            demo = self.demonstrations[self._index]
            self._index += 1
            return demo
        raise StopIteration

    def __getitem__(self, idx: int) -> Demonstration:
        return self.demonstrations[idx]

    def add(self, demo: Demonstration):
        """Add a demonstration to the dataset."""
        self.demonstrations.append(demo)

    def extend(self, demos: List[Demonstration]):
        """Add multiple demonstrations."""
        self.demonstrations.extend(demos)

    @property
    def total_steps(self) -> int:
        """Total number of steps across all demonstrations."""
        return sum(demo.num_steps for demo in self.demonstrations)

    @property
    def total_transitions(self) -> int:
        """Total number of transitions (steps - num_demos)."""
        return sum(max(0, demo.num_steps - 1) for demo in self.demonstrations)

    @property
    def observation_dim(self) -> int:
        """Observation dimension (from first demo)."""
        return self.demonstrations[0].observation_dim if self.demonstrations else 0

    @property
    def action_dim(self) -> int:
        """Action dimension (from first demo)."""
        return self.demonstrations[0].action_dim if self.demonstrations else 0

    def get_all_observations(self) -> np.ndarray:
        """Get all observations concatenated."""
        all_obs = []
        for demo in self.demonstrations:
            all_obs.append(demo.observations)
        return np.concatenate(all_obs, axis=0)

    def get_all_actions(self) -> np.ndarray:
        """Get all actions concatenated."""
        all_acts = []
        for demo in self.demonstrations:
            all_acts.append(demo.actions)
        return np.concatenate(all_acts, axis=0)

    def get_all_transitions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get all transitions from all demonstrations."""
        all_obs = []
        all_acts = []
        all_next_obs = []
        all_dones = []

        for demo in self.demonstrations:
            obs, acts, next_obs, dones = demo.get_transitions()
            all_obs.append(obs)
            all_acts.append(acts)
            all_next_obs.append(next_obs)
            all_dones.append(dones)

        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_acts, axis=0),
            np.concatenate(all_next_obs, axis=0),
            np.concatenate(all_dones, axis=0),
        )

    def sample_batch(
        self,
        batch_size: int,
        include_next_obs: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            include_next_obs: Whether to include next observations

        Returns:
            Dictionary with 'observations' and 'actions' arrays
        """
        all_obs = self.get_all_observations()
        all_acts = self.get_all_actions()

        indices = np.random.choice(len(all_obs), size=min(batch_size, len(all_obs)), replace=False)

        batch = {
            'observations': all_obs[indices],
            'actions': all_acts[indices],
        }

        if include_next_obs:
            # For transitions, we need to be careful at episode boundaries
            obs, acts, next_obs, dones = self.get_all_transitions()
            indices = np.random.choice(len(obs), size=min(batch_size, len(obs)), replace=False)
            batch = {
                'observations': obs[indices],
                'actions': acts[indices],
                'next_observations': next_obs[indices],
                'dones': dones[indices],
            }

        return batch

    def sample_transition_batch(
        self,
        batch_size: int,
        balance_by: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions, optionally balanced across demo groups.

        Args:
            batch_size: Number of transitions to sample
            balance_by: Metadata field to balance on (e.g., "task_type",
                        "pilot_id", "aircraft_type", "source", or "environment:<key>")

        Returns:
            Dictionary with observations, actions, next_observations, dones, and groups.
        """
        if not balance_by:
            return self.sample_batch(batch_size, include_next_obs=True)

        groups = defaultdict(list)
        for demo in self.demonstrations:
            key = self._get_group_key(demo, balance_by)
            groups[key].append(demo)

        if not groups:
            return self.sample_batch(batch_size, include_next_obs=True)

        per_group = max(1, batch_size // len(groups))
        obs_list = []
        act_list = []
        next_obs_list = []
        dones_list = []
        group_labels = []

        for key, demos in groups.items():
            group_obs = []
            group_acts = []
            group_next_obs = []
            group_dones = []

            for demo in demos:
                obs, acts, next_obs, dones = demo.get_transitions()
                if len(obs) == 0:
                    continue
                group_obs.append(obs)
                group_acts.append(acts)
                group_next_obs.append(next_obs)
                group_dones.append(dones)

            if not group_obs:
                continue

            group_obs = np.concatenate(group_obs, axis=0)
            group_acts = np.concatenate(group_acts, axis=0)
            group_next_obs = np.concatenate(group_next_obs, axis=0)
            group_dones = np.concatenate(group_dones, axis=0)

            sample_size = min(per_group, len(group_obs))
            indices = np.random.choice(len(group_obs), size=sample_size, replace=False)

            obs_list.append(group_obs[indices])
            act_list.append(group_acts[indices])
            next_obs_list.append(group_next_obs[indices])
            dones_list.append(group_dones[indices])
            group_labels.extend([key] * sample_size)

        if not obs_list:
            return self.sample_batch(batch_size, include_next_obs=True)

        obs_batch = np.concatenate(obs_list, axis=0)
        acts_batch = np.concatenate(act_list, axis=0)
        next_obs_batch = np.concatenate(next_obs_list, axis=0)
        dones_batch = np.concatenate(dones_list, axis=0)

        # Top up if needed
        if len(obs_batch) < batch_size:
            extra = self.sample_batch(batch_size - len(obs_batch), include_next_obs=True)
            obs_batch = np.concatenate([obs_batch, extra['observations']], axis=0)
            acts_batch = np.concatenate([acts_batch, extra['actions']], axis=0)
            next_obs_batch = np.concatenate([next_obs_batch, extra['next_observations']], axis=0)
            dones_batch = np.concatenate([dones_batch, extra['dones']], axis=0)
            group_labels.extend(["unbalanced"] * (batch_size - len(group_labels)))

        return {
            'observations': obs_batch,
            'actions': acts_batch,
            'next_observations': next_obs_batch,
            'dones': dones_batch,
            'groups': np.array(group_labels, dtype=object),
        }

    def _get_group_key(self, demo: Demonstration, balance_by: str) -> str:
        """Resolve grouping key for balanced sampling."""
        if balance_by.startswith("environment:"):
            env_key = balance_by.split("environment:", 1)[1]
            return str(demo.environment.get(env_key, "unknown"))
        if balance_by == "environment":
            return str(demo.environment.get("scenario", "unknown"))
        return str(getattr(demo, balance_by, "unknown"))

    def filter_by_task(self, task_type: str) -> 'DemonstrationDataset':
        """Get subset of demonstrations for a specific task."""
        filtered = [d for d in self.demonstrations if d.task_type == task_type]
        return DemonstrationDataset(filtered)

    def filter_by_skill(self, min_skill: str) -> 'DemonstrationDataset':
        """Filter by minimum pilot skill level."""
        skill_levels = {'novice': 0, 'intermediate': 1, 'expert': 2, 'professional': 3}
        min_level = skill_levels.get(min_skill, 0)

        filtered = [
            d for d in self.demonstrations
            if skill_levels.get(d.pilot_skill_level, 0) >= min_level
        ]
        return DemonstrationDataset(filtered)

    def filter_by_quality(self, min_quality: float) -> 'DemonstrationDataset':
        """Filter by minimum quality score."""
        filtered = [d for d in self.demonstrations if d.quality_score >= min_quality]
        return DemonstrationDataset(filtered)

    def filter_by_aircraft(self, aircraft_type: str) -> 'DemonstrationDataset':
        """Filter by aircraft type."""
        filtered = [d for d in self.demonstrations if d.aircraft_type == aircraft_type]
        return DemonstrationDataset(filtered)

    def statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.demonstrations:
            return {'empty': True}

        durations = [d.duration_seconds for d in self.demonstrations]
        qualities = [d.quality_score for d in self.demonstrations]

        return {
            'num_demonstrations': len(self.demonstrations),
            'total_steps': self.total_steps,
            'total_transitions': self.total_transitions,
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'duration': {
                'total': sum(durations),
                'mean': np.mean(durations),
                'min': min(durations),
                'max': max(durations),
            },
            'quality': {
                'mean': np.mean(qualities),
                'min': min(qualities),
                'max': max(qualities),
            },
            'task_types': list(set(d.task_type for d in self.demonstrations)),
            'aircraft_types': list(set(d.aircraft_type for d in self.demonstrations)),
            'pilot_skill_levels': list(set(d.pilot_skill_level for d in self.demonstrations)),
            'sources': list(set(d.source for d in self.demonstrations)),
        }

    def save(self, directory: str, format: str = 'pickle'):
        """
        Save dataset to directory.

        Args:
            directory: Output directory
            format: 'pickle' or 'json'
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'num_demonstrations': len(self.demonstrations),
            'statistics': self.statistics(),
            'format': format,
        }

        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save each demonstration
        for i, demo in enumerate(self.demonstrations):
            if format == 'json':
                demo.save(str(path / f'demo_{i:04d}.json'))
            else:
                demo.save(str(path / f'demo_{i:04d}.pkl'))

        print(f"Saved {len(self.demonstrations)} demonstrations to {directory}")

    @classmethod
    def load(cls, directory: str) -> 'DemonstrationDataset':
        """Load dataset from directory."""
        path = Path(directory)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        format = metadata.get('format', 'pickle')
        ext = '.json' if format == 'json' else '.pkl'

        # Load demonstrations
        demos = []
        for demo_file in sorted(path.glob(f'demo_*{ext}')):
            demos.append(Demonstration.load(str(demo_file)))

        print(f"Loaded {len(demos)} demonstrations from {directory}")
        return cls(demos)


class DemonstrationRecorder:
    """
    Records demonstrations from live control inputs.

    Use this to capture demonstrations from:
    - Flight simulators
    - Hardware-in-the-loop setups
    - Gamepad/joystick inputs
    """

    def __init__(
        self,
        pilot_id: str = "unknown",
        aircraft_type: str = "quadcopter",
        task_type: str = "freeform",
        sample_rate_hz: float = 100.0,
    ):
        self.pilot_id = pilot_id
        self.aircraft_type = aircraft_type
        self.task_type = task_type
        self.sample_rate_hz = sample_rate_hz

        self._recording = False
        self._current_demo: Optional[Demonstration] = None
        self._start_time: float = 0.0
        self._demos: List[Demonstration] = []

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def recorded_demos(self) -> List[Demonstration]:
        return self._demos

    def start_recording(
        self,
        task_description: str = "",
        environment: Optional[Dict[str, Any]] = None,
    ):
        """Start recording a new demonstration."""
        self._current_demo = Demonstration(
            pilot_id=self.pilot_id,
            aircraft_type=self.aircraft_type,
            task_type=self.task_type,
            task_description=task_description,
            sample_rate_hz=self.sample_rate_hz,
            source="live_recording",
            environment=environment or {},
        )
        self._recording = True
        self._start_time = 0.0
        print(f"Started recording demonstration: {self._current_demo.demo_id}")

    def record_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        timestamp: Optional[float] = None,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Record a single timestep."""
        if not self._recording or self._current_demo is None:
            return

        # Calculate timestamp
        if timestamp is None:
            if self._current_demo.steps:
                timestamp = self._current_demo.steps[-1].timestamp + (1.0 / self.sample_rate_hz)
            else:
                timestamp = 0.0

        dt = 1.0 / self.sample_rate_hz
        if self._current_demo.steps:
            dt = timestamp - self._current_demo.steps[-1].timestamp

        step = DemonstrationStep(
            observation=observation,
            action=action,
            timestamp=timestamp,
            dt=dt,
            position=position,
            velocity=velocity,
            orientation=orientation,
            info=info or {},
        )

        self._current_demo.add_step(step)

    def stop_recording(self, success: bool = True, quality_score: float = 1.0) -> Demonstration:
        """Stop recording and return the demonstration."""
        if not self._recording or self._current_demo is None:
            raise RuntimeError("Not currently recording")

        self._current_demo.success = success
        self._current_demo.quality_score = quality_score

        demo = self._current_demo
        self._demos.append(demo)

        self._recording = False
        self._current_demo = None

        print(f"Stopped recording. Duration: {demo.duration_seconds:.2f}s, Steps: {demo.num_steps}")
        return demo

    def cancel_recording(self):
        """Cancel current recording without saving."""
        self._recording = False
        self._current_demo = None
        print("Recording cancelled")

    def save_all(self, directory: str):
        """Save all recorded demonstrations."""
        dataset = DemonstrationDataset(self._demos)
        dataset.save(directory)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_demonstrations(path: str) -> Union[Demonstration, DemonstrationDataset]:
    """
    Load demonstrations from file or directory.

    Args:
        path: Path to file or directory

    Returns:
        Single Demonstration or DemonstrationDataset
    """
    p = Path(path)

    if p.is_dir():
        return DemonstrationDataset.load(path)
    elif p.suffix == '.json':
        return Demonstration.load(path)
    elif p.suffix == '.pkl':
        return Demonstration.load(path)
    else:
        raise ValueError(f"Unknown file format: {p.suffix}")


def save_demonstrations(
    demos: Union[Demonstration, List[Demonstration], DemonstrationDataset],
    path: str,
):
    """
    Save demonstrations to file or directory.

    Args:
        demos: Single demo, list of demos, or dataset
        path: Output path
    """
    if isinstance(demos, Demonstration):
        demos.save(path)
    elif isinstance(demos, DemonstrationDataset):
        demos.save(path)
    elif isinstance(demos, list):
        dataset = DemonstrationDataset(demos)
        dataset.save(path)


def merge_datasets(*datasets: DemonstrationDataset) -> DemonstrationDataset:
    """Merge multiple datasets into one."""
    all_demos = []
    for ds in datasets:
        all_demos.extend(ds.demonstrations)
    return DemonstrationDataset(all_demos)


def create_synthetic_demonstration(
    env,
    policy,
    num_steps: int = 1000,
    pilot_id: str = "synthetic",
    task_type: str = "generated",
) -> Demonstration:
    """
    Create a synthetic demonstration by rolling out a policy.

    Useful for:
    - Testing the imitation learning pipeline
    - Generating "expert" demos from trained policies
    - Data augmentation

    Args:
        env: Gymnasium environment
        policy: Policy with predict(obs) method
        num_steps: Maximum steps to record
        pilot_id: ID for synthetic pilot
        task_type: Task type label

    Returns:
        Demonstration with recorded trajectory
    """
    demo = Demonstration(
        pilot_id=pilot_id,
        source="synthetic",
        task_type=task_type,
        sample_rate_hz=1.0 / env.dt if hasattr(env, 'dt') else 100.0,
    )

    obs, _ = env.reset()

    for t in range(num_steps):
        action, _ = policy.predict(obs, deterministic=True)

        step = DemonstrationStep(
            observation=obs,
            action=action,
            timestamp=t * (1.0 / demo.sample_rate_hz),
        )
        demo.add_step(step)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    return demo

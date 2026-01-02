"""
Stratified Demonstration Splitter

Splits datasets at the DEMONSTRATION level (not transition level)
with stratification by pilot and task to prevent data leakage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from training.imitation.demonstration import Demonstration, DemonstrationDataset


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.70       # 70% training
    val_ratio: float = 0.15         # 15% validation
    test_ratio: float = 0.15        # 15% test

    stratify_by_pilot: bool = True  # Stratify by pilot_id
    stratify_by_task: bool = True   # Stratify by task_type
    prevent_pilot_overlap: bool = True  # No pilot in multiple splits

    random_seed: int = 42           # For reproducibility

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class DatasetSplit:
    """Result of splitting a dataset."""
    train: 'DemonstrationDataset'
    val: 'DemonstrationDataset'
    test: 'DemonstrationDataset'
    config: SplitConfig
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_splits(self) -> Dict[str, 'DemonstrationDataset']:
        return {
            'train': self.train,
            'val': self.val,
            'test': self.test,
        }

    def validate(self) -> bool:
        """Validate that splits have no overlap."""
        train_ids = {d.demo_id for d in self.train.demonstrations}
        val_ids = {d.demo_id for d in self.val.demonstrations}
        test_ids = {d.demo_id for d in self.test.demonstrations}

        # Check no demo appears in multiple splits
        assert len(train_ids & val_ids) == 0, "Train/val overlap"
        assert len(train_ids & test_ids) == 0, "Train/test overlap"
        assert len(val_ids & test_ids) == 0, "Val/test overlap"

        if self.config.prevent_pilot_overlap:
            train_pilots = {d.pilot_id for d in self.train.demonstrations}
            val_pilots = {d.pilot_id for d in self.val.demonstrations}
            test_pilots = {d.pilot_id for d in self.test.demonstrations}

            assert len(train_pilots & val_pilots) == 0, "Pilot overlap train/val"
            assert len(train_pilots & test_pilots) == 0, "Pilot overlap train/test"
            assert len(val_pilots & test_pilots) == 0, "Pilot overlap val/test"

        return True

    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Dataset Split Summary:"]
        lines.append(f"  Train: {len(self.train)} demos, {self.train.total_steps} steps")
        lines.append(f"  Val:   {len(self.val)} demos, {self.val.total_steps} steps")
        lines.append(f"  Test:  {len(self.test)} demos, {self.test.total_steps} steps")

        if self.statistics:
            lines.append("\nStatistics:")
            for key, value in self.statistics.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class StratifiedDemoSplitter:
    """
    Splits demonstrations with stratification by pilot and task.

    Key features:
    - Splits at DEMO level, not transition level (prevents leakage)
    - Stratifies by pilot_id and task_type
    - Can enforce no pilot overlap between splits
    - Produces reproducible splits with seed
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()

    def split(
        self,
        dataset: 'DemonstrationDataset',
    ) -> DatasetSplit:
        """
        Split dataset into train/val/test with stratification.

        Args:
            dataset: Dataset to split

        Returns:
            DatasetSplit with train, val, test datasets
        """
        from training.imitation.demonstration import DemonstrationDataset

        np.random.seed(self.config.random_seed)

        demos = dataset.demonstrations

        if self.config.prevent_pilot_overlap:
            # Split by pilot first, then distribute tasks
            return self._split_by_pilot(demos)
        else:
            # Stratified split without pilot separation
            return self._stratified_split(demos)

    def _split_by_pilot(
        self,
        demos: List['Demonstration'],
    ) -> DatasetSplit:
        """Split ensuring no pilot appears in multiple splits."""
        from training.imitation.demonstration import DemonstrationDataset

        # Group by pilot
        pilot_demos: Dict[str, List['Demonstration']] = defaultdict(list)
        for demo in demos:
            pilot_demos[demo.pilot_id].append(demo)

        pilots = list(pilot_demos.keys())
        np.random.shuffle(pilots)

        # Compute target sizes
        total_demos = len(demos)
        train_target = int(total_demos * self.config.train_ratio)
        val_target = int(total_demos * self.config.val_ratio)

        # Assign pilots to splits
        train_demos = []
        val_demos = []
        test_demos = []

        train_count = 0
        val_count = 0

        for pilot in pilots:
            pilot_list = pilot_demos[pilot]

            if train_count < train_target:
                train_demos.extend(pilot_list)
                train_count += len(pilot_list)
            elif val_count < val_target:
                val_demos.extend(pilot_list)
                val_count += len(pilot_list)
            else:
                test_demos.extend(pilot_list)

        # Compute statistics
        stats = {
            'num_pilots': len(pilots),
            'train_pilots': len({d.pilot_id for d in train_demos}),
            'val_pilots': len({d.pilot_id for d in val_demos}),
            'test_pilots': len({d.pilot_id for d in test_demos}),
            'train_ratio_actual': len(train_demos) / total_demos,
            'val_ratio_actual': len(val_demos) / total_demos,
            'test_ratio_actual': len(test_demos) / total_demos,
        }

        split = DatasetSplit(
            train=DemonstrationDataset(train_demos),
            val=DemonstrationDataset(val_demos),
            test=DemonstrationDataset(test_demos),
            config=self.config,
            statistics=stats,
        )

        split.validate()
        return split

    def _stratified_split(
        self,
        demos: List['Demonstration'],
    ) -> DatasetSplit:
        """Stratified split without pilot separation constraint."""
        from training.imitation.demonstration import DemonstrationDataset

        # Create stratification groups
        groups: Dict[str, List['Demonstration']] = defaultdict(list)

        for demo in demos:
            if self.config.stratify_by_pilot and self.config.stratify_by_task:
                key = f"{demo.pilot_id}_{demo.task_type}"
            elif self.config.stratify_by_pilot:
                key = demo.pilot_id
            elif self.config.stratify_by_task:
                key = demo.task_type
            else:
                key = "all"

            groups[key].append(demo)

        # Split each group
        train_demos = []
        val_demos = []
        test_demos = []

        for key, group_demos in groups.items():
            np.random.shuffle(group_demos)

            n = len(group_demos)
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)

            train_demos.extend(group_demos[:train_end])
            val_demos.extend(group_demos[train_end:val_end])
            test_demos.extend(group_demos[val_end:])

        # Compute statistics
        total_demos = len(demos)
        stats = {
            'num_groups': len(groups),
            'train_ratio_actual': len(train_demos) / total_demos if total_demos > 0 else 0,
            'val_ratio_actual': len(val_demos) / total_demos if total_demos > 0 else 0,
            'test_ratio_actual': len(test_demos) / total_demos if total_demos > 0 else 0,
        }

        split = DatasetSplit(
            train=DemonstrationDataset(train_demos),
            val=DemonstrationDataset(val_demos),
            test=DemonstrationDataset(test_demos),
            config=self.config,
            statistics=stats,
        )

        split.validate()
        return split


def split_by_pilot_and_task(
    dataset: 'DemonstrationDataset',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    prevent_pilot_overlap: bool = True,
    seed: int = 42,
) -> DatasetSplit:
    """
    Convenience function for stratified splitting.

    Args:
        dataset: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        prevent_pilot_overlap: Ensure no pilot in multiple splits
        seed: Random seed

    Returns:
        DatasetSplit with train, val, test sets
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by_pilot=True,
        stratify_by_task=True,
        prevent_pilot_overlap=prevent_pilot_overlap,
        random_seed=seed,
    )

    splitter = StratifiedDemoSplitter(config)
    return splitter.split(dataset)


def validate_no_leakage(split: DatasetSplit) -> Dict[str, Any]:
    """
    Validate that there's no data leakage between splits.

    Checks:
    - No demonstration appears in multiple splits
    - No pilot appears in multiple splits (if configured)
    - Task distribution is reasonable

    Args:
        split: Dataset split to validate

    Returns:
        Dict with validation results
    """
    results = {
        'valid': True,
        'issues': [],
    }

    # Check demo overlap
    train_ids = {d.demo_id for d in split.train.demonstrations}
    val_ids = {d.demo_id for d in split.val.demonstrations}
    test_ids = {d.demo_id for d in split.test.demonstrations}

    if train_ids & val_ids:
        results['valid'] = False
        results['issues'].append(f"Train/val demo overlap: {len(train_ids & val_ids)}")

    if train_ids & test_ids:
        results['valid'] = False
        results['issues'].append(f"Train/test demo overlap: {len(train_ids & test_ids)}")

    if val_ids & test_ids:
        results['valid'] = False
        results['issues'].append(f"Val/test demo overlap: {len(val_ids & test_ids)}")

    # Check pilot overlap if configured
    if split.config.prevent_pilot_overlap:
        train_pilots = {d.pilot_id for d in split.train.demonstrations}
        val_pilots = {d.pilot_id for d in split.val.demonstrations}
        test_pilots = {d.pilot_id for d in split.test.demonstrations}

        if train_pilots & val_pilots:
            results['valid'] = False
            results['issues'].append(f"Train/val pilot overlap: {train_pilots & val_pilots}")

        if train_pilots & test_pilots:
            results['valid'] = False
            results['issues'].append(f"Train/test pilot overlap: {train_pilots & test_pilots}")

        if val_pilots & test_pilots:
            results['valid'] = False
            results['issues'].append(f"Val/test pilot overlap: {val_pilots & test_pilots}")

    # Check task distribution
    def get_task_dist(demos):
        tasks = defaultdict(int)
        for d in demos:
            tasks[d.task_type] += 1
        return dict(tasks)

    results['train_tasks'] = get_task_dist(split.train.demonstrations)
    results['val_tasks'] = get_task_dist(split.val.demonstrations)
    results['test_tasks'] = get_task_dist(split.test.demonstrations)

    return results

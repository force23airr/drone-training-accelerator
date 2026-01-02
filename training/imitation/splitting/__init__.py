"""
Splitting Module for Demonstration Data

Demo-level splitting with stratification.
"""

from training.imitation.splitting.stratified_splitter import (
    SplitConfig,
    DatasetSplit,
    StratifiedDemoSplitter,
    split_by_pilot_and_task,
    validate_no_leakage,
)

__all__ = [
    "SplitConfig",
    "DatasetSplit",
    "StratifiedDemoSplitter",
    "split_by_pilot_and_task",
    "validate_no_leakage",
]

from __future__ import annotations

"""Lightweight data helpers for fluent experiments."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .datasets import Dataset, DatasetSplit


class SupervisedDataBuilder:
    """Deferred split builder for supervised input/target arrays."""

    def __init__(self, inputs: Sequence[float] | np.ndarray, targets: Sequence[float] | np.ndarray, metadata: Mapping[str, Any] | None = None):
        self.inputs = np.asarray(inputs, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.metadata = dict(metadata or {})

    def split(
        self,
        *,
        washout: int = 0,
        train: int,
        test: int,
        indices: DatasetSplit | Mapping[str, Sequence[int]] | None = None,
    ) -> Dataset:
        """Create a Dataset with the requested split."""

        if indices is not None:
            return Dataset.from_arrays(self.inputs, self.targets, split=indices, metadata=self.metadata)
        return Dataset.from_arrays(self.inputs, self.targets, washout=washout, train=train, test=test, metadata=self.metadata)


class TimeSeriesDataBuilder:
    """Deferred split builder for scalar forecasting series."""

    def __init__(self, series: Sequence[float] | np.ndarray, target_horizon: int = 1, metadata: Mapping[str, Any] | None = None):
        self.series = np.asarray(series, dtype=float)
        self.target_horizon = int(target_horizon)
        self.metadata = dict(metadata or {})

    def split(self, *, washout: int = 0, train: int, test: int) -> Dataset:
        """Create a forecasting Dataset with contiguous split ranges."""

        return Dataset.timeseries(
            self.series,
            target_horizon=self.target_horizon,
            washout=washout,
            train=train,
            test=test,
            metadata=self.metadata,
        )


def arrays(
    inputs: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SupervisedDataBuilder:
    """Start a supervised array dataset builder."""

    return SupervisedDataBuilder(inputs, targets, metadata=metadata)


def timeseries(
    series: Sequence[float] | np.ndarray,
    *,
    target_horizon: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> TimeSeriesDataBuilder:
    """Start a scalar time-series forecasting dataset builder."""

    return TimeSeriesDataBuilder(series, target_horizon=target_horizon, metadata=metadata)


def npz(
    path: str | Path,
    *,
    inputs_key: str = "inputs",
    targets_key: str = "targets",
    metadata: Mapping[str, Any] | None = None,
) -> SupervisedDataBuilder:
    """Start a supervised dataset builder from an NPZ file."""

    with np.load(Path(path), allow_pickle=False) as data:
        return SupervisedDataBuilder(data[inputs_key], data[targets_key], metadata=metadata)

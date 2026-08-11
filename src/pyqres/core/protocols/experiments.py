"""Structural contracts for datasets, readouts, experiments, and tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .reservoirs import StreamingReservoirProtocol
from .specifications import ReservoirSpecProtocol
from .types import FeatureMatrix, IndexSequence, MetricCallable, TargetArray


@runtime_checkable
class DatasetSplitProtocol(Protocol):
    """Index split contract used by supervised experiments."""

    washout: np.ndarray
    train: np.ndarray
    test: np.ndarray

    def validate(self, n_samples: int) -> None: ...

    def to_dict(self) -> dict[str, list[int]]: ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """Dataset contract consumed by task-agnostic experiments."""

    inputs: np.ndarray
    targets: np.ndarray
    split: DatasetSplitProtocol
    metadata: Mapping[str, Any] | None

    def validate_features(self, features: np.ndarray) -> None: ...

    def save_npz(self, path: str | Path) -> Path: ...


@runtime_checkable
class SupervisedDataBuilderProtocol(Protocol):
    """Deferred dataset builder for supervised arrays."""

    inputs: np.ndarray
    targets: np.ndarray
    metadata: Mapping[str, Any]

    def split(
        self,
        *,
        washout: int = 0,
        train: int,
        test: int,
        indices: DatasetSplitProtocol | Mapping[str, IndexSequence] | None = None,
    ) -> DatasetProtocol: ...


@runtime_checkable
class TimeSeriesDataBuilderProtocol(Protocol):
    """Deferred dataset builder for scalar forecasting series."""

    series: np.ndarray
    target_horizon: int
    metadata: Mapping[str, Any]

    def split(self, *, washout: int = 0, train: int, test: int) -> DatasetProtocol: ...


@runtime_checkable
class ReadoutProtocol(Protocol):
    """Supervised readout contract used by Experiment."""

    def fit(self, features: FeatureMatrix, targets: TargetArray) -> ReadoutProtocol: ...

    def predict(self, features: FeatureMatrix) -> np.ndarray: ...


@runtime_checkable
class ExperimentResultProtocol(Protocol):
    """Persistable result produced by experiment runners."""

    metrics: Mapping[str, Mapping[str, float]]
    features: np.ndarray
    predictions: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def save(self, outdir: str | Path) -> Path: ...


@runtime_checkable
class ExperimentProtocol(Protocol):
    """Runnable supervised reservoir experiment."""

    reservoir: Any
    dataset: DatasetProtocol
    readout: ReadoutProtocol | None
    metrics: Mapping[str, MetricCallable] | list[str] | tuple[str, ...] | None
    metadata: Mapping[str, Any] | None

    def run(self) -> ExperimentResultProtocol: ...


@runtime_checkable
class SweepResultProtocol(Protocol):
    """Persistable collection of sweep experiment results."""

    parameter: str

    def rows(self) -> list[dict[str, Any]]: ...

    def save(self, outdir: str | Path) -> Path: ...


@runtime_checkable
class SweepProtocol(Protocol):
    """One-parameter reservoir sweep contract."""

    base: ReservoirSpecProtocol
    parameter: str
    values: Iterable[float]

    def specs(self) -> list[ReservoirSpecProtocol]: ...

    def run(
        self,
        dataset: DatasetProtocol,
        *,
        backend: str = "exact",
        readout_factory: Callable[[], ReadoutProtocol] | None = None,
        metrics: Mapping[str, MetricCallable]
        | list[str]
        | tuple[str, ...]
        | None = None,
    ) -> SweepResultProtocol: ...


@runtime_checkable
class TaskDatasetFactoryProtocol(Protocol):
    """Factory contract used by pyqres-tasks adapters."""

    def __call__(self, cfg: Any) -> DatasetProtocol: ...


@runtime_checkable
class TaskRunnerProtocol(Protocol):
    """Task-runner contract for external task packages that stream directly."""

    res: StreamingReservoirProtocol
    cfg: Any

    def run(self) -> Mapping[str, float] | Mapping[int, Mapping[str, float]]: ...

from __future__ import annotations

"""Generic experiment and sweep orchestration."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .builders import compile_reservoir, transform
from .datasets import Dataset
from .metrics import Metric, resolve_metrics
from .readout import ReadoutModel, Ridge
from .specs import ReservoirSpec


@dataclass
class ExperimentResult:
    """Result object returned by Experiment.run."""

    metrics: dict[str, dict[str, float]]
    features: np.ndarray
    predictions: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Run a reservoir on a generic supervised dataset and fit a readout."""

    reservoir: Any
    dataset: Dataset
    readout: ReadoutModel | None = None
    metrics: Mapping[str, Metric] | list[str] | tuple[str, ...] | None = None

    def run(self) -> ExperimentResult:
        """Collect features, fit the readout, and score train/test splits."""

        features = transform(self.reservoir, self.dataset.inputs)
        self.dataset.validate_features(features)
        readout = self.readout if self.readout is not None else Ridge()
        metric_fns = resolve_metrics(self.metrics)

        train_idx = self.dataset.split.train
        test_idx = self.dataset.split.test
        readout.fit(features[train_idx], self.dataset.targets[train_idx])
        train_pred = readout.predict(features[train_idx])
        test_pred = readout.predict(features[test_idx])

        score_table = {
            "train": {name: fn(self.dataset.targets[train_idx], train_pred) for name, fn in metric_fns.items()},
            "test": {name: fn(self.dataset.targets[test_idx], test_pred) for name, fn in metric_fns.items()},
        }
        return ExperimentResult(
            metrics=score_table,
            features=features,
            predictions={"train": np.asarray(train_pred), "test": np.asarray(test_pred)},
            metadata={"dataset": dict(self.dataset.metadata or {})},
        )


@dataclass(frozen=True)
class Sweep:
    """One-parameter sweep over ReservoirSpec values."""

    base: ReservoirSpec
    parameter: str
    values: Iterable[float]

    def specs(self) -> list[ReservoirSpec]:
        """Materialize all specs in the sweep."""

        out: list[ReservoirSpec] = []
        for value in self.values:
            if hasattr(self.base, self.parameter):
                out.append(replace(self.base, **{self.parameter: float(value)}))
            else:
                kwargs = dict(self.base.model_kwargs)
                kwargs[self.parameter] = float(value)
                out.append(replace(self.base, model_kwargs=kwargs))
        return out

    def run(
        self,
        dataset: Dataset,
        *,
        backend: str = "exact",
        readout_factory: Callable[[], ReadoutModel] | None = None,
        metrics: Mapping[str, Metric] | list[str] | tuple[str, ...] | None = None,
    ) -> list[tuple[ReservoirSpec, ExperimentResult]]:
        """Run an Experiment for every spec in the sweep."""

        results: list[tuple[ReservoirSpec, ExperimentResult]] = []
        for spec in self.specs():
            reservoir = compile_reservoir(spec, backend=backend)
            readout = readout_factory() if readout_factory is not None else Ridge()
            results.append((spec, Experiment(reservoir, dataset, readout=readout, metrics=metrics).run()))
        return results

from __future__ import annotations

"""Generic experiment and sweep orchestration."""

from dataclasses import dataclass, field, replace
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from pyqres.core.builders import compile_reservoir, transform
from pyqres.core.specs import ReservoirSpec
from pyqres.experiments.datasets import Dataset
from pyqres.experiments.metrics import Metric, resolve_metrics
from pyqres.experiments.readout import ReadoutModel, Ridge


@dataclass
class ExperimentResult:
    """Result object returned by Experiment.run."""

    metrics: dict[str, dict[str, float]]
    features: np.ndarray
    predictions: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, outdir: str | Path) -> Path:
        """Persist metrics, arrays, and metadata to a run directory."""

        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2, sort_keys=True)
        with (path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(_to_builtin(self.metadata), handle, indent=2, sort_keys=True)
        arrays = {"features": np.asarray(self.features, dtype=float)}
        for name, values in self.predictions.items():
            arrays[f"predictions_{name}"] = np.asarray(values)
        np.savez_compressed(path / "arrays.npz", **arrays)
        return path


@dataclass
class Experiment:
    """Run a reservoir on a generic supervised dataset and fit a readout."""

    reservoir: Any
    dataset: Dataset
    readout: ReadoutModel | None = None
    metrics: Mapping[str, Metric] | list[str] | tuple[str, ...] | None = None
    metadata: Mapping[str, Any] | None = None

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
        full_pred = readout.predict(features)

        score_table = {
            "train": {name: fn(self.dataset.targets[train_idx], train_pred) for name, fn in metric_fns.items()},
            "test": {name: fn(self.dataset.targets[test_idx], test_pred) for name, fn in metric_fns.items()},
        }
        return ExperimentResult(
            metrics=score_table,
            features=features,
            predictions={
                "full": np.asarray(full_pred),
                "train": np.asarray(train_pred),
                "test": np.asarray(test_pred),
            },
            metadata={
                "dataset": dict(self.dataset.metadata or {}),
                "experiment": dict(self.metadata or {}),
                "split": self.dataset.split.to_dict(),
            },
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
    ) -> "SweepResult":
        """Run an Experiment for every spec in the sweep."""

        results: list[tuple[ReservoirSpec, ExperimentResult]] = []
        for spec in self.specs():
            reservoir = compile_reservoir(spec, backend=backend)
            readout = readout_factory() if readout_factory is not None else Ridge()
            results.append((spec, Experiment(reservoir, dataset, readout=readout, metrics=metrics).run()))
        return SweepResult(parameter=self.parameter, results=results)


@dataclass
class SweepResult:
    """Results from a one-parameter ReservoirSpec sweep."""

    parameter: str
    results: list[tuple[ReservoirSpec, ExperimentResult]]

    def rows(self) -> list[dict[str, Any]]:
        """Flatten sweep metrics into one row per spec."""

        out: list[dict[str, Any]] = []
        for spec, result in self.results:
            row: dict[str, Any] = {
                "parameter": self.parameter,
                "value": getattr(spec, self.parameter, spec.model_kwargs.get(self.parameter, np.nan)),
                "family": spec.family,
                "n_system": spec.n_system,
                "n_ancilla": spec.n_ancilla,
                "n_memory": spec.n_memory,
                "n_readout": spec.n_readout,
                "tau": spec.tau,
                "input_scale": spec.input_scale,
                "seed": spec.seed,
            }
            for split, metrics in result.metrics.items():
                for name, value in metrics.items():
                    row[f"{split}_{name}"] = float(value)
            out.append(row)
        return out

    def save(self, outdir: str | Path) -> Path:
        """Persist sweep summary plus each run's artifacts."""

        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        if rows:
            fieldnames = list(rows[0])
            with (path / "sweep_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        for idx, (spec, result) in enumerate(self.results):
            run_dir = path / f"run_{idx:04d}"
            result.metadata.setdefault("reservoir_spec", spec.to_dict())
            result.save(run_dir)
        return path


def _to_builtin(obj: Any) -> Any:
    """Convert common numerical containers into JSON-safe values."""

    if isinstance(obj, dict):
        return {str(key): _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

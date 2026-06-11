from __future__ import annotations

"""Generic experiment CLI helpers.

Task-specific benchmark CLIs now live in the separate ``pyqres-tasks`` package.
This module is intentionally small so core ``pyqres`` stays focused on generic
reservoir construction, simulation, and analysis workflows.
"""

from pathlib import Path
from typing import Any

import numpy as np

from pyqres import Dataset, Experiment, ReservoirSpec, Ridge, compile_reservoir


def run_array_experiment(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    reservoir_spec: ReservoirSpec,
    backend: str = "exact",
    washout: int = 0,
    train: int,
    test: int,
    l2: float = 1e-6,
) -> dict[str, Any]:
    """Run a generic supervised array experiment and return metrics."""

    dataset = Dataset.from_arrays(inputs, targets, washout=washout, train=train, test=test)
    reservoir = compile_reservoir(reservoir_spec, backend=backend)
    result = Experiment(reservoir=reservoir, dataset=dataset, readout=Ridge(l2=l2)).run()
    return {"metrics": result.metrics, "feature_shape": tuple(result.features.shape)}


def save_features(path: str | Path, features: np.ndarray) -> Path:
    """Save a generic feature matrix for external task tooling."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(features, dtype=float))
    return out

from __future__ import annotations

"""Generic pyqres experiment CLI."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from pyqres.core.builders import compile_reservoir
from pyqres.core.specs import ReservoirSpec
from pyqres.experiments.datasets import Dataset
from pyqres.experiments.readout import Ridge
from pyqres.experiments.runner import Experiment
from pyqres.experiments.common import run_experiment_from_config


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a generic pyqres experiment YAML file.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    cfg = OmegaConf.load(config_path)
    result = run_experiment_from_config(
        cfg,
        output_dir_override=args.output_dir,
        base_dir=config_path.parent,
    )
    for split, metrics in result.metrics.items():
        for name, value in metrics.items():
            print(f"{split}_{name}: {value:.6g}")


if __name__ == "__main__":
    main()

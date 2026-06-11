from __future__ import annotations

"""Reusable helpers for compact pyqres experiment scripts.

The functions here handle config-to-dataclass mapping, model construction,
observable-readout selection, output directory resolution, and raw array saving.
They intentionally avoid plotting and sweep logic so scripts can stay focused
on reservoir and dataset specification.
"""

from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any, TypeVar

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pyqres.core import ConfigMapping
from pyqres.core.builders import compile_reservoir
from pyqres.core.specs import ReservoirSpec
from pyqres.dim import (
    IsingReservoirModel,
    IsingReservoirParameters,
    MemoryObservableStreamingReservoir,
    RandomPauliReservoirModel,
    RandomPauliReservoirParameters,
    SYKReservoirModel,
    SYKReservoirParameters,
)
from pyqres.experiments.datasets import Dataset
from pyqres.experiments.readout import Ridge
from pyqres.experiments.runner import Experiment


T = TypeVar("T")


def to_builtin(obj: Any) -> Any:
    """Convert NumPy and OmegaConf values into JSON-safe Python values."""

    if isinstance(obj, DictConfig):
        return to_builtin(OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, dict):
        return {str(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dataclass_from_config(dataclass_type: type[T], cfg: DictConfig | ConfigMapping | None) -> T:
    """Instantiate a dataclass using only fields present on that dataclass."""

    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type!r} is not a dataclass type")
    raw = {} if cfg is None else OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    if not isinstance(raw, dict):
        raise TypeError("Config section must resolve to a mapping")

    names = {field.name for field in fields(dataclass_type)}
    extra = sorted(set(raw) - names)
    if extra:
        raise ValueError(f"Unexpected fields for {dataclass_type.__name__}: {extra}")

    return dataclass_type(**{key: raw[key] for key in raw if key in names})


def resolve_output_dir(
    cfg: DictConfig,
    output_dir_override: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve paths.output_dir and optional timestamping from a config."""

    base = Path(output_dir_override) if output_dir_override else Path(str(cfg.paths.output_dir))
    if not base.is_absolute() and base_dir is not None:
        base = Path(base_dir) / base
    if bool(cfg.paths.get("timestamped", True)):
        base = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return base


def build_model(cfg: DictConfig) -> Any:
    """Build a supported reservoir model from a compact model config."""

    model_type = str(cfg.type).lower()
    registry = {
        "ising": (IsingReservoirParameters, IsingReservoirModel),
        "random_pauli": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "randompauli": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "syk": (SYKReservoirParameters, SYKReservoirModel),
    }
    if model_type in registry:
        parameter_cls, model_cls = registry[model_type]
        params = dataclass_from_config(parameter_cls, cfg.get("params", {}))
        return model_cls(params)
    raise ValueError(f"Unsupported model.type '{cfg.type}'")


def select_observable_specs(model: Any, cfg: DictConfig) -> list[str]:
    """Resolve observable preset, custom specs, and optional truncation count."""

    specs = model.default_memory_observable_specs(
        preset=str(cfg.preset),
        custom_specs=list(cfg.get("custom", [])),
    )
    count = cfg.get("count")
    if count is None:
        return specs
    count_int = int(count)
    if count_int < 1 or count_int > len(specs):
        raise ValueError(f"Observable count {count_int} outside valid range [1, {len(specs)}]")
    return specs[:count_int]


def build_memory_observable_reservoir(model: Any, readout_cfg: DictConfig) -> tuple[MemoryObservableStreamingReservoir, list[str]]:
    """Build a memory-observable streaming reservoir from readout config."""

    specs = select_observable_specs(model, readout_cfg.observables)
    observables = [model.parse_memory_observable(spec) for spec in specs]
    reservoir = MemoryObservableStreamingReservoir(
        model=model,
        observables=observables,
        include_bias=bool(readout_cfg.include_bias),
        init_state=str(readout_cfg.init_state),
    )
    return reservoir, specs


def save_raw_dataset(outdir: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any], cfg: DictConfig) -> None:
    """Save raw arrays, metadata, and the resolved config for an experiment."""

    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "raw_dataset.npz", **arrays)
    OmegaConf.save(cfg, outdir / "resolved_config.yaml", resolve=True)
    with (outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(metadata), f, indent=2)


def _mapping(cfg: DictConfig | ConfigMapping) -> dict[str, Any]:
    """Resolve OmegaConf or plain mappings into a mutable dict."""

    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def dataset_from_config(cfg: DictConfig | ConfigMapping, base_dir: str | Path | None = None) -> Dataset:
    """Build a generic Dataset from a config section.

    Supported sources:
    - arrays: inline inputs/targets
    - npz: inputs/targets loaded from a compressed NPZ file
    - timeseries: inline scalar series with a target horizon
    """

    raw = _mapping(cfg)
    source = str(raw.get("source", "arrays")).lower()
    split_cfg = raw.get("split")
    metadata = raw.get("metadata", {})

    def split_kwargs() -> dict[str, Any]:
        if split_cfg is None:
            return {}
        if "indices" in split_cfg:
            return {"split": split_cfg["indices"]}
        return {
            "washout": int(split_cfg.get("washout", 0)),
            "train": int(split_cfg["train"]),
            "test": int(split_cfg["test"]),
        }

    if source == "arrays":
        return Dataset.from_arrays(raw["inputs"], raw["targets"], metadata=metadata, **split_kwargs())
    if source == "timeseries":
        return Dataset.timeseries(
            raw["series"],
            target_horizon=int(raw.get("target_horizon", 1)),
            metadata=metadata,
            **split_kwargs(),
        )
    if source == "npz":
        path = Path(str(raw["path"]))
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        return Dataset.from_npz(
            path,
            inputs_key=str(raw.get("inputs_key", "inputs")),
            targets_key=str(raw.get("targets_key", "targets")),
            metadata=metadata,
            **split_kwargs(),
        )
    raise ValueError(f"Unsupported dataset source '{source}'")


def reservoir_spec_from_config(cfg: DictConfig | ConfigMapping) -> ReservoirSpec:
    """Build a ReservoirSpec from a config section."""

    raw = _mapping(cfg)
    return ReservoirSpec.from_mapping(raw)


def readout_from_config(cfg: DictConfig | ConfigMapping | None) -> Any:
    """Build a readout model from config."""

    raw = {} if cfg is None else _mapping(cfg)
    kind = str(raw.get("kind", "ridge")).lower()
    if kind == "ridge":
        return Ridge(l2=float(raw.get("l2", 1e-6)), include_bias=bool(raw.get("include_bias", False)))
    raise ValueError(f"Unsupported readout kind '{kind}'")


def run_experiment_from_config(
    cfg: DictConfig | ConfigMapping,
    *,
    output_dir_override: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Any:
    """Run a generic pyqres Experiment from a config tree and save artifacts."""

    raw = _mapping(cfg)
    dataset = dataset_from_config(raw["dataset"], base_dir=base_dir)
    spec = reservoir_spec_from_config(raw["reservoir"])
    backend = str(raw.get("backend", "exact"))
    readout = readout_from_config(raw.get("readout"))
    metrics = raw.get("metrics")
    reservoir = compile_reservoir(spec, backend=backend)
    result = Experiment(
        reservoir=reservoir,
        dataset=dataset,
        readout=readout,
        metrics=metrics,
        metadata={"backend": backend, "reservoir_spec": spec.to_dict()},
    ).run()

    paths = raw.get("paths", {})
    if output_dir_override is not None:
        outdir = Path(output_dir_override)
    else:
        outdir = Path(str(paths.get("output_dir", "outputs/pyqres_experiment")))
        if not outdir.is_absolute() and base_dir is not None:
            outdir = Path(base_dir) / outdir
        if bool(paths.get("timestamped", True)):
            outdir = outdir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result.save(outdir)
    with (outdir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(raw), handle, indent=2, sort_keys=True)
    return result

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

from pyqres.dim import IsingReservoirModel, IsingReservoirParameters, MemoryObservableStreamingReservoir


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


def dataclass_from_config(dataclass_type: type[T], cfg: DictConfig | dict[str, Any] | None) -> T:
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
    if model_type == "ising":
        params = dataclass_from_config(IsingReservoirParameters, cfg.get("params", {}))
        return IsingReservoirModel(params)
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

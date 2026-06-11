from __future__ import annotations

"""Generic dataset containers for task-agnostic reservoir experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """Index split used by an experiment readout."""

    washout: np.ndarray
    train: np.ndarray
    test: np.ndarray

    @classmethod
    def contiguous(cls, washout: int, train: int, test: int) -> "DatasetSplit":
        """Build contiguous washout/train/test index ranges."""

        washout = int(washout)
        train = int(train)
        test = int(test)
        if min(washout, train, test) < 0:
            raise ValueError("split lengths must be non-negative")
        train_start = washout
        train_end = train_start + train
        test_end = train_end + test
        return cls(
            washout=np.arange(0, train_start, dtype=int),
            train=np.arange(train_start, train_end, dtype=int),
            test=np.arange(train_end, test_end, dtype=int),
        )

    def validate(self, n_samples: int) -> None:
        """Validate all split indices against a sample count."""

        n_samples = int(n_samples)
        for name, values in (("washout", self.washout), ("train", self.train), ("test", self.test)):
            arr = np.asarray(values, dtype=int)
            if arr.ndim != 1:
                raise ValueError(f"{name} indices must be one-dimensional")
            if arr.size and (arr.min() < 0 or arr.max() >= n_samples):
                raise ValueError(f"{name} indices are outside [0, {n_samples})")

    def to_dict(self) -> dict[str, list[int]]:
        """Return split indices as plain lists."""

        return {
            "washout": np.asarray(self.washout, dtype=int).tolist(),
            "train": np.asarray(self.train, dtype=int).tolist(),
            "test": np.asarray(self.test, dtype=int).tolist(),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Sequence[int]]) -> "DatasetSplit":
        """Build a split from explicit index arrays."""

        return cls(
            washout=np.asarray(data.get("washout", []), dtype=int),
            train=np.asarray(data["train"], dtype=int),
            test=np.asarray(data["test"], dtype=int),
        )


@dataclass(frozen=True)
class Dataset:
    """Inputs, targets, and split metadata for a generic supervised task."""

    inputs: np.ndarray
    targets: np.ndarray
    split: DatasetSplit
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_arrays(
        cls,
        inputs: Sequence[float] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        split: DatasetSplit | Mapping[str, Sequence[int]] | None = None,
        *,
        washout: int | None = None,
        train: int | None = None,
        test: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Dataset":
        """Create a dataset from arbitrary input/target arrays."""

        x = np.asarray(inputs, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"inputs and targets must share the first dimension, got {x.shape[0]} and {y.shape[0]}")

        if split is None:
            if washout is None or train is None or test is None:
                raise ValueError("provide split or washout/train/test lengths")
            split_obj = DatasetSplit.contiguous(washout=washout, train=train, test=test)
        elif isinstance(split, DatasetSplit):
            split_obj = split
        else:
            split_obj = DatasetSplit(
                washout=np.asarray(split.get("washout", []), dtype=int),
                train=np.asarray(split["train"], dtype=int),
                test=np.asarray(split["test"], dtype=int),
            )
        split_obj.validate(x.shape[0])
        return cls(inputs=x, targets=y, split=split_obj, metadata=dict(metadata or {}))

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        inputs_key: str = "inputs",
        targets_key: str = "targets",
        split: DatasetSplit | Mapping[str, Sequence[int]] | None = None,
        washout: int | None = None,
        train: int | None = None,
        test: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Dataset":
        """Load inputs/targets from an NPZ file."""

        with np.load(Path(path), allow_pickle=False) as data:
            inputs = np.asarray(data[inputs_key], dtype=float)
            targets = np.asarray(data[targets_key], dtype=float)
            if split is None and {"washout_indices", "train_indices", "test_indices"}.issubset(data.files):
                split = {
                    "washout": np.asarray(data["washout_indices"], dtype=int),
                    "train": np.asarray(data["train_indices"], dtype=int),
                    "test": np.asarray(data["test_indices"], dtype=int),
                }
        return cls.from_arrays(
            inputs,
            targets,
            split=split,
            washout=washout,
            train=train,
            test=test,
            metadata=metadata,
        )

    @classmethod
    def timeseries(
        cls,
        series: Sequence[float] | np.ndarray,
        *,
        target_horizon: int = 1,
        washout: int,
        train: int,
        test: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Dataset":
        """Create one-step or multi-step forecasting data from a scalar series."""

        horizon = int(target_horizon)
        if horizon < 1:
            raise ValueError("target_horizon must be >= 1")
        values = np.asarray(series, dtype=float)
        if values.shape[0] <= horizon:
            raise ValueError("series is shorter than target_horizon")
        return cls.from_arrays(
            values[:-horizon],
            values[horizon:],
            washout=washout,
            train=train,
            test=test,
            metadata={"target_horizon": horizon, **dict(metadata or {})},
        )

    def validate_features(self, features: np.ndarray) -> None:
        """Validate reservoir features before readout fitting."""

        arr = np.asarray(features)
        if arr.ndim != 2:
            raise ValueError(f"features must be a 2D matrix, got shape {arr.shape}")
        if arr.shape[0] != self.inputs.shape[0]:
            raise ValueError(f"feature rows must match inputs, got {arr.shape[0]} and {self.inputs.shape[0]}")
        if not np.isfinite(arr).all():
            raise FloatingPointError("non-finite reservoir features")

    def save_npz(self, path: str | Path) -> Path:
        """Save inputs, targets, and split indices to a compressed NPZ file."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            inputs=np.asarray(self.inputs, dtype=float),
            targets=np.asarray(self.targets, dtype=float),
            washout_indices=np.asarray(self.split.washout, dtype=int),
            train_indices=np.asarray(self.split.train, dtype=int),
            test_indices=np.asarray(self.split.test, dtype=int),
        )
        return out

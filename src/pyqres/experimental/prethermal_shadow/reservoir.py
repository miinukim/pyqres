from __future__ import annotations

"""Reservoir object that emits prethermal classical-shadow features."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .circuits import build_streaming_circuit
from .config import PrethermalShadowConfig, ResolvedPrethermalShadowConfig, validate_and_resolve_config
from .shadows import (
    PauliLabel,
    counts_to_outcomes,
    estimate_shadow_features,
    feature_label_strings,
    generate_pauli_labels,
    group_basis_schedules,
    sample_basis_schedules,
)

try:
    from qiskit import transpile
except Exception:  # pragma: no cover - optional dependency
    transpile = None  # type: ignore

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - optional dependency
    AerSimulator = None  # type: ignore


BasisSampler = Callable[[int, int, int, Sequence[str], int], np.ndarray]
ScheduleExecutor = Callable[[Sequence[float], np.ndarray, ResolvedPrethermalShadowConfig, int], Mapping[str, int]]
FeatureBuilder = Callable[[np.ndarray, np.ndarray, Sequence[PauliLabel], bool], np.ndarray]


class PrethermalShadowReservoir:
    """Qiskit/Aer prethermal shadow reservoir.

    The object follows pyqres' duck-typed reservoir contract by exposing
    ``run_stream``, ``run``, and ``transform``.
    """

    def __init__(
        self,
        cfg: PrethermalShadowConfig,
        *,
        basis_sampler: BasisSampler | None = None,
        schedule_executor: ScheduleExecutor | None = None,
        feature_builder: FeatureBuilder | None = None,
    ):
        self.cfg = validate_and_resolve_config(cfg)
        self.basis_sampler = basis_sampler or sample_basis_schedules
        self.schedule_executor = schedule_executor or self._execute_schedule
        self.feature_builder = feature_builder or self._build_features
        self.pauli_labels = generate_pauli_labels(self.cfg.base.n_readout, self.cfg.base.shadow.pauli_k)
        self.feature_labels = feature_label_strings(
            self.cfg.base.n_readout,
            self.cfg.base.shadow.pauli_k,
            include_bias=self.cfg.base.shadow.include_bias,
        )
        self.last_basis_schedules: np.ndarray | None = None

    def run(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Run a stream and return a feature matrix."""

        return self.run_stream(inputs)

    def transform(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Scikit-style alias."""

        return self.run_stream(inputs)

    def run_stream(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Sample shadow schedules, execute grouped circuits, and estimate features."""

        values = np.asarray(inputs, dtype=float).reshape(-1)
        if values.size == 0:
            return np.empty((0, len(self.feature_labels)), dtype=float)
        shadow = self.cfg.base.shadow
        schedules = self.basis_sampler(
            int(shadow.shots),
            int(values.shape[0]),
            int(self.cfg.base.n_readout),
            tuple(shadow.bases),
            int(shadow.seed),
        )
        schedules = np.asarray(schedules, dtype="U1")
        expected = (int(shadow.shots), int(values.shape[0]), int(self.cfg.base.n_readout))
        if schedules.shape != expected:
            raise ValueError(f"basis_sampler returned shape {schedules.shape}, expected {expected}.")
        self.last_basis_schedules = schedules.copy()

        schedule_blocks: list[np.ndarray] = []
        outcome_blocks: list[np.ndarray] = []
        for schedule, group_size, _ in group_basis_schedules(schedules):
            counts = self.schedule_executor(values, schedule, self.cfg, group_size)
            outcomes = counts_to_outcomes(
                counts,
                shots=group_size,
                n_steps=int(values.shape[0]),
                n_readout=int(self.cfg.base.n_readout),
            )
            schedule_blocks.append(np.repeat(schedule[None, :, :], group_size, axis=0))
            outcome_blocks.append(outcomes)

        merged_schedules = np.concatenate(schedule_blocks, axis=0)
        merged_outcomes = np.concatenate(outcome_blocks, axis=0)
        features = self.feature_builder(
            merged_schedules,
            merged_outcomes,
            self.pauli_labels,
            bool(shadow.include_bias),
        )
        return np.asarray(features, dtype=float)

    @staticmethod
    def _build_features(schedules: np.ndarray, outcomes: np.ndarray, labels: Sequence[PauliLabel], include_bias: bool) -> np.ndarray:
        """Default feature-builder adapter."""

        return estimate_shadow_features(schedules, outcomes, labels, include_bias=include_bias)

    def build_streaming_circuit(self, inputs: Sequence[float] | np.ndarray, basis_schedule: np.ndarray) -> Any:
        """Build one Qiskit circuit for a fixed basis schedule."""

        return build_streaming_circuit(inputs, basis_schedule, self.cfg)

    def _execute_schedule(self, inputs: Sequence[float] | np.ndarray, basis_schedule: np.ndarray, cfg: ResolvedPrethermalShadowConfig, shots: int) -> Mapping[str, int]:
        """Execute one grouped basis schedule with Aer."""

        if AerSimulator is None:
            raise ImportError("qiskit-aer is required for PrethermalShadowReservoir execution.")
        if transpile is None:
            raise ImportError("qiskit is required for PrethermalShadowReservoir execution.")
        qc = build_streaming_circuit(inputs, basis_schedule, cfg)
        backend_options: dict[str, Any] = {
            "method": cfg.base.simulator_method,
            "seed_simulator": int(cfg.base.seed_simulator),
            **dict(cfg.base.aer_options),
        }
        if cfg.base.simulator_device != "automatic":
            backend_options["device"] = cfg.base.simulator_device
        backend = AerSimulator(**backend_options)
        executable = transpile(
            qc,
            backend=backend,
            optimization_level=int(cfg.base.transpile_optimization_level),
        )
        result = backend.run(executable, shots=int(shots)).result()
        return result.get_counts(0)

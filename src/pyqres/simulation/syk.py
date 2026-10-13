"""Streaming SYK reservoir with exact or classical-shadow readout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyqres.dim.model import SYKReservoirModel, SYKReservoirParameters
from pyqres.prethermal_shadow.dynamics import (
    density_zero,
    partial_trace_memory_first,
    project_density,
)
from pyqres.prethermal_shadow.shadows import (
    exact_readout_expectations,
    generate_pauli_labels,
    partial_shadow_feature_names,
    sample_partial_shadow_features,
)


@dataclass(frozen=True)
class SYKReservoirConfig:
    """Configuration for a number-conserving fermionic SYK reservoir."""

    n_memory: int = 6
    n_readout: int = 3
    tau: float = 1.0
    j4_strength: float = 1.0
    kappa2_strength: float = 0.0
    normalize_syk4_by_spectral_norm: bool = True
    normalize_syk2_by_spectral_norm: bool = True
    input_qubit: int = 0
    input_bias: float = 0.0
    input_scale: float = 1.0
    input_clip_eps: float = 1.0e-9
    pauli_k: int = 2
    include_bias: bool = True
    seed: int = 31
    exact_expectations: bool = True
    shots: int = 1024
    shadow_seed: int = 37
    return_raw_shots: bool = False

    def __post_init__(self) -> None:
        if self.n_memory < 1 or self.n_readout < 1:
            raise ValueError("n_memory and n_readout must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.j4_strength < 0.0 or self.kappa2_strength < 0.0:
            raise ValueError("SYK coupling strengths must be non-negative")
        if self.j4_strength == 0.0 and self.kappa2_strength == 0.0:
            raise ValueError("at least one SYK coupling strength must be positive")
        if not 0 <= self.input_qubit < self.n_readout:
            raise ValueError("input_qubit must select a readout qubit")
        if not 0.0 <= self.input_clip_eps < 0.5:
            raise ValueError("input_clip_eps must lie in [0, 0.5)")
        if self.pauli_k < 1 or self.pauli_k > self.n_readout:
            raise ValueError("pauli_k must lie between 1 and n_readout")
        if self.shots < 1:
            raise ValueError("shots must be positive")


class SYKReservoir:
    """Partial-reset streaming adapter for the fermionic SYK QRC model.

    The input replaces one readout fermionic mode with
    ``sqrt(1-u)|0> + sqrt(u)|1>``. Remaining readout modes are reset to zero.
    The full register then evolves under a fixed number-conserving SYK4+SYK2
    Hamiltonian before readout measurement and reset.
    """

    def __init__(self, config: SYKReservoirConfig):
        self.config = config
        self.n_memory = int(config.n_memory)
        self.n_readout = int(config.n_readout)
        self.n_qubits = self.n_memory + self.n_readout
        self.dim_memory = 2**self.n_memory
        self.dim_readout = 2**self.n_readout
        params = SYKReservoirParameters(
            n_memory=config.n_memory,
            n_readout=config.n_readout,
            tau=config.tau,
            j4_strength=config.j4_strength,
            kappa2_strength=config.kappa2_strength,
            seed=config.seed,
            input_qubit=config.input_qubit,
            input_bias=config.input_bias,
            input_scale=config.input_scale,
            input_clip_eps=config.input_clip_eps,
            normalize_syk4_by_spectral_norm=config.normalize_syk4_by_spectral_norm,
            normalize_syk2_by_spectral_norm=config.normalize_syk2_by_spectral_norm,
            reset_to_zero_state=True,
        )
        self.model = SYKReservoirModel(params)
        self.fixed_unitary = self.model.unitary(0.0)
        self._labels = generate_pauli_labels(config.n_readout, config.pauli_k)
        self._feature_names = partial_shadow_feature_names(
            config.n_readout,
            config.pauli_k,
            include_bias=config.include_bias,
        )
        self.last_readout_state: np.ndarray | None = None
        self.last_raw_shots: dict[str, np.ndarray] | None = None
        self.reset_state()

    def _input_readout_state(self, value: float) -> np.ndarray:
        encoded = self.model.input_state_vector(float(value)).reshape(2, 1)
        left = self.config.input_qubit
        right = self.n_readout - left - 1
        state = encoded
        if left:
            zero = np.zeros((2**left, 1), dtype=complex)
            zero[0, 0] = 1.0
            state = np.kron(zero, state)
        if right:
            zero = np.zeros((2**right, 1), dtype=complex)
            zero[0, 0] = 1.0
            state = np.kron(state, zero)
        return state @ state.conj().T

    def reset_state(self) -> None:
        self.rho_memory = density_zero(self.n_memory)
        self._shadow_rng = np.random.default_rng(self.config.shadow_seed)
        self.last_readout_state = None
        self.last_raw_shots = None

    def reset(self) -> None:
        self.reset_state()

    def step(self, value: float) -> np.ndarray:
        rho_initial = np.kron(
            self.rho_memory,
            self._input_readout_state(float(value)),
        )
        rho_pre = project_density(
            self.fixed_unitary @ rho_initial @ self.fixed_unitary.conj().T
        )
        self.last_readout_state = project_density(
            partial_trace_memory_first(
                rho_pre, self.n_memory, self.n_readout, "readout"
            )
        )
        self.rho_memory = project_density(
            partial_trace_memory_first(
                rho_pre, self.n_memory, self.n_readout, "memory"
            )
        )
        if self.config.exact_expectations:
            self.last_raw_shots = None
            return exact_readout_expectations(
                self.last_readout_state,
                self._labels,
                include_bias=self.config.include_bias,
            )
        result = sample_partial_shadow_features(
            self.last_readout_state,
            self._labels,
            shots=self.config.shots,
            include_bias=self.config.include_bias,
            rng=self._shadow_rng,
            return_raw=self.config.return_raw_shots,
        )
        if isinstance(result, tuple):
            features, raw = result
            self.last_raw_shots = raw
            return np.asarray(features, dtype=float)
        self.last_raw_shots = None
        return np.asarray(result, dtype=float)

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        self.reset_state()
        values = [self.step(float(value)) for value in inputs]
        if not values:
            return np.empty((0, len(self._feature_names)), dtype=float)
        return np.vstack(values)

    def get_feature_names(self) -> list[str]:
        return list(self._feature_names)


__all__ = ["SYKReservoir", "SYKReservoirConfig"]

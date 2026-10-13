"""Brickwork random-circuit reservoir with partial readout reset."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyqres.prethermal_shadow.dynamics import (
    PAULI,
    density_zero,
    kron_all,
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
class ScramblingCircuitConfig:
    """Configuration for a fixed brickwork random-circuit reservoir."""

    n_memory: int = 6
    n_readout: int = 3
    depth: int = 6
    two_qubit_angle: float = np.pi / 2.0
    input_beta: float = -1.0
    input_bias: float = 0.0
    input_axis: str = "X"
    pauli_k: int = 2
    include_bias: bool = True
    seed: int = 23
    exact_expectations: bool = True
    shots: int = 1024
    shadow_seed: int = 37
    return_raw_shots: bool = False

    def __post_init__(self) -> None:
        if self.n_memory < 1 or self.n_readout < 1:
            raise ValueError("n_memory and n_readout must be positive")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if not np.isfinite(self.two_qubit_angle) or self.two_qubit_angle < 0.0:
            raise ValueError("two_qubit_angle must be finite and non-negative")
        if self.input_axis.upper() not in {"X", "Y", "Z"}:
            raise ValueError("input_axis must be X, Y, or Z")
        if self.pauli_k < 1 or self.pauli_k > self.n_readout:
            raise ValueError("pauli_k must lie between 1 and n_readout")
        if self.shots < 1:
            raise ValueError("shots must be positive")


class ScramblingCircuitReservoir:
    """Streaming reservoir based on a fixed random brickwork circuit.

    Each circuit layer applies independent fixed random single-qubit SU(2)
    rotations followed by two-qubit Pauli rotations on alternating even and odd
    nearest-neighbor bonds. The first readout qubit receives the scalar input;
    the readout register is traced and reset after every step.
    """

    def __init__(self, config: ScramblingCircuitConfig):
        self.config = config
        self.n_memory = int(config.n_memory)
        self.n_readout = int(config.n_readout)
        self.n_qubits = self.n_memory + self.n_readout
        self.dim_memory = 2**self.n_memory
        self.dim_readout = 2**self.n_readout
        self.dim_total = 2**self.n_qubits
        self.input_qubit = self.n_memory
        self.fixed_unitary = self._build_fixed_unitary()
        self._labels = generate_pauli_labels(config.n_readout, config.pauli_k)
        self._feature_names = partial_shadow_feature_names(
            config.n_readout,
            config.pauli_k,
            include_bias=config.include_bias,
        )
        self.last_readout_state: np.ndarray | None = None
        self.last_raw_shots: dict[str, np.ndarray] | None = None
        self.reset_state()

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        return np.diag(
            [np.exp(-0.5j * angle), np.exp(0.5j * angle)]
        ).astype(complex)

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        cosine = np.cos(0.5 * angle)
        sine = np.sin(0.5 * angle)
        return np.asarray([[cosine, -sine], [sine, cosine]], dtype=complex)

    def _random_su2(self, rng: np.random.Generator) -> np.ndarray:
        alpha = 2.0 * np.pi * rng.random()
        gamma = 2.0 * np.pi * rng.random()
        beta = 2.0 * np.arccos(np.sqrt(rng.random()))
        return self._rotation_z(alpha) @ self._rotation_y(beta) @ self._rotation_z(
            gamma
        )

    def _two_qubit_rotation(self, rng: np.random.Generator) -> np.ndarray:
        axes = rng.choice(np.asarray(["X", "Y", "Z"]), size=2)
        generator = np.kron(PAULI[str(axes[0])], PAULI[str(axes[1])])
        angle = self.config.two_qubit_angle * rng.uniform(0.75, 1.25)
        return (
            np.cos(0.5 * angle) * np.eye(4, dtype=complex)
            - 1j * np.sin(0.5 * angle) * generator
        )

    def _brickwork_layer(
        self, rng: np.random.Generator, start: int
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        site = 0
        while site < self.n_qubits:
            if site >= start and (site - start) % 2 == 0 and site + 1 < self.n_qubits:
                blocks.append(self._two_qubit_rotation(rng))
                site += 2
            else:
                blocks.append(PAULI["I"])
                site += 1
        return kron_all(blocks)

    def _build_fixed_unitary(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        unitary = np.eye(self.dim_total, dtype=complex)
        for layer in range(self.config.depth):
            local_layer = kron_all(
                [self._random_su2(rng) for _ in range(self.n_qubits)]
            )
            entangling_layer = self._brickwork_layer(rng, layer % 2)
            unitary = entangling_layer @ local_layer @ unitary
        if not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(self.dim_total, dtype=complex),
            atol=1e-10,
        ):
            raise FloatingPointError("scrambling circuit is not unitary")
        return unitary

    def _input_unitary(self, value: float) -> np.ndarray:
        scale = self.config.input_beta * (float(value) + self.config.input_bias)
        local = (
            np.cos(scale) * PAULI["I"]
            - 1j * np.sin(scale) * PAULI[self.config.input_axis.upper()]
        )
        factors = [PAULI["I"] for _ in range(self.n_qubits)]
        factors[self.input_qubit] = local
        return kron_all(factors)

    def reset_state(self) -> None:
        self.rho_memory = density_zero(self.n_memory)
        self._shadow_rng = np.random.default_rng(self.config.shadow_seed)
        self.last_readout_state = None
        self.last_raw_shots = None

    def reset(self) -> None:
        self.reset_state()

    def step(self, value: float) -> np.ndarray:
        rho_initial = np.kron(self.rho_memory, density_zero(self.n_readout))
        step_unitary = self.fixed_unitary @ self._input_unitary(float(value))
        rho_pre = project_density(
            step_unitary @ rho_initial @ step_unitary.conj().T
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


__all__ = ["ScramblingCircuitConfig", "ScramblingCircuitReservoir"]

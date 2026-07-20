from __future__ import annotations

"""Configuration for global-Floquet partial-shadow reservoirs."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GlobalFloquetConfig:
    """All-qubit fast-driven Floquet dynamics.

    By default, memory occupies ``0..n_memory-1`` and readout occupies the
    remaining qubits. ``readout_qubits`` can select and order arbitrary physical
    readout indices; memory then becomes the ascending complement.
    """

    n_qubits: int
    n_memory: int
    n_readout: int
    omega: float
    n_cycles_per_step: int
    drive_type: str = "two_step_square"
    drive_amplitude: float = 0.5
    drive_axis: str = "x"
    h_scale: float = 1.0
    jz_scale: float = 0.3
    jxy_scale: float = 0.08
    break_scale: float = 0.0
    break_axis: str = "y"
    random_h: bool = True
    random_jz: bool = True
    random_jxy: bool = True
    random_drive: bool = True
    random_drive_sign: bool | None = None
    random_break: bool = True
    h_range: tuple[float, float] = (0.8, 1.2)
    jz_range: tuple[float, float] = (0.5, 1.5)
    jxy_range: tuple[float, float] = (0.5, 1.5)
    drive_range: tuple[float, float] = (0.5, 1.5)
    break_range: tuple[float, float] = (-1.0, 1.0)
    parameter_draw_order: str = "grouped"
    seed: int = 0
    topology: str = "chain"
    include_mr_couplings: bool = True
    readout_qubits: Sequence[int] | None = None


@dataclass(frozen=True)
class InputEncodingConfig:
    """Input-dependent local rotations."""

    input_qubits: str | Sequence[int] = "memory"
    axis: str = "y"
    beta: float | Sequence[float] = 0.1
    bias: float = 0.0
    random_beta: bool = True
    seed: int = 1
    operator: str | None = None
    normalize_operator: bool = True


@dataclass(frozen=True)
class PartialShadowReadoutConfig:
    """Partial local Pauli-shadow feature extraction.

    ``feature_scope='readout'`` is the physical shadow readout path. ``memory``
    and ``full`` are exact-expectation diagnostic scopes.
    """

    pauli_k: int = 2
    feature_scope: str = "readout"
    shots: int = 1024
    measurement_type: str = "projective"
    weak_strength: float = 1.0
    basis_randomization: str = "local_pauli"
    include_bias: bool = True
    seed: int = 2
    exact_expectations: bool = False
    return_shadow_estimates: bool = True
    return_raw_shots: bool = False


@dataclass(frozen=True)
class ReadoutResetConfig:
    """Trace/reset behavior for the measured readout subsystem."""

    reset_after_measurement: bool = True
    reset_state: str = "zero"
    trace_readout_after_step: bool = True


__all__ = [
    "GlobalFloquetConfig",
    "InputEncodingConfig",
    "PartialShadowReadoutConfig",
    "ReadoutResetConfig",
]

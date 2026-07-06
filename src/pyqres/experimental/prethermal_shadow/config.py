from __future__ import annotations

"""Configuration dataclasses for the prethermal shadow reservoir."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np


Axis = Literal["x", "y", "z"]
FeatureQubits = Literal["readout"]
SimulatorDevice = Literal["automatic", "CPU", "GPU"]


@dataclass(frozen=True)
class PrethermalFloquetConfig:
    """Memory-only prethermal Floquet dynamics."""

    n_floquet: int = 4
    tau: float = 0.2
    h: Sequence[float] | None = None
    jz: Sequence[float] | None = None
    jxy: Sequence[float] | None = None
    x_break: Sequence[float] | None = None
    periodic: bool = False
    seed: int = 1234


@dataclass(frozen=True)
class InputWriteConfig:
    """Input-dependent weak write rotations on memory qubits."""

    axis: Axis = "y"
    beta: Sequence[float] | float = 0.1
    bias: float = 0.0


@dataclass(frozen=True)
class TransducerConfig:
    """Memory-to-readout RZZ transducer settings."""

    tau_c: float = 0.05
    g: Any | None = None
    seed: int = 1234


@dataclass(frozen=True)
class ShadowReadoutConfig:
    """Classical-shadow readout settings."""

    pauli_k: int = 2
    shots: int = 2048
    include_bias: bool = True
    bases: tuple[str, ...] = ("X", "Y", "Z")
    seed: int = 1234
    feature_qubits: FeatureQubits = "readout"


@dataclass(frozen=True)
class PrethermalShadowConfig:
    """Full prethermal shadow reservoir configuration."""

    n_memory: int
    n_readout: int
    floquet: PrethermalFloquetConfig
    input_write: InputWriteConfig = field(default_factory=InputWriteConfig)
    transducer: TransducerConfig | None = None
    shadow: ShadowReadoutConfig = field(default_factory=ShadowReadoutConfig)
    simulator_method: str = "density_matrix"
    simulator_device: SimulatorDevice = "automatic"
    aer_options: dict[str, Any] = field(default_factory=dict)
    seed_simulator: int = 123
    transpile_optimization_level: int = 1


@dataclass(frozen=True)
class ResolvedPrethermalShadowConfig:
    """Validated config with all arrays materialized."""

    base: PrethermalShadowConfig
    h: np.ndarray
    jz: np.ndarray
    jxy: np.ndarray
    x_break: np.ndarray
    beta: np.ndarray
    g: np.ndarray | None
    edges: tuple[tuple[int, int], ...]


def floquet_edges(n_memory: int, periodic: bool) -> tuple[tuple[int, int], ...]:
    """Return open-chain or periodic memory edges."""

    n_memory = int(n_memory)
    edges = [(i, i + 1) for i in range(n_memory - 1)]
    if periodic and n_memory > 2:
        edges.append((n_memory - 1, 0))
    return tuple(edges)


def _array_or_random(
    value: Sequence[float] | None,
    *,
    length: int,
    rng: np.random.Generator,
    name: str,
) -> np.ndarray:
    if value is None:
        return rng.uniform(-1.0, 1.0, size=int(length)).astype(float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (int(length),):
        raise ValueError(f"{name} must have length {length}, got shape {arr.shape}.")
    return arr


def _array_or_zeros(value: Sequence[float] | None, *, length: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(int(length), dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (int(length),):
        raise ValueError(f"{name} must have length {length}, got shape {arr.shape}.")
    return arr


def _beta_array(value: Sequence[float] | float, n_memory: int) -> np.ndarray:
    if isinstance(value, (int, float, np.floating)):
        return np.full(int(n_memory), float(value), dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (int(n_memory),):
        raise ValueError(f"input_write.beta must be scalar or length {n_memory}, got shape {arr.shape}.")
    return arr


def _g_array(value: Any | None, *, n_readout: int, n_memory: int, seed: int) -> np.ndarray:
    if value is None:
        rng = np.random.default_rng(int(seed))
        return rng.uniform(-1.0, 1.0, size=(int(n_readout), int(n_memory))).astype(float)
    arr = np.asarray(value, dtype=float)
    expected = (int(n_readout), int(n_memory))
    if arr.shape != expected:
        raise ValueError(f"transducer.g must have shape {expected}, got {arr.shape}.")
    return arr


def validate_and_resolve_config(cfg: PrethermalShadowConfig) -> ResolvedPrethermalShadowConfig:
    """Validate a config and materialize seed-controlled defaults."""

    if not isinstance(cfg.floquet, PrethermalFloquetConfig):
        floquet_raw = dict(cfg.floquet)  # type: ignore[arg-type]
        if "n_memory" in floquet_raw:
            cfg = replace(cfg, n_memory=int(floquet_raw.pop("n_memory")))
        cfg = replace(cfg, floquet=PrethermalFloquetConfig(**floquet_raw))
    if not isinstance(cfg.input_write, InputWriteConfig):
        cfg = replace(cfg, input_write=InputWriteConfig(**dict(cfg.input_write)))  # type: ignore[arg-type]
    if cfg.transducer is not None and not isinstance(cfg.transducer, TransducerConfig):
        cfg = replace(cfg, transducer=TransducerConfig(**dict(cfg.transducer)))  # type: ignore[arg-type]
    if not isinstance(cfg.shadow, ShadowReadoutConfig):
        cfg = replace(cfg, shadow=ShadowReadoutConfig(**dict(cfg.shadow)))  # type: ignore[arg-type]

    n_memory = int(cfg.n_memory)
    n_readout = int(cfg.n_readout)
    n_floquet = int(cfg.floquet.n_floquet)
    shots = int(cfg.shadow.shots)
    pauli_k = int(cfg.shadow.pauli_k)
    if n_memory <= 0:
        raise ValueError("n_memory must be positive.")
    if n_readout <= 0:
        raise ValueError("n_readout must be positive.")
    if n_floquet <= 0:
        raise ValueError("floquet.n_floquet must be positive.")
    if shots <= 0:
        raise ValueError("shadow.shots must be positive.")
    if pauli_k < 1 or pauli_k > n_readout:
        raise ValueError(f"shadow.pauli_k must lie in [1, {n_readout}].")
    if cfg.input_write.axis not in {"x", "y", "z"}:
        raise ValueError(f"unsupported input axis {cfg.input_write.axis!r}.")
    bases = tuple(str(item).upper() for item in cfg.shadow.bases)
    if not bases or any(item not in {"X", "Y", "Z"} for item in bases):
        raise ValueError("shadow.bases must contain only X, Y, and Z.")
    if cfg.shadow.feature_qubits != "readout":
        raise ValueError("only feature_qubits='readout' is supported.")
    if cfg.simulator_device not in {"automatic", "CPU", "GPU"}:
        raise ValueError("simulator_device must be automatic, CPU, or GPU.")

    normalized = replace(
        cfg,
        n_memory=n_memory,
        n_readout=n_readout,
        floquet=replace(cfg.floquet, n_floquet=n_floquet),
        shadow=replace(cfg.shadow, shots=shots, pauli_k=pauli_k, bases=bases),
        aer_options=dict(cfg.aer_options or {}),
    )
    edges = floquet_edges(n_memory, bool(normalized.floquet.periodic))
    rng = np.random.default_rng(int(normalized.floquet.seed))
    h = _array_or_random(normalized.floquet.h, length=n_memory, rng=rng, name="floquet.h")
    jz = _array_or_random(normalized.floquet.jz, length=len(edges), rng=rng, name="floquet.jz")
    jxy = _array_or_random(normalized.floquet.jxy, length=len(edges), rng=rng, name="floquet.jxy")
    x_break = _array_or_zeros(normalized.floquet.x_break, length=n_memory, name="floquet.x_break")
    beta = _beta_array(normalized.input_write.beta, n_memory)
    g = None
    if normalized.transducer is not None:
        g = _g_array(
            normalized.transducer.g,
            n_readout=n_readout,
            n_memory=n_memory,
            seed=int(normalized.transducer.seed),
        )
    return ResolvedPrethermalShadowConfig(
        base=normalized,
        h=h,
        jz=jz,
        jxy=jxy,
        x_break=x_break,
        beta=beta,
        g=g,
        edges=edges,
    )

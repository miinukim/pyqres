from __future__ import annotations

"""Configuration dataclasses for the prethermal shadow reservoir."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Sequence

import numpy as np


Axis = Literal["x", "y", "z"]
FeatureQubits = Literal["readout"]
FloquetMode = Literal["effective_static", "fast_drive"]
DrivePattern = Literal["global", "random"]
SimulatorDevice = Literal["automatic", "CPU", "GPU"]


@dataclass(frozen=True)
class FastDriveConfig:
    """Explicit high-frequency memory-only drive settings."""

    enabled: bool = True
    omega: float = 20.0
    n_cycles_per_input: int = 8
    drive_amplitude: float = 1.0
    drive_axis: Axis = "x"
    drive_pattern: DrivePattern = "random"
    drive_seed: Optional[int] = None
    normalize_drive: bool = True


@dataclass(frozen=True)
class PrethermalFloquetConfig:
    """Memory-only prethermal Floquet dynamics."""

    mode: FloquetMode = "fast_drive"
    fast_drive: FastDriveConfig = field(default_factory=FastDriveConfig)
    h_min: float = 0.8
    h_max: float = 1.2
    jz_min: float = 0.2
    jz_max: float = 0.5
    jxy_min: float = 0.04
    jxy_max: float = 0.12
    x_break_scale: float = 0.005
    n_floquet: int = 4
    tau: float = 0.2
    h: Sequence[float] | None = None
    jz: Sequence[float] | None = None
    jxy: Sequence[float] | None = None
    x_break: Sequence[float] | None = None
    periodic: bool = False
    seed: Optional[int] = 1234


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
    drive: np.ndarray
    edges: tuple[tuple[int, int], ...]
    fast_drive_period: float | None
    reservoir_dt: float


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


def _array_or_uniform_range(
    value: Sequence[float] | None,
    *,
    length: int,
    rng: np.random.Generator,
    low: float,
    high: float,
    name: str,
) -> np.ndarray:
    if value is None:
        return rng.uniform(float(low), float(high), size=int(length)).astype(float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (int(length),):
        raise ValueError(f"{name} must have length {length}, got shape {arr.shape}.")
    return arr


def _array_or_breaking(
    value: Sequence[float] | None,
    *,
    length: int,
    rng: np.random.Generator,
    scale: float,
    name: str,
) -> np.ndarray:
    if value is None:
        scale = float(scale)
        if scale == 0.0:
            return np.zeros(int(length), dtype=float)
        return rng.uniform(-scale, scale, size=int(length)).astype(float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (int(length),):
        raise ValueError(f"{name} must have length {length}, got shape {arr.shape}.")
    return arr


def _drive_array(cfg: PrethermalShadowConfig, n_memory: int) -> np.ndarray:
    fd = cfg.floquet.fast_drive
    if not fd.enabled or cfg.floquet.mode != "fast_drive":
        return np.zeros(int(n_memory), dtype=float)
    if fd.omega <= 0.0:
        raise ValueError("floquet.fast_drive.omega must be positive.")
    if fd.n_cycles_per_input <= 0:
        raise ValueError("floquet.fast_drive.n_cycles_per_input must be positive.")
    if fd.drive_axis not in {"x", "y", "z"}:
        raise ValueError(f"unsupported drive axis {fd.drive_axis!r}.")
    if fd.drive_pattern == "global":
        coeffs = np.ones(int(n_memory), dtype=float)
    elif fd.drive_pattern == "random":
        seed = cfg.floquet.seed if fd.drive_seed is None else fd.drive_seed
        rng = np.random.default_rng(seed)
        coeffs = rng.uniform(-1.0, 1.0, size=int(n_memory)).astype(float)
        if fd.normalize_drive:
            rms = float(np.sqrt(np.mean(coeffs**2)))
            if rms > 1e-15:
                coeffs = coeffs / rms
    else:
        raise ValueError(f"unsupported drive pattern {fd.drive_pattern!r}.")
    return float(fd.drive_amplitude) * coeffs


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
        if isinstance(floquet_raw.get("fast_drive"), dict):
            floquet_raw["fast_drive"] = FastDriveConfig(**floquet_raw["fast_drive"])
        cfg = replace(cfg, floquet=PrethermalFloquetConfig(**floquet_raw))
    elif isinstance(cfg.floquet.fast_drive, dict):
        cfg = replace(cfg, floquet=replace(cfg.floquet, fast_drive=FastDriveConfig(**cfg.floquet.fast_drive)))
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
    if cfg.floquet.mode not in {"effective_static", "fast_drive"}:
        raise ValueError("floquet.mode must be 'effective_static' or 'fast_drive'.")
    if cfg.floquet.fast_drive.omega <= 0.0:
        raise ValueError("floquet.fast_drive.omega must be positive.")
    if cfg.floquet.fast_drive.n_cycles_per_input <= 0:
        raise ValueError("floquet.fast_drive.n_cycles_per_input must be positive.")
    if cfg.floquet.fast_drive.drive_axis not in {"x", "y", "z"}:
        raise ValueError("floquet.fast_drive.drive_axis must be x, y, or z.")
    if cfg.floquet.fast_drive.drive_pattern not in {"global", "random"}:
        raise ValueError("floquet.fast_drive.drive_pattern must be global or random.")
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
    rng = np.random.default_rng(normalized.floquet.seed)
    if normalized.floquet.mode == "effective_static":
        h = _array_or_random(normalized.floquet.h, length=n_memory, rng=rng, name="floquet.h")
        jz = _array_or_random(normalized.floquet.jz, length=len(edges), rng=rng, name="floquet.jz")
        jxy = _array_or_random(normalized.floquet.jxy, length=len(edges), rng=rng, name="floquet.jxy")
        x_break = _array_or_breaking(normalized.floquet.x_break, length=n_memory, rng=rng, scale=0.0, name="floquet.x_break")
        fast_drive_period = None
        reservoir_dt = float(normalized.floquet.n_floquet) * float(normalized.floquet.tau)
    else:
        h = _array_or_uniform_range(
            normalized.floquet.h,
            length=n_memory,
            rng=rng,
            low=normalized.floquet.h_min,
            high=normalized.floquet.h_max,
            name="floquet.h",
        )
        jz = _array_or_uniform_range(
            normalized.floquet.jz,
            length=len(edges),
            rng=rng,
            low=normalized.floquet.jz_min,
            high=normalized.floquet.jz_max,
            name="floquet.jz",
        )
        jxy = _array_or_uniform_range(
            normalized.floquet.jxy,
            length=len(edges),
            rng=rng,
            low=normalized.floquet.jxy_min,
            high=normalized.floquet.jxy_max,
            name="floquet.jxy",
        )
        x_break = _array_or_breaking(
            normalized.floquet.x_break,
            length=n_memory,
            rng=rng,
            scale=normalized.floquet.x_break_scale,
            name="floquet.x_break",
        )
        fast_drive_period = 2.0 * np.pi / float(normalized.floquet.fast_drive.omega)
        reservoir_dt = int(normalized.floquet.fast_drive.n_cycles_per_input) * fast_drive_period
    beta = _beta_array(normalized.input_write.beta, n_memory)
    drive = _drive_array(normalized, n_memory)
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
        drive=drive,
        edges=edges,
        fast_drive_period=fast_drive_period,
        reservoir_dt=float(reservoir_dt),
    )

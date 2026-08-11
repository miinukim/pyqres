"""Qubit layout, topology, and randomized Floquet parameters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import GlobalFloquetConfig, InputEncodingConfig


def resolve_input_qubits(
    cfg: InputEncodingConfig, floquet: GlobalFloquetConfig
) -> tuple[int, ...]:
    """Resolve symbolic input targets to physical qubit indices."""

    spec = cfg.input_qubits
    if isinstance(spec, str):
        key = spec.lower()
        if key == "memory":
            memory_qubits, _ = resolve_qubit_partition(floquet)
            return memory_qubits
        if key == "all":
            return tuple(range(int(floquet.n_qubits)))
        raise ValueError(
            "input_qubits must be 'memory', 'all', or a sequence of indices."
        )
    qubits = tuple(int(qubit) for qubit in spec)
    if not qubits:
        raise ValueError("input_qubits sequence must be non-empty.")
    if len(set(qubits)) != len(qubits):
        raise ValueError("input_qubits must not contain duplicates.")
    if any(qubit < 0 or qubit >= int(floquet.n_qubits) for qubit in qubits):
        raise ValueError("input_qubits contains an out-of-range qubit.")
    return qubits


def input_beta_array(cfg: InputEncodingConfig, n_targets: int) -> np.ndarray:
    """Resolve deterministic or seeded per-target input strengths."""

    if isinstance(cfg.beta, (int, float, np.floating)):
        base = np.full(int(n_targets), float(cfg.beta), dtype=float)
    else:
        base = np.asarray(cfg.beta, dtype=float).reshape(-1)
        if base.shape != (int(n_targets),):
            raise ValueError(f"beta must be scalar or length {n_targets}.")
    if not cfg.random_beta:
        return base
    rng = np.random.default_rng(int(cfg.seed))
    return base * rng.uniform(0.5, 1.5, size=int(n_targets))


def resolve_qubit_partition(
    cfg: GlobalFloquetConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return logical memory and readout qubits in their subsystem order."""

    n_qubits = int(cfg.n_qubits)
    if cfg.readout_qubits is None:
        readout = tuple(range(int(cfg.n_memory), n_qubits))
    else:
        readout = tuple(int(qubit) for qubit in cfg.readout_qubits)
    if len(readout) != int(cfg.n_readout):
        raise ValueError("readout_qubits must contain exactly n_readout indices.")
    if len(set(readout)) != len(readout):
        raise ValueError("readout_qubits must not contain duplicates.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in readout):
        raise ValueError("readout_qubits contains an out-of-range qubit.")
    readout_set = set(readout)
    memory = tuple(qubit for qubit in range(n_qubits) if qubit not in readout_set)
    if len(memory) != int(cfg.n_memory):
        raise ValueError("readout_qubits is inconsistent with n_memory and n_qubits.")
    return memory, readout


def chain_edges(
    n_qubits: int,
    n_memory: int,
    include_mr_couplings: bool = True,
    readout_qubits: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    readout = set(
        range(int(n_memory), int(n_qubits))
        if readout_qubits is None
        else (int(qubit) for qubit in readout_qubits)
    )
    edges = []
    for i in range(int(n_qubits) - 1):
        crosses_mr = (i in readout) != (i + 1 in readout)
        if not include_mr_couplings and crosses_mr:
            continue
        edges.append((i, i + 1))
    return tuple(edges)


def topology_edges(
    n_qubits: int,
    n_memory: int,
    topology: str,
    include_mr_couplings: bool = True,
    readout_qubits: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    key = str(topology).lower()
    n = int(n_qubits)
    n_mem = int(n_memory)
    readout = set(
        range(n_mem, n)
        if readout_qubits is None
        else (int(qubit) for qubit in readout_qubits)
    )
    if key == "chain":
        return chain_edges(
            n,
            n_mem,
            include_mr_couplings=include_mr_couplings,
            readout_qubits=tuple(readout),
        )
    if key == "all_to_all":
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                crosses_mr = (i in readout) != (j in readout)
                if include_mr_couplings or not crosses_mr:
                    edges.append((i, j))
        return tuple(edges)
    raise ValueError("topology must be chain or all_to_all.")


def _uniform_or_ones(
    rng: np.random.Generator, randomize: bool, bounds: tuple[float, float], size: int
) -> np.ndarray:
    if not randomize:
        return np.ones(int(size), dtype=float)
    low, high = (float(bounds[0]), float(bounds[1]))
    if low > high:
        raise ValueError(f"invalid random coefficient range {bounds!r}")
    return rng.uniform(low, high, size=int(size)).astype(float)


def _uniform_scalar_or_one(
    rng: np.random.Generator, randomize: bool, bounds: tuple[float, float]
) -> float:
    return float(_uniform_or_ones(rng, randomize, bounds, 1)[0])


def validate_floquet_config(cfg: GlobalFloquetConfig) -> None:
    if int(cfg.n_qubits) <= 0:
        raise ValueError("n_qubits must be positive.")
    if int(cfg.n_memory) <= 0:
        raise ValueError("n_memory must be positive.")
    if int(cfg.n_readout) <= 0:
        raise ValueError("n_readout must be positive.")
    if int(cfg.n_memory) + int(cfg.n_readout) != int(cfg.n_qubits):
        raise ValueError("n_qubits must equal n_memory + n_readout.")
    resolve_qubit_partition(cfg)
    if float(cfg.omega) <= 0.0:
        raise ValueError("omega must be positive.")
    if int(cfg.n_cycles_per_step) <= 0:
        raise ValueError("n_cycles_per_step must be positive.")
    if cfg.drive_type != "two_step_square":
        raise ValueError("only drive_type='two_step_square' is supported.")
    if str(cfg.drive_axis).lower() not in {"x", "y", "z"}:
        raise ValueError("drive_axis must be x, y, or z.")
    if str(cfg.break_axis).lower() not in {"x", "y", "z"}:
        raise ValueError("break_axis must be x, y, or z.")
    if cfg.topology not in {"chain", "all_to_all"}:
        raise ValueError("topology must be chain or all_to_all.")
    if cfg.parameter_draw_order not in {"grouped", "operator_test"}:
        raise ValueError("parameter_draw_order must be grouped or operator_test.")


def generate_hamiltonian_parameters(
    cfg: GlobalFloquetConfig,
) -> dict[str, np.ndarray | tuple[tuple[int, int], ...]]:
    validate_floquet_config(cfg)
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.n_qubits)
    _, readout_qubits = resolve_qubit_partition(cfg)
    edges = topology_edges(
        n,
        int(cfg.n_memory),
        str(cfg.topology),
        bool(cfg.include_mr_couplings),
        readout_qubits=readout_qubits,
    )
    random_drive_sign = bool(
        cfg.random_drive if cfg.random_drive_sign is None else cfg.random_drive_sign
    )
    h = _uniform_or_ones(rng, bool(cfg.random_h), cfg.h_range, n)
    if cfg.parameter_draw_order == "operator_test":
        jz = np.empty(len(edges), dtype=float)
        jxy = np.empty(len(edges), dtype=float)
        for edge_idx in range(len(edges)):
            jz[edge_idx] = _uniform_scalar_or_one(
                rng, bool(cfg.random_jz), cfg.jz_range
            )
            jxy[edge_idx] = _uniform_scalar_or_one(
                rng, bool(cfg.random_jxy), cfg.jxy_range
            )
        signs = (
            rng.choice(np.array([-1.0, 1.0]), size=n)
            if random_drive_sign
            else np.ones(n, dtype=float)
        )
        drive = signs * _uniform_or_ones(
            rng, bool(cfg.random_drive), cfg.drive_range, n
        )
    else:
        jz = _uniform_or_ones(rng, bool(cfg.random_jz), cfg.jz_range, len(edges))
        jxy = _uniform_or_ones(rng, bool(cfg.random_jxy), cfg.jxy_range, len(edges))
        drive = _uniform_or_ones(rng, bool(cfg.random_drive), cfg.drive_range, n)
        if random_drive_sign:
            drive *= rng.choice(np.array([-1.0, 1.0]), size=n)
    break_coeffs = _uniform_or_ones(rng, bool(cfg.random_break), cfg.break_range, n)
    return {
        "edges": edges,
        "h": float(cfg.h_scale) * h.astype(float),
        "jz": float(cfg.jz_scale) * jz.astype(float),
        "jxy": float(cfg.jxy_scale) * jxy.astype(float),
        "drive_coeffs": float(cfg.drive_amplitude) * drive.astype(float),
        "break_coeffs": float(cfg.break_scale) * break_coeffs.astype(float),
    }

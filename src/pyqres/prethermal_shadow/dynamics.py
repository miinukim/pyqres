from __future__ import annotations

"""Dense global-Floquet dynamics helpers."""

from collections.abc import Sequence

import numpy as np
import scipy.linalg as la

from .config import GlobalFloquetConfig


PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def pauli_string(n_qubits: int, placements: Sequence[tuple[int, str]]) -> np.ndarray:
    ops = [PAULI["I"] for _ in range(int(n_qubits))]
    for site, pauli in placements:
        key = str(pauli).upper()
        if key not in PAULI or key == "I":
            raise ValueError(f"unsupported Pauli {pauli!r}")
        ops[int(site)] = PAULI[key]
    return kron_all(ops)


def parse_pauli_placements(spec: str) -> tuple[tuple[int, str], ...]:
    """Parse one Pauli string like ``0:Z,2:X`` into placements."""

    text = str(spec).strip()
    if not text or text.upper() == "I":
        return tuple()
    placements = []
    for part in text.split(","):
        site_text, pauli_text = part.strip().split(":", 1)
        pauli = pauli_text.strip().upper()
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"unsupported Pauli {pauli!r}")
        placements.append((int(site_text), pauli))
    return tuple(placements)


def parse_pauli_operator(
    n_qubits: int,
    spec: str,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Parse sums of Pauli strings into a dense Hermitian operator.

    Supported examples include ``0:Y``, ``0:Z,1:Z`` and
    ``0.5*0:X + -1.2*2:Z``.
    """

    text = str(spec).strip()
    dim = 2 ** int(n_qubits)
    if not text or text.upper() == "I":
        out = np.eye(dim, dtype=complex)
    else:
        out = np.zeros((dim, dim), dtype=complex)
        for raw_term in text.replace("-", "+-").split("+"):
            term = raw_term.strip()
            if not term:
                continue
            if "*" in term:
                coeff_text, pauli_text = term.split("*", 1)
                coeff = float(coeff_text.strip())
            else:
                coeff = 1.0
                pauli_text = term
            out += coeff * pauli_string(int(n_qubits), parse_pauli_placements(pauli_text))
    out = 0.5 * (out + out.conj().T)
    if normalize:
        norm2 = np.real_if_close(np.trace(out.conj().T @ out) / dim)
        norm = float(np.sqrt(max(float(norm2), 0.0)))
        if norm > 1e-15:
            out = out / norm
    return out


def chain_edges(n_qubits: int, n_memory: int, include_mr_couplings: bool = True) -> tuple[tuple[int, int], ...]:
    edges = []
    for i in range(int(n_qubits) - 1):
        if not include_mr_couplings and i == int(n_memory) - 1:
            continue
        edges.append((i, i + 1))
    return tuple(edges)


def topology_edges(n_qubits: int, n_memory: int, topology: str, include_mr_couplings: bool = True) -> tuple[tuple[int, int], ...]:
    key = str(topology).lower()
    n = int(n_qubits)
    n_mem = int(n_memory)
    if key == "chain":
        return chain_edges(n, n_mem, include_mr_couplings=include_mr_couplings)
    if key == "all_to_all":
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                crosses_mr = i < n_mem <= j
                if include_mr_couplings or not crosses_mr:
                    edges.append((i, j))
        return tuple(edges)
    raise ValueError("topology must be chain or all_to_all.")


def _uniform_or_ones(rng: np.random.Generator, randomize: bool, bounds: tuple[float, float], size: int) -> np.ndarray:
    if not randomize:
        return np.ones(int(size), dtype=float)
    low, high = (float(bounds[0]), float(bounds[1]))
    if low > high:
        raise ValueError(f"invalid random coefficient range {bounds!r}")
    return rng.uniform(low, high, size=int(size)).astype(float)


def _uniform_scalar_or_one(rng: np.random.Generator, randomize: bool, bounds: tuple[float, float]) -> float:
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


def generate_hamiltonian_parameters(cfg: GlobalFloquetConfig) -> dict[str, np.ndarray | tuple[tuple[int, int], ...]]:
    validate_floquet_config(cfg)
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.n_qubits)
    edges = topology_edges(n, int(cfg.n_memory), str(cfg.topology), bool(cfg.include_mr_couplings))
    random_drive_sign = bool(cfg.random_drive if cfg.random_drive_sign is None else cfg.random_drive_sign)
    h = _uniform_or_ones(rng, bool(cfg.random_h), cfg.h_range, n)
    if cfg.parameter_draw_order == "operator_test":
        jz = np.empty(len(edges), dtype=float)
        jxy = np.empty(len(edges), dtype=float)
        for edge_idx in range(len(edges)):
            jz[edge_idx] = _uniform_scalar_or_one(rng, bool(cfg.random_jz), cfg.jz_range)
            jxy[edge_idx] = _uniform_scalar_or_one(rng, bool(cfg.random_jxy), cfg.jxy_range)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n) if random_drive_sign else np.ones(n, dtype=float)
        drive = signs * _uniform_or_ones(rng, bool(cfg.random_drive), cfg.drive_range, n)
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


def build_global_h0(
    cfg: GlobalFloquetConfig,
    *,
    h: np.ndarray,
    jz: np.ndarray,
    jxy: np.ndarray,
    break_coeffs: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> np.ndarray:
    n = int(cfg.n_qubits)
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    for i, coeff in enumerate(h):
        out += float(coeff) * pauli_string(n, [(i, "Z")])
    for e, (i, j) in enumerate(edges):
        out += float(jz[e]) * pauli_string(n, [(i, "Z"), (j, "Z")])
        out += float(jxy[e]) * pauli_string(n, [(i, "X"), (j, "X")])
        out += float(jxy[e]) * pauli_string(n, [(i, "Y"), (j, "Y")])
    axis = str(cfg.break_axis).upper()
    for i, coeff in enumerate(break_coeffs):
        if float(coeff) != 0.0:
            out += float(coeff) * pauli_string(n, [(i, axis)])
    return 0.5 * (out + out.conj().T)


def build_drive_hamiltonian(cfg: GlobalFloquetConfig, drive_coeffs: np.ndarray) -> np.ndarray:
    n = int(cfg.n_qubits)
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    axis = str(cfg.drive_axis).upper()
    for i, coeff in enumerate(drive_coeffs):
        out += float(coeff) * pauli_string(n, [(i, axis)])
    return 0.5 * (out + out.conj().T)


def fast_period(cfg: GlobalFloquetConfig) -> float:
    return 2.0 * np.pi / float(cfg.omega)


def step_duration(cfg: GlobalFloquetConfig) -> float:
    return int(cfg.n_cycles_per_step) * fast_period(cfg)


def build_fast_period_unitary(h0: np.ndarray, drive: np.ndarray, period: float) -> np.ndarray:
    half = 0.5 * float(period)
    u_plus = la.expm(-1j * half * (h0 + drive))
    u_minus = la.expm(-1j * half * (h0 - drive))
    return u_minus @ u_plus


def build_step_unitary(cfg: GlobalFloquetConfig, h0: np.ndarray, drive: np.ndarray) -> np.ndarray:
    u_period = build_fast_period_unitary(h0, drive, fast_period(cfg))
    return np.linalg.matrix_power(u_period, int(cfg.n_cycles_per_step))


def density_zero(n_qubits: int) -> np.ndarray:
    rho = np.zeros((2 ** int(n_qubits), 2 ** int(n_qubits)), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def density_plus(n_qubits: int) -> np.ndarray:
    one = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    return kron_all([one for _ in range(int(n_qubits))])


def partial_trace_memory_first(rho: np.ndarray, n_memory: int, n_readout: int, keep: str) -> np.ndarray:
    dim_m = 2 ** int(n_memory)
    dim_r = 2 ** int(n_readout)
    tensor = np.asarray(rho, dtype=complex).reshape(dim_m, dim_r, dim_m, dim_r)
    if keep == "memory":
        return np.trace(tensor, axis1=1, axis2=3)
    if keep == "readout":
        return np.trace(tensor, axis1=0, axis2=2)
    raise ValueError("keep must be memory or readout.")


def project_density(rho: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    rho = 0.5 * (np.asarray(rho, dtype=complex) + np.asarray(rho, dtype=complex).conj().T)
    tr = np.trace(rho)
    if abs(tr) > tol:
        rho = rho / tr
    return rho


__all__ = [
    "PAULI",
    "build_drive_hamiltonian",
    "build_fast_period_unitary",
    "build_global_h0",
    "build_step_unitary",
    "chain_edges",
    "density_plus",
    "density_zero",
    "fast_period",
    "generate_hamiltonian_parameters",
    "kron_all",
    "partial_trace_memory_first",
    "pauli_string",
    "parse_pauli_operator",
    "parse_pauli_placements",
    "project_density",
    "step_duration",
    "topology_edges",
    "validate_floquet_config",
]

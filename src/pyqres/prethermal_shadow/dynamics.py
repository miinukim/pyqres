from __future__ import annotations

"""Dense global-Floquet dynamics helpers."""

from collections.abc import Sequence

import numpy as np
import scipy.linalg as la

from .config import GlobalFloquetConfig, InputEncodingConfig


PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

PauliPlacements = tuple[tuple[int, str], ...]
PauliTerm = tuple[float, PauliPlacements]


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def pauli_string(n_qubits: int, placements: Sequence[tuple[int, str]]) -> np.ndarray:
    ops = [PAULI["I"] for _ in range(int(n_qubits))]
    for site, pauli in placements:
        index = int(site)
        if index < 0 or index >= int(n_qubits):
            raise ValueError(f"Pauli placement qubit {index} is outside [0, {int(n_qubits) - 1}].")
        key = str(pauli).upper()
        if key not in PAULI or key == "I":
            raise ValueError(f"unsupported Pauli {pauli!r}")
        ops[index] = PAULI[key]
    return kron_all(ops)


def parse_pauli_placements(spec: str) -> PauliPlacements:
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


def parse_pauli_terms(spec: str, *, normalize: bool = True) -> tuple[PauliTerm, ...]:
    """Parse a weighted sum of Pauli strings without allocating dense matrices."""

    text = str(spec).strip()
    raw_terms = ("I",) if not text or text.upper() == "I" else text.replace("-", "+-").split("+")
    combined: dict[PauliPlacements, float] = {}
    for raw_term in raw_terms:
        term = raw_term.strip()
        if not term:
            continue
        if "*" in term:
            coeff_text, pauli_text = term.split("*", 1)
            coeff = float(coeff_text.strip())
        else:
            coeff = 1.0
            pauli_text = term
        placements = parse_pauli_placements(pauli_text)
        by_site = {int(site): str(pauli).upper() for site, pauli in placements}
        canonical = tuple(sorted(by_site.items()))
        combined[canonical] = combined.get(canonical, 0.0) + coeff

    terms = [(coeff, placements) for placements, coeff in combined.items() if abs(coeff) > 1e-15]
    if normalize and terms:
        norm = float(np.sqrt(sum(float(coeff) ** 2 for coeff, _ in terms)))
        terms = [(float(coeff) / norm, placements) for coeff, placements in terms]
    return tuple(terms)


def pauli_terms_matrix(n_qubits: int, terms: Sequence[PauliTerm]) -> np.ndarray:
    """Construct a dense Hermitian matrix from symbolic Pauli terms."""

    dim = 2 ** int(n_qubits)
    out = np.zeros((dim, dim), dtype=complex)
    for coeff, placements in terms:
        out += float(coeff) * pauli_string(int(n_qubits), placements)
    return 0.5 * (out + out.conj().T)


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

    return pauli_terms_matrix(
        int(n_qubits),
        parse_pauli_terms(spec, normalize=normalize),
    )


def validate_axis(axis: str, name: str) -> str:
    """Normalize and validate a single-qubit Pauli axis."""

    out = str(axis).upper()
    if out not in {"X", "Y", "Z"}:
        raise ValueError(f"{name} must be x, y, or z.")
    return out


def resolve_input_qubits(cfg: InputEncodingConfig, floquet: GlobalFloquetConfig) -> tuple[int, ...]:
    """Resolve symbolic input targets to physical qubit indices."""

    spec = cfg.input_qubits
    if isinstance(spec, str):
        key = spec.lower()
        if key == "memory":
            memory_qubits, _ = resolve_qubit_partition(floquet)
            return memory_qubits
        if key == "all":
            return tuple(range(int(floquet.n_qubits)))
        raise ValueError("input_qubits must be 'memory', 'all', or a sequence of indices.")
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


def resolve_qubit_partition(cfg: GlobalFloquetConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
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


def generate_hamiltonian_parameters(cfg: GlobalFloquetConfig) -> dict[str, np.ndarray | tuple[tuple[int, int], ...]]:
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
    return pauli_terms_matrix(
        int(cfg.n_qubits),
        global_h0_pauli_terms(
            cfg,
            h=h,
            jz=jz,
            jxy=jxy,
            break_coeffs=break_coeffs,
            edges=edges,
        ),
    )


def global_h0_pauli_terms(
    cfg: GlobalFloquetConfig,
    *,
    h: np.ndarray,
    jz: np.ndarray,
    jxy: np.ndarray,
    break_coeffs: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> tuple[PauliTerm, ...]:
    """Return the static prethermal Hamiltonian as symbolic Pauli terms."""

    terms: list[PauliTerm] = []
    for i, coeff in enumerate(h):
        if float(coeff) != 0.0:
            terms.append((float(coeff), ((i, "Z"),)))
    for e, (i, j) in enumerate(edges):
        if float(jz[e]) != 0.0:
            terms.append((float(jz[e]), ((i, "Z"), (j, "Z"))))
        if float(jxy[e]) != 0.0:
            terms.append((float(jxy[e]), ((i, "X"), (j, "X"))))
            terms.append((float(jxy[e]), ((i, "Y"), (j, "Y"))))
    axis = str(cfg.break_axis).upper()
    for i, coeff in enumerate(break_coeffs):
        if float(coeff) != 0.0:
            terms.append((float(coeff), ((i, axis),)))
    return tuple(terms)


def build_drive_hamiltonian(cfg: GlobalFloquetConfig, drive_coeffs: np.ndarray) -> np.ndarray:
    return pauli_terms_matrix(
        int(cfg.n_qubits),
        drive_pauli_terms(cfg, drive_coeffs),
    )


def drive_pauli_terms(cfg: GlobalFloquetConfig, drive_coeffs: np.ndarray) -> tuple[PauliTerm, ...]:
    """Return the square-drive Hamiltonian as symbolic Pauli terms."""

    axis = str(cfg.drive_axis).upper()
    return tuple(
        (float(coeff), ((i, axis),))
        for i, coeff in enumerate(drive_coeffs)
        if float(coeff) != 0.0
    )


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
    n_memory = int(n_memory)
    n_readout = int(n_readout)
    if keep == "memory":
        keep_qubits = tuple(range(n_memory))
    elif keep == "readout":
        keep_qubits = tuple(range(n_memory, n_memory + n_readout))
    else:
        raise ValueError("keep must be memory or readout.")
    return partial_trace_qubits(rho, n_memory + n_readout, keep_qubits)


def reorder_qubit_operator(
    operator: np.ndarray,
    current_order: Sequence[int],
    target_order: Sequence[int],
) -> np.ndarray:
    """Reorder matrix tensor factors from ``current_order`` to ``target_order``."""

    current = tuple(int(qubit) for qubit in current_order)
    target = tuple(int(qubit) for qubit in target_order)
    if (
        len(current) != len(target)
        or len(set(current)) != len(current)
        or len(set(target)) != len(target)
        or set(current) != set(target)
    ):
        raise ValueError("current_order and target_order must contain the same unique qubits.")
    n_qubits = len(current)
    dim = 2**n_qubits
    matrix = np.asarray(operator, dtype=complex)
    if matrix.shape != (dim, dim):
        raise ValueError(f"operator must have shape {(dim, dim)}.")
    positions = {qubit: index for index, qubit in enumerate(current)}
    row_axes = [positions[qubit] for qubit in target]
    axes = [*row_axes, *(n_qubits + axis for axis in row_axes)]
    return matrix.reshape((2,) * (2 * n_qubits)).transpose(axes).reshape(dim, dim)


def combine_subsystem_states(
    rho_memory: np.ndarray,
    rho_readout: np.ndarray,
    memory_qubits: Sequence[int],
    readout_qubits: Sequence[int],
) -> np.ndarray:
    """Embed a memory/readout product state into physical qubit order."""

    memory = tuple(int(qubit) for qubit in memory_qubits)
    readout = tuple(int(qubit) for qubit in readout_qubits)
    subsystem_order = (*memory, *readout)
    return reorder_qubit_operator(
        np.kron(np.asarray(rho_memory, dtype=complex), np.asarray(rho_readout, dtype=complex)),
        subsystem_order,
        tuple(range(len(subsystem_order))),
    )


def partial_trace_qubits(
    rho: np.ndarray,
    n_qubits: int,
    keep_qubits: Sequence[int],
) -> np.ndarray:
    """Trace out arbitrary qubits and order the result as ``keep_qubits``."""

    n_qubits = int(n_qubits)
    keep = tuple(int(qubit) for qubit in keep_qubits)
    if len(set(keep)) != len(keep):
        raise ValueError("keep_qubits must not contain duplicates.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in keep):
        raise ValueError("keep_qubits contains an out-of-range qubit.")
    dim = 2**n_qubits
    matrix = np.asarray(rho, dtype=complex)
    if matrix.shape != (dim, dim):
        raise ValueError(f"rho must have shape {(dim, dim)}.")
    keep_set = set(keep)
    traced = tuple(qubit for qubit in range(n_qubits) if qubit not in keep_set)
    order = (*keep, *traced)
    tensor = reorder_qubit_operator(matrix, tuple(range(n_qubits)), order)
    dim_keep = 2 ** len(keep)
    dim_traced = 2 ** len(traced)
    tensor = tensor.reshape(dim_keep, dim_traced, dim_keep, dim_traced)
    return np.trace(tensor, axis1=1, axis2=3)


def project_density(rho: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    rho = 0.5 * (np.asarray(rho, dtype=complex) + np.asarray(rho, dtype=complex).conj().T)
    tr = np.trace(rho)
    if abs(tr) > tol:
        rho = rho / tr
    return rho


__all__ = [
    "PAULI",
    "PauliPlacements",
    "PauliTerm",
    "build_drive_hamiltonian",
    "build_fast_period_unitary",
    "build_global_h0",
    "build_step_unitary",
    "chain_edges",
    "combine_subsystem_states",
    "density_plus",
    "density_zero",
    "fast_period",
    "generate_hamiltonian_parameters",
    "global_h0_pauli_terms",
    "drive_pauli_terms",
    "input_beta_array",
    "kron_all",
    "partial_trace_memory_first",
    "partial_trace_qubits",
    "pauli_string",
    "parse_pauli_operator",
    "parse_pauli_placements",
    "parse_pauli_terms",
    "pauli_terms_matrix",
    "project_density",
    "reorder_qubit_operator",
    "resolve_qubit_partition",
    "resolve_input_qubits",
    "step_duration",
    "topology_edges",
    "validate_floquet_config",
    "validate_axis",
]

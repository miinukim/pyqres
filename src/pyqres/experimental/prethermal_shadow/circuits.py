from __future__ import annotations

"""Qiskit circuit builders for prethermal shadow reservoirs."""

from collections.abc import Sequence

import numpy as np

from .config import ResolvedPrethermalShadowConfig

try:
    from qiskit import QuantumCircuit
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = None  # type: ignore


def require_qiskit() -> None:
    """Raise a clear error if Qiskit is unavailable."""

    if QuantumCircuit is None:
        raise ImportError("qiskit is required for prethermal shadow circuits.")


def append_input_write(qc: "QuantumCircuit", memory: Sequence[int], u: float, cfg: ResolvedPrethermalShadowConfig) -> None:
    """Append input-dependent rotations on memory qubits."""

    angle_scale = float(u + cfg.base.input_write.bias)
    axis = cfg.base.input_write.axis
    for i, q in enumerate(memory):
        angle = 2.0 * float(cfg.beta[i]) * angle_scale
        if axis == "x":
            qc.rx(angle, int(q))
        elif axis == "y":
            qc.ry(angle, int(q))
        elif axis == "z":
            qc.rz(angle, int(q))
        else:  # pragma: no cover - validated earlier
            raise ValueError(f"unsupported input axis {axis}")


def append_floquet_period(qc: "QuantumCircuit", memory: Sequence[int], cfg: ResolvedPrethermalShadowConfig) -> None:
    """Append one memory-only Floquet period."""

    tau = float(cfg.base.floquet.tau)
    for i, q in enumerate(memory):
        qc.rz(2.0 * float(cfg.h[i]) * tau, int(q))
    for e, (i, j) in enumerate(cfg.edges):
        qc.rzz(2.0 * float(cfg.jz[e]) * tau, int(memory[i]), int(memory[j]))
    for e, (i, j) in enumerate(cfg.edges):
        angle = 2.0 * float(cfg.jxy[e]) * tau
        qc.rxx(angle, int(memory[i]), int(memory[j]))
        qc.ryy(angle, int(memory[i]), int(memory[j]))
    for i, q in enumerate(memory):
        angle = 2.0 * float(cfg.x_break[i]) * tau
        if angle != 0.0:
            qc.rx(angle, int(q))


def append_prethermal_block(qc: "QuantumCircuit", memory: Sequence[int], cfg: ResolvedPrethermalShadowConfig) -> None:
    """Append all repeated Floquet periods for one time step."""

    for _ in range(int(cfg.base.floquet.n_floquet)):
        append_floquet_period(qc, memory, cfg)


def append_prepare_readout_plus(qc: "QuantumCircuit", readout: Sequence[int]) -> None:
    """Prepare readout qubits in |+>."""

    for q in readout:
        qc.h(int(q))


def append_transducer(qc: "QuantumCircuit", memory: Sequence[int], readout: Sequence[int], cfg: ResolvedPrethermalShadowConfig) -> None:
    """Append the optional memory-readout RZZ transducer."""

    if cfg.base.transducer is None or cfg.g is None:
        return
    tau_c = float(cfg.base.transducer.tau_c)
    if tau_c == 0.0:
        return
    for alpha, rq in enumerate(readout):
        for i, mq in enumerate(memory):
            qc.rzz(2.0 * tau_c * float(cfg.g[alpha, i]), int(rq), int(mq))


def append_shadow_basis_rotations(qc: "QuantumCircuit", readout: Sequence[int], basis_for_step: Sequence[str]) -> None:
    """Rotate readout qubits so computational measurements sample Pauli bases."""

    for q, basis in zip(readout, basis_for_step):
        basis = str(basis).upper()
        if basis == "X":
            qc.h(int(q))
        elif basis == "Y":
            qc.sdg(int(q))
            qc.h(int(q))
        elif basis == "Z":
            pass
        else:
            raise ValueError(f"unsupported shadow basis {basis}")


def build_streaming_circuit(inputs: Sequence[float] | np.ndarray, basis_schedule: np.ndarray, cfg: ResolvedPrethermalShadowConfig) -> "QuantumCircuit":
    """Build one full streaming circuit for a fixed basis schedule."""

    require_qiskit()
    values = np.asarray(inputs, dtype=float).reshape(-1)
    schedule = np.asarray(basis_schedule, dtype="U1")
    n_memory = int(cfg.base.n_memory)
    n_readout = int(cfg.base.n_readout)
    if schedule.shape != (values.shape[0], n_readout):
        raise ValueError(f"basis_schedule must have shape {(values.shape[0], n_readout)}, got {schedule.shape}.")
    total_qubits = n_memory + n_readout
    memory = tuple(range(n_memory))
    readout = tuple(range(n_memory, total_qubits))
    qc = QuantumCircuit(total_qubits, values.shape[0] * n_readout)
    cidx = 0
    for t, u_t in enumerate(values):
        append_prepare_readout_plus(qc, readout)
        append_input_write(qc, memory, float(u_t), cfg)
        append_prethermal_block(qc, memory, cfg)
        append_transducer(qc, memory, readout, cfg)
        append_shadow_basis_rotations(qc, readout, schedule[t])
        for k, rq in enumerate(readout):
            qc.measure(int(rq), cidx + k)
        cidx += n_readout
        for rq in readout:
            qc.reset(int(rq))
    return qc

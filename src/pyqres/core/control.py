"""Measurement and feedback primitives shared by the reservoir systems.

The rest of the package treats the measurement protocol as a small, explicit
state machine: evolve the joint system, measure the ancilla register, optionally
apply a conditioned gate, and either reset or keep the ancilla state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


MeasurementMode = Literal["projective", "weak"]
PostMeasurementMode = Literal["reset", "keep"]
ConditionedGate = Literal["none", "system_x", "system_rx", "system_rz", "ancilla_x", "ancilla_rx", "ancilla_rz"]
ConditionRule = Literal["nonzero", "all_one"]


@dataclass
class MeasurementControlConfig:
    """Configuration for the post-evolution measurement/control step.

    `measurement_mode` chooses projective versus weak ancilla measurement.
    `post_measurement_mode` controls whether the ancilla is reset to |0...0>
    after the branch mixture is formed. The conditioned-gate fields implement a
    simple output feedback rule keyed by the classical measurement outcome.
    """

    measurement_mode: MeasurementMode = "projective"
    measurement_strength: float = 1.0
    post_measurement_mode: PostMeasurementMode = "reset"
    conditioned_gate: ConditionedGate = "none"
    conditioned_gate_angle: float = float(np.pi)
    conditioned_gate_target: int = 0
    conditioned_gate_condition: ConditionRule = "nonzero"

    def validated(self, n_system: int, n_ancilla: int) -> "MeasurementControlConfig":
        # The exact simulator indexes system and ancilla targets in different
        # local coordinate systems, so validate targets before dense matrices are
        # built. This catches config mistakes while the error message is still
        # small and readable.
        strength = float(self.measurement_strength)
        if not (0.0 <= strength <= 1.0):
            raise ValueError("measurement_strength must lie in [0, 1].")
        if self.conditioned_gate.startswith("system_"):
            if not (0 <= int(self.conditioned_gate_target) < n_system):
                raise ValueError("conditioned_gate_target must index a system qubit.")
        elif self.conditioned_gate.startswith("ancilla_"):
            if not (0 <= int(self.conditioned_gate_target) < n_ancilla):
                raise ValueError("conditioned_gate_target must index an ancilla qubit.")
        return self


def single_qubit_gate(kind: str, angle: float) -> np.ndarray:
    """Return the dense one-qubit gate used by output-conditioned feedback."""

    if kind == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if kind == "rx":
        c = np.cos(0.5 * angle)
        s = np.sin(0.5 * angle)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if kind == "rz":
        return np.array(
            [
                [np.exp(-0.5j * angle), 0.0],
                [0.0, np.exp(0.5j * angle)],
            ],
            dtype=complex,
        )
    raise ValueError(f"Unsupported conditioned gate kind '{kind}'.")


def embed_single_qubit_gate(n_qubits: int, target: int, gate: np.ndarray) -> np.ndarray:
    """Embed a one-qubit dense gate into the full register by Kronecker product."""

    eye = np.eye(2, dtype=complex)
    out = np.array([[1.0 + 0.0j]])
    for idx in range(n_qubits):
        out = np.kron(out, gate if idx == target else eye)
    return out


def weak_measurement_kraus(n_ancilla: int, strength: float) -> list[np.ndarray]:
    """Build tensor-product weak-measurement Kraus operators for all outcomes."""

    m0 = np.array(
        [
            [np.sqrt((1.0 + strength) / 2.0), 0.0],
            [0.0, np.sqrt((1.0 - strength) / 2.0)],
        ],
        dtype=complex,
    )
    m1 = np.array(
        [
            [np.sqrt((1.0 - strength) / 2.0), 0.0],
            [0.0, np.sqrt((1.0 + strength) / 2.0)],
        ],
        dtype=complex,
    )
    ops = []
    for outcome in range(2**n_ancilla):
        op = np.array([[1.0 + 0.0j]])
        for bit in range(n_ancilla):
            # Outcomes are interpreted as big-endian ancilla bit strings so the
            # generated Kraus list lines up with computational-basis indexing.
            local = m1 if (outcome >> (n_ancilla - 1 - bit)) & 1 else m0
            op = np.kron(op, local)
        ops.append(op)
    return ops


def projective_measurement_kraus(n_ancilla: int) -> list[np.ndarray]:
    """Build computational-basis projectors for the ancilla register."""

    dim = 2**n_ancilla
    ops = []
    for outcome in range(dim):
        op = np.zeros((dim, dim), dtype=complex)
        op[outcome, outcome] = 1.0
        ops.append(op)
    return ops

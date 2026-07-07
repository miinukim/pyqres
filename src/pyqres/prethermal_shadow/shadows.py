from __future__ import annotations

"""Classical-shadow sampling and feature reconstruction helpers."""

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations, product

import numpy as np

from .dynamics import PAULI, kron_all, pauli_string


PauliLabel = tuple[tuple[int, str], ...]


def generate_pauli_labels(n_readout: int, pauli_k: int) -> list[PauliLabel]:
    """Generate non-identity readout Pauli labels up to the requested weight."""

    n_readout = int(n_readout)
    pauli_k = int(pauli_k)
    labels: list[PauliLabel] = []
    for weight in range(1, pauli_k + 1):
        for qubits in combinations(range(n_readout), weight):
            for paulis in product(("X", "Y", "Z"), repeat=weight):
                labels.append(tuple((int(q), str(p)) for q, p in zip(qubits, paulis)))
    return labels


def pauli_label_to_string(label: PauliLabel) -> str:
    """Return a compact string representation such as Z0 or X0*Y2."""

    return "*".join(f"{pauli}{qubit}" for qubit, pauli in label)


def feature_label_strings(n_readout: int, pauli_k: int, include_bias: bool = True) -> list[str]:
    """Return feature label strings in the same order as reconstructed columns."""

    labels = [pauli_label_to_string(item) for item in generate_pauli_labels(n_readout, pauli_k)]
    if include_bias:
        return ["bias", *labels]
    return labels


def partial_shadow_feature_names(n_readout: int, pauli_k: int, include_bias: bool = True) -> list[str]:
    """Return readout-subset feature names such as X_r0 and X_r0 Y_r1."""

    labels = []
    for label in generate_pauli_labels(n_readout, pauli_k):
        labels.append(" ".join(f"{pauli}_r{qubit}" for qubit, pauli in label))
    if include_bias:
        return ["bias", *labels]
    return labels


def readout_pauli_operator(n_readout: int, label: PauliLabel) -> np.ndarray:
    """Build a dense Pauli string acting only on readout Hilbert space."""

    return pauli_string(int(n_readout), label)


def exact_readout_expectations(
    rho_readout: np.ndarray,
    labels: Sequence[PauliLabel],
    *,
    include_bias: bool = True,
) -> np.ndarray:
    """Compute exact readout Pauli expectations from a readout marginal."""

    rho = np.asarray(rho_readout, dtype=complex)
    values = [float(np.real_if_close(np.trace(readout_pauli_operator(int(np.log2(rho.shape[0])), label) @ rho))) for label in labels]
    if include_bias:
        return np.asarray([1.0, *values], dtype=float)
    return np.asarray(values, dtype=float)


def weak_povm_effect(pauli: str, m: int, strength: float) -> np.ndarray:
    """Return E_m = (I + m s A) / 2 for one weak Pauli measurement."""

    s = float(strength)
    if not (0.0 < s <= 1.0):
        raise ValueError("weak_strength must lie in (0, 1].")
    axis = str(pauli).upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("weak POVM axis must be X, Y, or Z.")
    return 0.5 * (PAULI["I"] + int(m) * s * PAULI[axis])


def joint_povm_effect(axes: Sequence[str], outcomes: Sequence[int], strength: float) -> np.ndarray:
    """Return tensor-product weak POVM effect for a readout shot."""

    return kron_all([weak_povm_effect(axis, int(m), strength) for axis, m in zip(axes, outcomes)])


def outcome_probabilities(rho_readout: np.ndarray, axes: Sequence[str], strength: float) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Return all local weak/projective outcome probabilities for chosen axes."""

    n_readout = len(tuple(axes))
    outcomes = list(product((-1, 1), repeat=n_readout))
    probs = []
    for outcome in outcomes:
        effect = joint_povm_effect(axes, outcome, strength)
        probs.append(float(np.real_if_close(np.trace(effect @ rho_readout))))
    arr = np.asarray(probs, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        arr = np.full_like(arr, 1.0 / arr.size)
    else:
        arr /= total
    return outcomes, arr


def sample_partial_shadow_features(
    rho_readout: np.ndarray,
    labels: Sequence[PauliLabel],
    *,
    shots: int,
    measurement_type: str = "projective",
    weak_strength: float = 1.0,
    include_bias: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Estimate readout Pauli features with local projective or weak shadows."""

    if int(shots) <= 0:
        raise ValueError("shots must be positive.")
    rho = np.asarray(rho_readout, dtype=complex)
    n_readout = int(round(np.log2(rho.shape[0])))
    if rho.shape != (2**n_readout, 2**n_readout):
        raise ValueError("rho_readout must be a square 2**n_readout matrix.")
    kind = str(measurement_type).lower()
    if kind not in {"projective", "weak"}:
        raise ValueError("measurement_type must be projective or weak.")
    strength = 1.0 if kind == "projective" else float(weak_strength)
    if not (0.0 < strength <= 1.0):
        raise ValueError("weak_strength must lie in (0, 1].")

    local_rng = np.random.default_rng(seed) if rng is None else rng
    axes_all = np.asarray(["X", "Y", "Z"], dtype="U1")
    estimates = np.zeros((int(shots), len(labels)), dtype=float)
    axes_record = np.empty((int(shots), n_readout), dtype="U1")
    outcomes_record = np.empty((int(shots), n_readout), dtype=np.int8)
    for shot in range(int(shots)):
        axes = local_rng.choice(axes_all, size=n_readout)
        outcomes, probs = outcome_probabilities(rho, axes, strength)
        sampled = outcomes[int(local_rng.choice(len(outcomes), p=probs))]
        axes_record[shot] = axes
        outcomes_record[shot] = np.asarray(sampled, dtype=np.int8)
        for col, label in enumerate(labels):
            value = 1.0
            for qubit, pauli in label:
                if str(axes[int(qubit)]) != str(pauli):
                    value = 0.0
                    break
                value *= 3.0 * float(sampled[int(qubit)]) / strength
            estimates[shot, col] = value
    features = estimates.mean(axis=0)
    if include_bias:
        features = np.asarray([1.0, *features], dtype=float)
    if return_raw:
        return features, {"axes": axes_record, "outcomes": outcomes_record, "shot_estimates": estimates}
    return features


def sample_basis_schedules(
    shots: int,
    n_steps: int,
    n_readout: int,
    bases: Sequence[str] = ("X", "Y", "Z"),
    seed: int | None = None,
) -> np.ndarray:
    """Sample independent random Pauli-basis schedules."""

    rng = np.random.default_rng(seed)
    base_arr = np.asarray([str(item).upper() for item in bases], dtype="U1")
    return rng.choice(base_arr, size=(int(shots), int(n_steps), int(n_readout)))


def group_basis_schedules(schedules: np.ndarray) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """Group identical full schedules.

    Returns tuples of ``(schedule, count, shot_indices)``.
    """

    arr = np.asarray(schedules, dtype="U1")
    if arr.ndim != 3:
        raise ValueError(f"schedules must be a 3D array, got shape {arr.shape}.")
    groups: dict[bytes, list[int]] = {}
    for idx, schedule in enumerate(arr):
        groups.setdefault(np.ascontiguousarray(schedule).tobytes(), []).append(idx)
    out = []
    for indices in groups.values():
        shot_indices = np.asarray(indices, dtype=int)
        out.append((arr[indices[0]].copy(), int(len(indices)), shot_indices))
    return out


def bit_at_from_right(bitstring: str, idx: int) -> int:
    """Read Qiskit classical bit index using the existing backend convention."""

    compact = bitstring.replace(" ", "")
    return 1 if compact[-(int(idx) + 1)] == "1" else 0


def counts_to_outcomes(counts: Mapping[str, int], *, shots: int, n_steps: int, n_readout: int) -> np.ndarray:
    """Expand Qiskit counts into measurement outcomes with shape shots,T,R."""

    out = np.zeros((int(shots), int(n_steps), int(n_readout)), dtype=np.int8)
    row = 0
    for bitstring, count in counts.items():
        for _ in range(int(count)):
            if row >= int(shots):
                raise ValueError("counts contain more outcomes than requested shots.")
            for t in range(int(n_steps)):
                offset = t * int(n_readout)
                for q in range(int(n_readout)):
                    out[row, t, q] = bit_at_from_right(bitstring, offset + q)
            row += 1
    if row != int(shots):
        raise ValueError(f"counts contain {row} outcomes, expected {shots}.")
    return out


def merge_group_outcomes(groups: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Merge grouped schedules/outcomes into shot-major arrays."""

    schedules = []
    outcomes = []
    for schedule, group_outcomes in groups:
        count = int(group_outcomes.shape[0])
        schedules.append(np.repeat(np.asarray(schedule, dtype="U1")[None, :, :], count, axis=0))
        outcomes.append(np.asarray(group_outcomes, dtype=np.int8))
    if not schedules:
        raise ValueError("at least one outcome group is required.")
    return np.concatenate(schedules, axis=0), np.concatenate(outcomes, axis=0)


def estimate_shadow_features(
    schedules: np.ndarray,
    outcomes: np.ndarray,
    labels: Sequence[PauliLabel],
    *,
    include_bias: bool = True,
) -> np.ndarray:
    """Estimate Pauli shadow features from basis schedules and bit outcomes."""

    basis = np.asarray(schedules, dtype="U1")
    bits = np.asarray(outcomes, dtype=np.int8)
    if basis.shape != bits.shape:
        raise ValueError(f"schedules and outcomes must share shape, got {basis.shape} and {bits.shape}.")
    if basis.ndim != 3:
        raise ValueError(f"schedules and outcomes must be 3D, got shape {basis.shape}.")
    n_shots, n_steps, _ = basis.shape
    features = np.zeros((n_steps, len(labels)), dtype=float)
    signs = np.where(bits == 0, 1.0, -1.0)
    for col, label in enumerate(labels):
        values = np.ones((n_shots, n_steps), dtype=float)
        for qubit, pauli in label:
            match = basis[:, :, int(qubit)] == str(pauli)
            values *= np.where(match, 3.0 * signs[:, :, int(qubit)], 0.0)
        features[:, col] = values.mean(axis=0)
    if include_bias:
        features = np.hstack([np.ones((n_steps, 1), dtype=float), features])
    if not np.isfinite(features).all():
        raise FloatingPointError("non-finite shadow features")
    return features


def schedule_counts(schedules: np.ndarray) -> Counter[bytes]:
    """Return compact counts for testing/debugging schedule grouping."""

    return Counter(np.ascontiguousarray(item).tobytes() for item in np.asarray(schedules, dtype="U1"))

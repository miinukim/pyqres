"""Deterministic exact reservoir frontend for classical feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re

import numpy as np

from .exact_qrc import ExactQRCModel, ExactQRCModelConfig, partial_trace_ancilla


_PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _pauli_basis_matrices(n_qubits: int) -> tuple[np.ndarray, ...]:
    return tuple(_kron_all([_PAULI_1Q[label] for label in labels]) for labels in product(("I", "X", "Y", "Z"), repeat=n_qubits))


def _partial_trace_system(op: np.ndarray, dim_system: int, dim_ancilla: int) -> np.ndarray:
    """Trace out the first subsystem, interpreted as the memory/system register."""

    return np.trace(op.reshape(dim_system, dim_ancilla, dim_system, dim_ancilla), axis1=0, axis2=2)


def _pauli_placement_matrix(n_qubits: int, placements: list[tuple[int, str]]) -> np.ndarray:
    ops = [_PAULI_1Q["I"] for _ in range(int(n_qubits))]
    for site, pauli in placements:
        if not (0 <= int(site) < int(n_qubits)):
            raise ValueError(f"Observable site {site} is out of range for n_qubits={n_qubits}.")
        key = str(pauli).upper()
        if key not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported Pauli label '{pauli}'.")
        ops[int(site)] = _PAULI_1Q[key]
    return _kron_all(ops)


_COEFF_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*\*\s*|\s+)(.+)$")
_PAULI_TOKEN_RE = re.compile(r"^(?:([XYZxyz])(\d+)|(\d+):([XYZxyz]))$")


def _parse_observable_term(text: str) -> tuple[float, str]:
    match = _COEFF_RE.match(text.strip())
    if match is None:
        return 1.0, text.strip()
    return float(match.group(1)), match.group(2).strip()


def _parse_pauli_placements(text: str) -> list[tuple[int, str]]:
    raw = text.strip()
    if not raw or raw.upper() == "I":
        return []
    placements: list[tuple[int, str]] = []
    for token in re.split(r"[\s,;*]+", raw):
        if not token:
            continue
        match = _PAULI_TOKEN_RE.match(token)
        if match is None:
            raise ValueError(
                f"Unsupported Pauli token '{token}'. Use forms like Z0, X1, or 0:Z."
            )
        if match.group(1) is not None:
            pauli = match.group(1).upper()
            site = int(match.group(2))
        else:
            site = int(match.group(3))
            pauli = match.group(4).upper()
        placements.append((site, pauli))
    return placements


def parse_readout_observable(n_qubits: int, spec: str, *, normalize: bool = False) -> np.ndarray:
    """Parse a Hermitian Pauli-sum observable on the readout register.

    Supported examples include ``Z0``, ``0:Z``, ``Z0 Z1`` and
    ``0.5*Z0 + 0.5*Z1``.
    """

    dim = 2 ** int(n_qubits)
    text = str(spec).strip()
    if not text:
        raise ValueError("Observable spec must not be empty.")
    out = np.zeros((dim, dim), dtype=complex)
    for raw_term in re.sub(r"(?<![eE])-", "+-", text).split("+"):
        term = raw_term.strip()
        if not term:
            continue
        coeff, pauli_text = _parse_observable_term(term)
        out += coeff * _pauli_placement_matrix(int(n_qubits), _parse_pauli_placements(pauli_text))
    out = 0.5 * (out + out.conj().T)
    if normalize:
        norm2 = np.real_if_close(np.trace(out.conj().T @ out) / dim)
        norm = float(np.sqrt(max(float(norm2), 0.0)))
        if norm > 1e-15:
            out = out / norm
    return out


@dataclass
class ChannelMapReservoirConfig(ExactQRCModelConfig):
    """Configuration for expectation-value features from the exact channel."""

    include_bias: bool = True
    use_shot_noise: bool = False
    shots: int = 4096
    init_state: str = "maximally_mixed"  # "maximally_mixed" or "zero"


@dataclass
class ObservableChannelMapReservoirConfig(ChannelMapReservoirConfig):
    """Configuration for explicit observable features on a partial register."""

    observables: tuple[str, ...] = ("Z0",)
    normalize_observables: bool = False
    observable_register: str = "system"  # "system"/"memory" or "ancilla"/"readout"


class ChannelMapReservoir:
    """Exact expectation-value reservoir using the shared dense QRC model.

    This class tracks only the reduced system density matrix between steps. The
    ancilla is freshly reset inside ExactQRCModel.exact_step_from_system, which
    makes the output deterministic unless optional multinomial shot noise is
    requested.
    """

    def __init__(self, cfg: ChannelMapReservoirConfig):
        self.cfg = cfg
        self.core = ExactQRCModel(cfg)
        self.nS = self.core.nS
        self.nA = self.core.nA
        self.n = self.core.n
        self.rng = np.random.default_rng(cfg.seed)
        self._fixed_point_cache: np.ndarray | None = None
        self._ptm_cache: dict[float, np.ndarray] = {}
        self._memory_basis: tuple[np.ndarray, ...] | None = None
        self._memory_basis_stack: np.ndarray | None = None
        self.reset()

    def reset(self, rhoS0: np.ndarray | None = None) -> None:
        """Reset the memory state before a new stream or message."""

        if rhoS0 is None:
            self.rhoS = self.core.initial_system_density(self.cfg.init_state)
        else:
            self.rhoS = np.asarray(rhoS0, dtype=complex)
        self.rhoSE = np.kron(self.rhoS, self.core.ancilla_reset_density)

    def _memory_channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        return self.channel(u, op_memory)

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        """Apply the induced memory channel to a memory operator."""

        out = self.core.system_channel(float(u), np.asarray(op_memory, dtype=complex))
        if not np.isfinite(out).all():
            raise FloatingPointError("Non-finite output from channel-map memory channel.")
        return out

    def ptm(self, u: float) -> np.ndarray:
        """Return the Pauli transfer matrix of the induced memory channel."""

        key = float(u)
        cached = self._ptm_cache.get(key)
        if cached is not None:
            return cached.copy()

        if self._memory_basis is None or self._memory_basis_stack is None:
            self._memory_basis = _pauli_basis_matrices(self.nS)
            self._memory_basis_stack = np.stack(self._memory_basis, axis=0)
        outputs = np.stack([self.channel(key, basis_op) for basis_op in self._memory_basis], axis=0)
        transfer = np.einsum("mab,nab->mn", self._memory_basis_stack.conj(), outputs, optimize=True) / self.core.dim_system
        if not np.isfinite(transfer).all():
            raise FloatingPointError("Non-finite PTM from channel-map reservoir.")
        self._ptm_cache[key] = transfer
        return transfer.copy()

    def fixed_point(self) -> np.ndarray:
        """Iterate the zero-input channel until a stationary memory state is found."""

        if self.core.control.post_measurement_mode != "reset":
            raise NotImplementedError("fixed_point requires post_measurement_mode='reset'.")
        if self._fixed_point_cache is not None:
            return self._fixed_point_cache.copy()

        rho = self.core.initial_system_density(self.cfg.init_state)
        for _ in range(10000):
            # Symmetrize and renormalize after each application to control tiny
            # dense-linear-algebra drift away from a valid density operator.
            new_rho = self._memory_channel(0.0, rho)
            new_rho = 0.5 * (new_rho + new_rho.conj().T)
            tr = np.trace(new_rho)
            if abs(tr) > 1e-15:
                new_rho /= tr
            if np.linalg.norm(new_rho - rho, ord="fro") < 1e-12:
                self._fixed_point_cache = new_rho.copy()
                return new_rho
            rho = new_rho
        self._fixed_point_cache = rho.copy()
        return rho

    def step(self, u: float) -> np.ndarray:
        """Advance one scalar input and return ancilla probability features."""

        probs, rho_next = self.core.exact_step_from_system(self.rhoS, float(u))
        if self.cfg.use_shot_noise:
            # Keep the same deterministic channel state but expose finite-shot
            # readout noise to downstream classical tasks.
            counts = self.rng.multinomial(self.cfg.shots, probs)
            probs = counts.astype(float) / float(self.cfg.shots)
        self.rhoS = rho_next
        self.rhoSE = np.kron(self.rhoS, self.core.ancilla_reset_density)
        if self.cfg.include_bias:
            return np.concatenate([[1.0], probs])
        return probs

    def run(self, inputs: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Run a full input stream and stack one feature row per time step."""

        x = np.vstack([self.step(float(u)) for u in inputs])
        if not np.isfinite(x).all():
            raise FloatingPointError("Non-finite features from channel-map reservoir.")
        return x

    def run_stream(self, inputs: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        return self.run(inputs)

    def transform(self, inputs: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Scikit-style alias used by the generic experiment API."""

        return self.run_stream(inputs)


class ObservableChannelMapReservoir(ChannelMapReservoir):
    """Exact reset reservoir with explicit partial-register observables.

    The recurrent state is still only the reduced system density matrix. System
    observables are evaluated on the post-measurement reduced memory state,
    matching the memory-observable readout used by the dimension/STM baseline.
    Ancilla/readout observables are evaluated on the pre-measurement ancilla
    marginal, since the ancilla is reset after the measurement protocol.
    """

    def __init__(self, cfg: ObservableChannelMapReservoirConfig):
        self.observable_specs = tuple(str(obs) for obs in cfg.observables)
        if not self.observable_specs:
            raise ValueError("At least one readout observable is required.")
        super().__init__(cfg)
        self.cfg: ObservableChannelMapReservoirConfig
        register = str(cfg.observable_register).lower()
        if register in {"system", "memory"}:
            self.observable_register = "system"
            n_observable_qubits = self.nS
        elif register in {"ancilla", "readout"}:
            self.observable_register = "ancilla"
            n_observable_qubits = self.nA
        else:
            raise ValueError("observable_register must be one of: system, memory, ancilla, readout.")
        self._readout_observables = tuple(
            parse_readout_observable(
                n_observable_qubits,
                spec,
                normalize=bool(cfg.normalize_observables),
            )
            for spec in self.observable_specs
        )

    def get_feature_names(self) -> list[str]:
        names = [f"obs:{spec}" for spec in self.observable_specs]
        if self.cfg.include_bias:
            return ["bias", *names]
        return names

    def step(self, u: float) -> np.ndarray:
        """Advance one scalar input and return configured readout observables."""

        if self.core.control.post_measurement_mode != "reset":
            raise NotImplementedError("ObservableChannelMapReservoir requires post_measurement_mode='reset'.")
        joint = np.kron(self.rhoS, self.core.ancilla_reset_density)
        evolved = self.core.evolve_joint(joint, float(u))
        _, next_joint = self.core.apply_measurement_protocol_exact(evolved)
        rho_system = partial_trace_ancilla(next_joint, self.core.dim_system, self.core.dim_ancilla)
        if self.observable_register == "system":
            observable_state = rho_system
        else:
            observable_state = _partial_trace_system(evolved, self.core.dim_system, self.core.dim_ancilla)
        features = np.asarray(
            [float(np.real_if_close(np.trace(obs @ observable_state))) for obs in self._readout_observables],
            dtype=float,
        )
        self.rhoS = rho_system
        self.rhoSE = np.kron(self.rhoS, self.core.ancilla_reset_density)
        if self.cfg.include_bias:
            return np.concatenate([[1.0], features])
        return features

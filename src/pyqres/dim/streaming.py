"""Streaming task adapter built on the exact dense QRC core."""

from __future__ import annotations

from itertools import combinations, product
from typing import List, Sequence

import numpy as np

from .linalg_utils import ensure_finite
from .pauli import pauli_string

try:
    from pyqres.simulation.exact_qrc import ExactQRCModel, ExactQRCModelConfig, partial_trace_ancilla
except Exception as exc:  # pragma: no cover
    ExactQRCModel = None  # type: ignore
    ExactQRCModelConfig = None  # type: ignore
    partial_trace_ancilla = None  # type: ignore
    _QRCLIB_STREAM_IMPORT_ERROR = exc
else:  # pragma: no cover
    _QRCLIB_STREAM_IMPORT_ERROR = None


class SharedExactStreamingReservoir:
    """Task-side streaming adapter over pyqres' shared exact dense QRC core.

    Readout modes:
    - ancilla_probs: emit ancilla outcome probabilities after the measurement protocol
    - memory_observables: emit expectation values of chosen system observables
    """

    def __init__(
        self,
        config: "ExactQRCModelConfig | None" = None,
        core: "ExactQRCModel | None" = None,
        readout_mode: str = "ancilla_probs",
        include_bias: bool = True,
        use_shot_noise: bool = False,
        shots: int = 4096,
        init_state: str = "maximally_mixed",
        observable_preset: str = "z",
        custom_observables: Sequence[str] | None = None,
    ):
        if ExactQRCModel is None:
            raise ImportError("pyqres.simulation must be importable to use SharedExactStreamingReservoir.") from _QRCLIB_STREAM_IMPORT_ERROR
        if core is None and config is None:
            raise ValueError("Provide either an ExactQRCModelConfig or an ExactQRCModel instance.")
        self.core = core if core is not None else ExactQRCModel(config)  # type: ignore[arg-type]
        self.readout_mode = str(readout_mode)
        self.include_bias = bool(include_bias)
        self.use_shot_noise = bool(use_shot_noise)
        self.shots = int(shots)
        self.init_state = str(init_state)
        self.observable_preset = str(observable_preset)
        self.custom_observables = list(custom_observables) if custom_observables is not None else []
        self.rng = np.random.default_rng(self.core.cfg.seed)
        self.observables = self.default_memory_observables(self.observable_preset, self.custom_observables)
        self.reset()

    def reset(self, rhoS0: np.ndarray | None = None) -> None:
        """Reset the joint state before processing one stream/message."""

        if rhoS0 is None:
            rho_system = self.core.initial_system_density(self.init_state)
        else:
            rho_system = np.asarray(rhoS0, dtype=complex)
        # The dense exact core evolves the joint system state, so the streaming
        # wrapper keeps the memory state tensored with a freshly reset ancilla.
        self.rho_joint = np.kron(rho_system, self.core.ancilla_reset_density)

    def parse_memory_observable(self, spec: str) -> np.ndarray:
        """Parse observable specs such as Z0 or X0*Z2 on memory qubits."""

        cleaned = spec.replace(" ", "")
        if not cleaned:
            raise ValueError("Observable spec must be non-empty")
        factors = []
        for token in cleaned.split("*"):
            pauli = token[0].upper()
            if pauli not in {"X", "Y", "Z"}:
                raise ValueError(f"Unsupported Pauli observable token '{token}'")
            site = int(token[1:])
            if not (0 <= site < self.core.nS):
                raise ValueError(f"Observable token '{token}' is out of range for n_system={self.core.nS}")
            factors.append((site, pauli))
        return pauli_string(self.core.nS, tuple(sorted(factors)))

    def _single_site_specs(self, paulis: Sequence[str]) -> List[str]:
        return [f"{pauli}{site}" for pauli in paulis for site in range(self.core.nS)]

    def _pair_specs(self, paulis_left: Sequence[str], paulis_right: Sequence[str]) -> List[str]:
        specs: List[str] = []
        for left_site, right_site in combinations(range(self.core.nS), 2):
            for left_pauli, right_pauli in product(paulis_left, paulis_right):
                specs.append(f"{left_pauli}{left_site}*{right_pauli}{right_site}")
        return specs

    def default_memory_observable_specs(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> List[str]:
        """Return named observable presets used by task-side readout modes."""

        preset_key = preset.lower()
        if preset_key == "z":
            obs_specs = [f"Z{i}" for i in range(self.core.nS)]
        elif preset_key == "x":
            obs_specs = [f"X{i}" for i in range(self.core.nS)]
        elif preset_key == "y":
            obs_specs = [f"Y{i}" for i in range(self.core.nS)]
        elif preset_key == "xy":
            obs_specs = self._single_site_specs(("X", "Y"))
        elif preset_key == "zx":
            obs_specs = [f"Z{i}" for i in range(self.core.nS)] + [f"X{i}" for i in range(self.core.nS)]
        elif preset_key == "xyz":
            obs_specs = self._single_site_specs(("X", "Y", "Z"))
        elif preset_key == "zz_pairs":
            obs_specs = self._pair_specs(("Z",), ("Z",))
        elif preset_key == "pair_xyz":
            obs_specs = self._pair_specs(("X", "Y", "Z"), ("X", "Y", "Z"))
        elif preset_key == "rich":
            obs_specs = self._single_site_specs(("X", "Y", "Z")) + self._pair_specs(("X", "Y", "Z"), ("X", "Y", "Z"))
        elif preset_key == "custom":
            obs_specs = []
        else:
            raise ValueError(f"Unsupported observable preset '{preset}'")
        if custom_specs:
            obs_specs.extend(custom_specs)
        return list(dict.fromkeys(obs_specs))

    def default_memory_observables(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> List[np.ndarray]:
        return [self.parse_memory_observable(spec) for spec in self.default_memory_observable_specs(preset, custom_specs)]

    def _memory_observable_features(self, rho_joint_next: np.ndarray) -> np.ndarray:
        """Evaluate configured memory observables on the reduced memory state."""

        rho_system = partial_trace_ancilla(rho_joint_next, self.core.dim_system, self.core.dim_ancilla)
        # Readout in this mode happens after tracing out the ancilla, so the
        # emitted features are ordinary expectation values on the memory system.
        values = [float(np.real_if_close(np.trace(obs @ rho_system))) for obs in self.observables]
        return np.asarray(values, dtype=float)

    def _ancilla_features(self, probs: np.ndarray) -> np.ndarray:
        """Convert ancilla outcome probabilities into the requested feature row."""

        probs = np.asarray(probs, dtype=float)
        if self.use_shot_noise:
            if self.readout_mode != "ancilla_probs":
                raise NotImplementedError("Shot-noise emulation is currently supported only for ancilla_probs readout.")
            counts = self.rng.multinomial(self.shots, probs)
            probs = counts.astype(float) / float(self.shots)
        return probs

    def step(self, u: float) -> np.ndarray:
        """Advance one scalar input and emit the configured feature vector."""

        evolved = self.core.evolve_joint(self.rho_joint, float(u))
        probs, rho_joint_next = self.core.apply_measurement_protocol_exact(evolved)
        # The post-measurement joint state becomes the next recurrent state.
        self.rho_joint = rho_joint_next

        if self.readout_mode == "ancilla_probs":
            feats = self._ancilla_features(probs)
        elif self.readout_mode == "memory_observables":
            feats = self._memory_observable_features(rho_joint_next)
        else:
            raise ValueError(f"Unsupported readout_mode '{self.readout_mode}'")

        if self.include_bias:
            # A leading bias feature keeps the streaming output compatible with
            # standard linear readouts used elsewhere in the experiments.
            return np.concatenate([[1.0], np.asarray(feats, dtype=float)])
        return np.asarray(feats, dtype=float)

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        """Run a full input stream and stack one feature row per step."""

        x = np.vstack([self.step(float(u)) for u in inputs])
        if not np.isfinite(x).all():
            raise FloatingPointError("Non-finite features from shared exact streaming reservoir.")
        return x


class MemoryObservableStreamingReservoir:
    """Stream task features from any pyqres dimension-analysis reservoir model.

    The adapter applies the model's memory channel at each scalar input and emits
    expectation values of chosen memory observables. It is useful when a task
    experiment should use the same reservoir model and observables as the
    Volterra visibility analysis.
    """

    def __init__(
        self,
        model: object,
        observables: Sequence[np.ndarray],
        include_bias: bool = True,
        init_state: str = "zero",
    ):
        self.model = model
        self.observables = [np.asarray(observable, dtype=complex) for observable in observables]
        self.include_bias = bool(include_bias)
        self.init_state = str(init_state)
        self.reset()

    def _initial_density(self) -> np.ndarray:
        dim_memory = int(getattr(self.model, "dim_memory"))
        if self.init_state == "zero":
            rho = np.zeros((dim_memory, dim_memory), dtype=complex)
            rho[0, 0] = 1.0
            return rho
        if self.init_state == "fixed_point":
            return np.asarray(self.model.fixed_point(), dtype=complex)
        if self.init_state == "maximally_mixed":
            return np.eye(dim_memory, dtype=complex) / float(dim_memory)
        raise ValueError("init_state must be one of: zero, fixed_point, maximally_mixed")

    def reset(self) -> None:
        """Reset the memory state before running one input stream."""

        self.rho = ensure_finite("initial memory-observable density", self._initial_density())

    def step(self, u: float) -> np.ndarray:
        """Advance one scalar input and return one feature row."""

        rho_next = self.model.channel(float(u), self.rho)
        rho_next = 0.5 * (rho_next + rho_next.conj().T)
        trace = np.trace(rho_next)
        if abs(trace) > 1e-15:
            rho_next = rho_next / trace
        self.rho = ensure_finite("memory-observable streaming density", rho_next)

        features = np.asarray(
            [float(np.real_if_close(np.trace(observable @ self.rho))) for observable in self.observables],
            dtype=float,
        )
        if self.include_bias:
            features = np.concatenate([[1.0], features])
        return ensure_finite("memory-observable streaming features", features)

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        """Run a full input stream and stack one feature row per step."""

        self.reset()
        features = np.vstack([self.step(float(u)) for u in inputs])
        return ensure_finite("memory-observable streaming feature matrix", features)


__all__ = [
    "ExactQRCModel",
    "ExactQRCModelConfig",
    "MemoryObservableStreamingReservoir",
    "SharedExactStreamingReservoir",
]

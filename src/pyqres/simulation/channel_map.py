"""Deterministic exact reservoir frontend for classical feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_qrc import ExactQRCModel, ExactQRCModelConfig


@dataclass
class ChannelMapReservoirConfig(ExactQRCModelConfig):
    """Configuration for expectation-value features from the exact channel."""

    include_bias: bool = True
    use_shot_noise: bool = False
    shots: int = 4096
    init_state: str = "maximally_mixed"  # "maximally_mixed" or "zero"


class ChannelMapReservoir:
    """Exact expectation-value reservoir using the shared dense QRC model.

    This class tracks only the reduced system density matrix between steps. The
    ancilla is freshly reset inside `ExactQRCModel.exact_step_from_system`, which
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
        self.reset()

    def reset(self, rhoS0: np.ndarray | None = None) -> None:
        """Reset the memory state before a new stream or message."""

        if rhoS0 is None:
            self.rhoS = self.core.initial_system_density(self.cfg.init_state)
        else:
            self.rhoS = np.asarray(rhoS0, dtype=complex)
        self.rhoSE = np.kron(self.rhoS, self.core.ancilla_reset_density)

    def _memory_channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        return self.core.system_channel(float(u), np.asarray(op_memory, dtype=complex))

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

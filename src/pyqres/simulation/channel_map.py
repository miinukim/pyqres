from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_qrc import ExactQRCModel, ExactQRCModelConfig


@dataclass
class ChannelMapReservoirConfig(ExactQRCModelConfig):
    include_bias: bool = True
    use_shot_noise: bool = False
    shots: int = 4096
    init_state: str = "maximally_mixed"  # "maximally_mixed" or "zero"


class ChannelMapReservoir:
    """Exact expectation-value reservoir using the shared dense QRC model."""

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
        if rhoS0 is None:
            self.rhoS = self.core.initial_system_density(self.cfg.init_state)
        else:
            self.rhoS = np.asarray(rhoS0, dtype=complex)
        self.rhoSE = np.kron(self.rhoS, self.core.ancilla_reset_density)

    def _memory_channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        return self.core.system_channel(float(u), np.asarray(op_memory, dtype=complex))

    def fixed_point(self) -> np.ndarray:
        if self.core.control.post_measurement_mode != "reset":
            raise NotImplementedError("fixed_point requires post_measurement_mode='reset'.")
        if self._fixed_point_cache is not None:
            return self._fixed_point_cache.copy()

        rho = self.core.initial_system_density(self.cfg.init_state)
        for _ in range(10000):
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
        probs, rho_next = self.core.exact_step_from_system(self.rhoS, float(u))
        if self.cfg.use_shot_noise:
            counts = self.rng.multinomial(self.cfg.shots, probs)
            probs = counts.astype(float) / float(self.cfg.shots)
        self.rhoS = rho_next
        self.rhoSE = np.kron(self.rhoS, self.core.ancilla_reset_density)
        if self.cfg.include_bias:
            return np.concatenate([[1.0], probs])
        return probs

    def run(self, inputs: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        x = np.vstack([self.step(float(u)) for u in inputs])
        if not np.isfinite(x).all():
            raise FloatingPointError("Non-finite features from channel-map reservoir.")
        return x

    def run_stream(self, inputs: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        return self.run(inputs)

"""Trajectory-style exact reservoir frontend.

Unlike `ChannelMapReservoir`, this class samples measurement outcomes and
evolves each shot as a branch trajectory. It is still powered by the dense exact
core, but its feature estimates look like hardware counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .exact_qrc import ExactQRCModel, ExactQRCModelConfig


@dataclass
class HardwareTrajectoryReservoirConfig(ExactQRCModelConfig):
    """Configuration for finite-shot trajectory features."""

    include_bias: bool = True
    init_state: str = "zero"  # "maximally_mixed" or "zero"
    shots: int = 1024


class HardwareTrajectoryReservoir:
    """Shot-trajectory emulator built on the shared dense QRC model."""

    def __init__(self, cfg: HardwareTrajectoryReservoirConfig):
        self.cfg = cfg
        self.core = ExactQRCModel(cfg)
        self.nS = self.core.nS
        self.nA = self.core.nA
        self.n = self.core.n
        self.rng = np.random.default_rng(cfg.seed)
        if cfg.shots <= 0:
            raise ValueError("shots must be > 0.")

    def run(self, inputs: Sequence[float]) -> np.ndarray:
        """Sample `shots` independent trajectories over the same input stream."""

        counts = np.zeros((len(inputs), self.core.dim_ancilla), dtype=int)
        for _ in range(int(self.cfg.shots)):
            # Each shot starts from the requested initial joint state and then
            # follows its own sampled measurement branch through time.
            rho_joint = self.core.initial_joint_density(self.cfg.init_state)
            for t, u in enumerate(inputs):
                rho_joint = self.core.evolve_joint(rho_joint, float(u))
                outcome, rho_joint = self.core.sample_measurement_protocol(rho_joint, self.rng)
                counts[t, outcome] += 1
        probs = counts.astype(float) / float(self.cfg.shots)
        if self.cfg.include_bias:
            return np.hstack([np.ones((len(inputs), 1)), probs])
        return probs

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        x = self.run(inputs)
        if not np.isfinite(x).all():
            raise FloatingPointError("Non-finite features from hardware trajectory reservoir.")
        return x

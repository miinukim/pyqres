from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ReservoirStepResult:
    """One reservoir step result in the unified pyqres interface."""

    features: np.ndarray
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReservoirRunResult:
    """Batch reservoir run result in the unified pyqres interface."""

    features: np.ndarray
    metadata: dict[str, Any] | None = None


@runtime_checkable
class QRCReservoirProtocol(Protocol):
    """Minimal protocol shared by exact, Qiskit, and trajectory reservoirs."""

    def reset(self) -> None:
        ...

    def step(self, u: float) -> np.ndarray:
        ...

    def run(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class ChannelReservoirProtocol(QRCReservoirProtocol, Protocol):
    """Reservoir protocol for models that expose their induced memory channel."""

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        ...

    def ptm(self, u: float) -> np.ndarray:
        ...


@runtime_checkable
class CircuitReservoirProtocol(QRCReservoirProtocol, Protocol):
    """Reservoir protocol for implementations that can emit Qiskit circuits."""

    def circuit(self, inputs: Sequence[float] | np.ndarray, **kwargs: Any) -> Any:
        ...

from __future__ import annotations

"""Fluent reservoir construction API."""

from dataclasses import replace
from typing import Any, Sequence

from pyqres.core.builders import compile_reservoir
from pyqres.core.protocols import ReservoirBuilderProtocol
from pyqres.core.specs import ReservoirSpec


class ReservoirBuilder(ReservoirBuilderProtocol):
    """Chainable builder for the common interactive pyqres workflow."""

    def __init__(self, family: str = "ising", spec: ReservoirSpec | None = None):
        self._spec = spec if spec is not None else ReservoirSpec(family=family)
        self._backend = "exact"

    @property
    def spec(self) -> ReservoirSpec:
        """Return the current immutable reservoir spec."""

        return self._spec

    def memory_qubits(self, n_qubits: int) -> "ReservoirBuilder":
        """Set the number of recurrent memory/system qubits."""

        self._spec = replace(self._spec, n_memory=int(n_qubits), n_system=int(n_qubits))
        return self

    def readout_qubits(self, n_qubits: int) -> "ReservoirBuilder":
        """Set the number of readout/ancilla qubits."""

        self._spec = replace(self._spec, n_readout=int(n_qubits), n_ancilla=int(n_qubits))
        return self

    def seed(self, value: int) -> "ReservoirBuilder":
        """Set the reservoir random seed."""

        self._spec = replace(self._spec, seed=int(value))
        return self

    def input(
        self,
        axis: str = "Z",
        *,
        site: int = 0,
        sites: Sequence[int] | None = None,
        strength: float = 1.0,
        on_memory: bool = True,
        scale: float | None = None,
        normalization: str = "none",
    ) -> "ReservoirBuilder":
        """Configure scalar input encoding for supported Ising reservoirs."""

        axis_key = str(axis).upper()
        if axis_key not in {"X", "Y", "Z"}:
            raise ValueError("input axis must be one of X, Y, or Z")

        model_kwargs = dict(self._spec.model_kwargs)
        model_kwargs.update(
            {
                "input_axis": axis_key,
                "input_strength": float(strength),
                "input_on_memory": bool(on_memory),
                "input_site": int(site),
                "input_strength_normalization": str(normalization),
            }
        )
        if sites is not None:
            model_kwargs["input_sites"] = tuple(int(item) for item in sites)

        hamiltonian_kwargs = dict(self._spec.hamiltonian_kwargs)
        if axis_key == "Z":
            hamiltonian_kwargs["input_z_field_base"] = float(strength)
            hamiltonian_kwargs.setdefault("input_z_field_std", 0.0)
            hamiltonian_kwargs["input_z_field_scale"] = 1.0 if scale is None else float(scale)

        self._spec = replace(
            self._spec,
            input_scale=1.0 if scale is None else float(scale),
            model_kwargs=model_kwargs,
            hamiltonian_kwargs=hamiltonian_kwargs,
        )
        return self

    def evolution(self, *, tau: float | None = None, **kwargs: Any) -> "ReservoirBuilder":
        """Configure reservoir dynamics parameters."""

        updates: dict[str, Any] = {}
        model_kwargs = dict(self._spec.model_kwargs)
        hamiltonian_kwargs = dict(self._spec.hamiltonian_kwargs)
        if tau is not None:
            updates["tau"] = float(tau)
        for key, value in kwargs.items():
            model_kwargs[key] = value
            hamiltonian_kwargs[key] = value
        self._spec = replace(self._spec, **updates, model_kwargs=model_kwargs, hamiltonian_kwargs=hamiltonian_kwargs)
        return self

    def observables(
        self,
        preset: str | Sequence[str] = "z",
        *,
        count: int | None = None,
        custom: Sequence[str] | None = None,
        include_bias: bool = True,
        init_state: str = "zero",
    ) -> "ReservoirBuilder":
        """Use memory-observable features for reservoir readout."""

        readout = replace(
            self._spec.readout,
            mode="memory_observables",
            observables=preset,
            count=count,
            custom=tuple(custom or ()),
            include_bias=bool(include_bias),
            init_state=str(init_state),
        )
        self._spec = replace(self._spec, readout=readout)
        return self

    def ancilla_probabilities(
        self,
        *,
        include_bias: bool = True,
        init_state: str = "maximally_mixed",
        shot_noise: bool = False,
        shots: int = 4096,
    ) -> "ReservoirBuilder":
        """Use ancilla outcome probabilities as reservoir features."""

        readout = replace(
            self._spec.readout,
            mode="ancilla_probs",
            include_bias=bool(include_bias),
            init_state=str(init_state),
            use_shot_noise=bool(shot_noise),
            shots=int(shots),
        )
        self._spec = replace(self._spec, readout=readout)
        return self

    def model(self, **kwargs: Any) -> "ReservoirBuilder":
        """Set backend model parameters directly."""

        model_kwargs = dict(self._spec.model_kwargs)
        model_kwargs.update(kwargs)
        self._spec = replace(self._spec, model_kwargs=model_kwargs)
        return self

    def hamiltonian(self, **kwargs: Any) -> "ReservoirBuilder":
        """Set dense/Qiskit Hamiltonian generator parameters directly."""

        hamiltonian_kwargs = dict(self._spec.hamiltonian_kwargs)
        hamiltonian_kwargs.update(kwargs)
        self._spec = replace(self._spec, hamiltonian_kwargs=hamiltonian_kwargs)
        return self

    def backend(self, name: str = "exact") -> Any:
        """Compile and return an executable reservoir."""

        self._backend = str(name)
        backend = self._backend
        if backend.lower() in {"exact", "dense"} and self._spec.readout.mode == "memory_observables":
            backend = "memory_observable"
        return compile_reservoir(self._spec, backend=backend)

    def build(self, backend: str | None = None) -> Any:
        """Compile and return an executable reservoir."""

        return self.backend(self._backend if backend is None else backend)


def reservoir(family: str = "ising") -> ReservoirBuilder:
    """Start a chainable reservoir construction."""

    return ReservoirBuilder(family=family)

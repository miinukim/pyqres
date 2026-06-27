from __future__ import annotations

"""Dictionary-first reservoir construction API."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pyqres.core.builders import compile_reservoir
from pyqres.core.protocols import ReservoirBuilderProtocol
from pyqres.core.specs import DynamicsSpec, InputEncodingSpec, ReadoutSpec, ReservoirSpec


@dataclass
class ReservoirBuilder(ReservoirBuilderProtocol):
    """Minimal inspectable builder returned by qresreservoir.builder_from_dict."""

    _spec: ReservoirSpec
    _backend: str = "exact"

    @property
    def spec(self) -> ReservoirSpec:
        """Return the current immutable reservoir spec."""

        return self._spec

    def backend(self, name: str = "exact") -> Any:
        """Compile and return an executable reservoir."""

        self._backend = str(name)
        return compile_reservoir(self._spec, backend=self._backend)

    def build(self, backend: str | None = None) -> Any:
        """Compile and return an executable reservoir."""

        return self.backend(self._backend if backend is None else backend)


def _pop_any(raw: dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw.pop(name)
    return default


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


_HAMILTONIAN_KEYS = {
    "h0",
    "h1",
    "H0",
    "H1",
    "h0_terms",
    "h1_terms",
    "h0_matrix",
    "h1_matrix",
    "H0_matrix",
    "H1_matrix",
    "H0_hamiltonian",
    "H1_hamiltonian",
}


def _looks_like_quantum_circuit(value: Any) -> bool:
    return hasattr(value, "num_qubits") and hasattr(value, "to_instruction")


def _looks_like_existing_reservoir(value: Any) -> bool:
    return any(hasattr(value, name) for name in ("transform", "run_stream", "run", "step"))


def _infer_dynamics(value: Any, *, default_preset: str) -> tuple[DynamicsSpec | None, dict[str, Any]]:
    """Infer a DynamicsSpec from a user-supplied object or mapping.

    Runtime objects such as QuantumCircuit instances are returned in the
    auxiliary runtime mapping instead of being embedded into the serializable
    spec parameters.
    """

    if value is None:
        return None, {}
    if isinstance(value, DynamicsSpec):
        return value, {}
    if _looks_like_quantum_circuit(value):
        return DynamicsSpec(kind="circuit"), {"circuit": value}
    if _looks_like_existing_reservoir(value):
        return DynamicsSpec(kind="object"), {"reservoir": value}
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return DynamicsSpec(kind="hamiltonian", parameters={"h0": value[0], "h1": value[1]}), {}
    if isinstance(value, Mapping):
        raw = dict(value)
        circuit = _pop_any(raw, ("circuit",), None)
        if circuit is not None:
            return DynamicsSpec(kind=str(raw.pop("kind", "circuit")), parameters=raw), {"circuit": circuit}
        reservoir_obj = _pop_any(raw, ("reservoir",), None)
        if reservoir_obj is not None:
            return DynamicsSpec(kind=str(raw.pop("kind", "object")), parameters=raw), {"reservoir": reservoir_obj}
        if "kind" in raw or "name" in raw or "preset" in raw or "family" in raw:
            return DynamicsSpec.from_mapping(raw), {}
        if _HAMILTONIAN_KEYS.intersection(raw):
            return DynamicsSpec(kind="hamiltonian", parameters=raw), {}
        return DynamicsSpec(kind="preset", name=default_preset, parameters=raw), {}
    raise TypeError("dynamics must be a mapping, DynamicsSpec, QuantumCircuit-like object, existing reservoir object, or a two-item (h0, h1) Hamiltonian pair")


def _readout_from_config(value: Any) -> ReadoutSpec:
    readout_cfg = _as_mapping(value, name="readout")

    mode = str(readout_cfg.pop("mode", readout_cfg.pop("type", "memory_observables"))).lower()
    if mode in {"ancilla", "ancilla_probs", "ancilla_probabilities", "probabilities"}:
        readout = ReadoutSpec(
            mode="ancilla_probs",
            include_bias=bool(readout_cfg.pop("include_bias", True)),
            init_state=str(readout_cfg.pop("init_state", "maximally_mixed")),
            use_shot_noise=bool(readout_cfg.pop("shot_noise", readout_cfg.pop("use_shot_noise", False))),
            shots=int(readout_cfg.pop("shots", 4096)),
        )
        if readout_cfg:
            raise ValueError(f"Unknown readout fields: {sorted(readout_cfg)}")
        return readout

    observable_config = readout_cfg.pop("observables", "z")
    if isinstance(observable_config, Mapping):
        obs = dict(observable_config)
        preset = obs.pop("preset", obs.pop("name", "z"))
        count = obs.pop("count", readout_cfg.pop("count", None))
        custom = obs.pop("custom", readout_cfg.pop("custom", None))
        if obs:
            raise ValueError(f"Unknown observables fields: {sorted(obs)}")
    else:
        preset = observable_config
        count = readout_cfg.pop("count", None)
        custom = readout_cfg.pop("custom", None)

    readout = ReadoutSpec(
        mode="memory_observables",
        observables=preset,
        count=None if count is None else int(count),
        custom=tuple(custom or ()),
        include_bias=bool(readout_cfg.pop("include_bias", True)),
        init_state=str(readout_cfg.pop("init_state", "zero")),
    )
    if readout_cfg:
        raise ValueError(f"Unknown readout fields: {sorted(readout_cfg)}")
    return readout


def _promoted_spec_updates(parameters: Mapping[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for source, target in {
        "n_memory": "n_memory",
        "n_readout": "n_readout",
        "n_system": "n_system",
        "n_ancilla": "n_ancilla",
        "tau": "tau",
        "input_scale": "input_scale",
        "seed": "seed",
    }.items():
        if source in parameters:
            updates[target] = parameters[source]
    return updates


def _spec_from_parts(
    *,
    preset_name: str,
    n_memory: Any,
    n_readout: Any,
    seed: Any,
    tau: Any,
    encoding: InputEncodingSpec,
    dynamics: DynamicsSpec | None,
    runtime: Mapping[str, Any],
    readout: ReadoutSpec,
    model_kwargs: Mapping[str, Any],
    qiskit_kwargs: Mapping[str, Any],
) -> ReservoirSpec:
    dynamics = dynamics or DynamicsSpec(kind="preset", name=preset_name)
    kind = dynamics.kind.lower()
    parameters = dict(dynamics.parameters)
    updates = _promoted_spec_updates(parameters)
    if n_memory is not None:
        updates.update({"n_memory": int(n_memory), "n_system": int(n_memory)})
    if n_readout is not None:
        updates.update({"n_readout": int(n_readout), "n_ancilla": int(n_readout)})
    if seed is not None:
        updates["seed"] = int(seed)
    if tau is not None:
        updates["tau"] = float(tau)

    spec_kwargs: dict[str, Any] = {
        "family": str(dynamics.name or preset_name) if kind == "preset" else preset_name,
        "preset": str(dynamics.name or preset_name) if kind == "preset" else None,
        "source_kind": kind,
        "encoding": encoding,
        "dynamics": dynamics,
        "readout": readout,
        "runtime": dict(runtime),
        "qiskit_kwargs": dict(qiskit_kwargs),
        **updates,
    }

    if kind == "preset":
        spec_kwargs["source_kind"] = "preset"
        spec_kwargs["model_kwargs"] = {**parameters, **dict(model_kwargs)}
        spec_kwargs["hamiltonian_kwargs"] = dict(parameters)
    elif kind == "hamiltonian":
        spec_kwargs["hamiltonian_kwargs"] = parameters
    elif kind == "circuit":
        if "circuit" not in runtime:
            raise ValueError("circuit dynamics requires a QuantumCircuit-like object")
        spec_kwargs["circuit_kwargs"] = parameters
    elif kind == "object":
        if "reservoir" not in runtime:
            raise ValueError("object dynamics requires a reservoir object")
    else:
        spec_kwargs["model_kwargs"] = {**parameters, **dict(model_kwargs)}

    return ReservoirSpec(**spec_kwargs)


class qresreservoir:
    """Dictionary-first reservoir factory.

    This is a compact facade over the core compiler. It accepts plain Python
    dictionaries and returns compiled reservoirs by default.
    """

    @classmethod
    def builder_from_dict(cls, config: Mapping[str, Any]) -> ReservoirBuilder:
        """Build a ReservoirBuilder from a plain mapping without compiling it."""

        raw = _as_mapping(config, name="reservoir config")
        raw_dynamics_input = _pop_any(raw, ("dynamics",), None)
        dynamics, dynamics_runtime = _infer_dynamics(raw_dynamics_input, default_preset="ising")
        raw_preset = _pop_any(raw, ("preset",), None)
        if raw_preset is None and dynamics is not None and dynamics.kind.lower() == "preset":
            raw_preset = dynamics.name
        preset_name = str(raw_preset or "ising").lower()
        backend_name = str(_pop_any(raw, ("backend",), "exact"))

        n_memory = _pop_any(raw, ("memory_qubits", "n_memory", "n_system", "system_qubits"), None)
        n_readout = _pop_any(raw, ("readout_qubits", "n_readout", "n_ancilla", "ancilla_qubits"), None)
        seed = _pop_any(raw, ("seed",), None)

        encoding_config = _pop_any(raw, ("encoding",), None)
        encoding = InputEncodingSpec.from_mapping(encoding_config)

        tau = _pop_any(raw, ("tau",), None)
        model_cfg = _as_mapping(_pop_any(raw, ("model_kwargs", "model_params", "model_config"), None), name="model_kwargs")
        qiskit_cfg = _as_mapping(_pop_any(raw, ("qiskit", "qiskit_kwargs", "simulator"), None), name="qiskit")
        readout = _readout_from_config(_pop_any(raw, ("readout",), None))

        if raw:
            raise ValueError(f"Unknown reservoir config fields: {sorted(raw)}")
        spec = _spec_from_parts(
            preset_name=preset_name,
            n_memory=n_memory,
            n_readout=n_readout,
            seed=seed,
            tau=tau,
            encoding=encoding,
            dynamics=dynamics,
            runtime=dynamics_runtime,
            readout=readout,
            model_kwargs=model_cfg,
            qiskit_kwargs=qiskit_cfg,
        )
        return ReservoirBuilder(spec, _backend=backend_name)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Any:
        """Compile a reservoir from a plain Python dictionary."""

        builder = cls.builder_from_dict(config)
        return builder.build()

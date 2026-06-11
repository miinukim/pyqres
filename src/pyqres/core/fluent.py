from __future__ import annotations

"""Fluent reservoir construction API."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

from pyqres.core.builders import compile_reservoir
from pyqres.core.protocols import ReservoirBuilderProtocol
from pyqres.core.specs import ReservoirSpec


class ReservoirBuilder(ReservoirBuilderProtocol):
    """Chainable builder for the common interactive pyqres workflow."""

    def __init__(self, family: str = "ising", spec: ReservoirSpec | None = None):
        self._spec = spec if spec is not None else ReservoirSpec(family=family, preset=family, source_kind="preset")
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

    def preset(self, name: str, **kwargs: Any) -> "ReservoirBuilder":
        """Select a built-in reservoir preset and optionally set its parameters."""

        spec_updates: dict[str, Any] = {
            "family": str(name),
            "preset": str(name),
            "source_kind": "preset",
            "runtime": {},
        }
        for source, target in {
            "n_memory": "n_memory",
            "n_readout": "n_readout",
            "n_system": "n_system",
            "n_ancilla": "n_ancilla",
            "tau": "tau",
            "input_scale": "input_scale",
            "seed": "seed",
        }.items():
            if source in kwargs:
                spec_updates[target] = kwargs.pop(source)
        model_kwargs = dict(self._spec.model_kwargs)
        hamiltonian_kwargs = dict(self._spec.hamiltonian_kwargs)
        model_kwargs.update(kwargs)
        hamiltonian_kwargs.update(kwargs)
        self._spec = replace(
            self._spec,
            **spec_updates,
            model_kwargs=model_kwargs,
            hamiltonian_kwargs=hamiltonian_kwargs,
        )
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

    def hamiltonian(self, *args: Any, **kwargs: Any) -> "ReservoirBuilder":
        """Use an explicit Hamiltonian as the reservoir dynamics.

        Examples:
        - hamiltonian(h0=H0, h1=H1)
        - hamiltonian(h0_terms=[...], h1_terms=[...])
        - hamiltonian(kind="pauli_terms", h0_terms=[...])
        """

        if args:
            if len(args) > 2:
                raise TypeError("hamiltonian accepts at most positional h0 and h1 arguments")
            kwargs.setdefault("h0", args[0])
            if len(args) == 2:
                kwargs.setdefault("h1", args[1])
        hamiltonian_kwargs = dict(self._spec.hamiltonian_kwargs)
        hamiltonian_kwargs.update(kwargs)
        self._spec = replace(
            self._spec,
            source_kind="hamiltonian",
            hamiltonian_kwargs=hamiltonian_kwargs,
            runtime={},
        )
        return self

    def circuit(self, circuit: Any, **kwargs: Any) -> "ReservoirBuilder":
        """Use a user-supplied quantum circuit as the repeated reservoir block."""

        circuit_kwargs = dict(self._spec.circuit_kwargs)
        circuit_kwargs.update(kwargs)
        self._backend = "qiskit"
        self._spec = replace(
            self._spec,
            source_kind="circuit",
            circuit_kwargs=circuit_kwargs,
            runtime={"circuit": circuit},
        )
        return self

    def use(self, reservoir: Any) -> "ReservoirBuilder":
        """Use an already constructed reservoir object directly."""

        self._spec = replace(
            self._spec,
            source_kind="object",
            runtime={"reservoir": reservoir},
        )
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


def _apply_input_config(builder: ReservoirBuilder, input_config: Any) -> ReservoirBuilder:
    if input_config is None:
        return builder
    configs = input_config if isinstance(input_config, list) else [input_config]
    for item in configs:
        raw = _as_mapping(item, name="input")
        axis = raw.pop("axis", raw.pop("operator", raw.pop("pauli", "Z")))
        builder.input(
            str(axis),
            site=int(raw.pop("site", 0)),
            sites=raw.pop("sites", None),
            strength=float(raw.pop("strength", raw.pop("scale_strength", 1.0))),
            on_memory=bool(raw.pop("on_memory", True)),
            scale=raw.pop("scale", None),
            normalization=str(raw.pop("normalization", "none")),
        )
        if raw:
            raise ValueError(f"Unknown input fields: {sorted(raw)}")
    return builder


def _apply_readout_config(builder: ReservoirBuilder, cfg: Mapping[str, Any]) -> ReservoirBuilder:
    readout_cfg = _as_mapping(cfg.get("readout"), name="readout")
    observables_cfg = cfg.get("observables")
    if observables_cfg is not None and "observables" not in readout_cfg:
        readout_cfg["observables"] = observables_cfg

    mode = str(readout_cfg.pop("mode", readout_cfg.pop("type", "memory_observables"))).lower()
    if mode in {"ancilla", "ancilla_probs", "ancilla_probabilities", "probabilities"}:
        return builder.ancilla_probabilities(
            include_bias=bool(readout_cfg.pop("include_bias", True)),
            init_state=str(readout_cfg.pop("init_state", "maximally_mixed")),
            shot_noise=bool(readout_cfg.pop("shot_noise", readout_cfg.pop("use_shot_noise", False))),
            shots=int(readout_cfg.pop("shots", 4096)),
        )

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

    return builder.observables(
        preset,
        count=None if count is None else int(count),
        custom=custom,
        include_bias=bool(readout_cfg.pop("include_bias", True)),
        init_state=str(readout_cfg.pop("init_state", "zero")),
    )


class qresreservoir:
    """Dictionary-first reservoir factory.

    This is a compact facade over the fluent builder. It accepts plain Python
    dictionaries and returns compiled reservoirs by default.
    """

    @classmethod
    def builder_from_dict(cls, config: Mapping[str, Any]) -> ReservoirBuilder:
        """Build a ReservoirBuilder from a plain mapping without compiling it."""

        raw = _as_mapping(config, name="reservoir config")
        preset_name = str(_pop_any(raw, ("preset", "family", "type", "model"), "ising")).lower()
        backend_name = str(_pop_any(raw, ("backend",), "exact"))

        n_memory = _pop_any(raw, ("memory_qubits", "n_memory", "n_system", "system_qubits"), None)
        n_readout = _pop_any(raw, ("readout_qubits", "n_readout", "n_ancilla", "ancilla_qubits"), None)
        seed = _pop_any(raw, ("seed",), None)

        builder = reservoir(preset_name)
        if n_memory is not None:
            builder.memory_qubits(int(n_memory))
        if n_readout is not None:
            builder.readout_qubits(int(n_readout))
        if seed is not None:
            builder.seed(int(seed))

        input_config = _pop_any(raw, ("input", "inputs"), None)
        builder = _apply_input_config(builder, input_config)

        evolution_cfg = _as_mapping(_pop_any(raw, ("evolution", "dynamics"), None), name="evolution")
        tau = _pop_any(raw, ("tau",), None)
        if tau is not None and "tau" not in evolution_cfg:
            evolution_cfg["tau"] = tau
        model_cfg = _as_mapping(_pop_any(raw, ("model_kwargs", "model_params", "model_config"), None), name="model_kwargs")
        if model_cfg:
            builder.model(**model_cfg)
        if evolution_cfg:
            builder.evolution(**evolution_cfg)

        hamiltonian_cfg = _as_mapping(_pop_any(raw, ("hamiltonian", "hamiltonian_kwargs"), None), name="hamiltonian")
        for key in ("h0", "h1", "H0", "H1", "h0_terms", "h1_terms", "h0_matrix", "h1_matrix", "kind", "hamiltonian_kind"):
            if key in raw:
                hamiltonian_cfg[key] = raw.pop(key)
        if hamiltonian_cfg:
            builder.hamiltonian(**hamiltonian_cfg)

        circuit = _pop_any(raw, ("circuit", "quantum_circuit"), None)
        circuit_kwargs = _as_mapping(_pop_any(raw, ("circuit_kwargs", "circuit_config"), None), name="circuit_kwargs")
        if circuit is not None:
            builder.circuit(circuit, **circuit_kwargs)

        existing = _pop_any(raw, ("reservoir", "object"), None)
        if existing is not None:
            builder.use(existing)

        builder = _apply_readout_config(builder, raw)
        raw.pop("readout", None)
        raw.pop("observables", None)

        if raw:
            builder.model(**raw)
        builder._backend = backend_name
        return builder

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Any:
        """Compile a reservoir from a plain Python dictionary."""

        builder = cls.builder_from_dict(config)
        return builder.build()

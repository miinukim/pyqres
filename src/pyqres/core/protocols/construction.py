"""Structural contracts for reservoir factories, builders, and presets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .reservoirs import DimensionModelProtocol
from .specifications import DynamicsSpecProtocol, ReservoirSpecProtocol
from .types import DynamicsLike, FeatureMatrix, InputSequence, QiskitArtifactMap


@runtime_checkable
class ReservoirBuilderProtocol(Protocol):
    """Internal builder contract returned by dictionary reservoir construction."""

    @property
    def spec(self) -> ReservoirSpecProtocol: ...

    def backend(self, name: str = "exact") -> Any: ...

    def build(self, backend: str | None = None) -> Any: ...


@runtime_checkable
class ReservoirCompilerProtocol(Protocol):
    """Callable object/function that compiles a spec into an executable reservoir."""

    def __call__(self, spec: ReservoirSpecProtocol, backend: str = "exact") -> Any: ...


@runtime_checkable
class RunFunctionProtocol(Protocol):
    """Function contract for running any supported reservoir object on inputs."""

    def __call__(self, reservoir: Any, inputs: InputSequence) -> FeatureMatrix: ...


@runtime_checkable
class ReservoirFactoryProtocol(Protocol):
    """Dictionary-first reservoir factory contract."""

    @classmethod
    def builder_from_dict(
        cls, config: Mapping[str, Any]
    ) -> ReservoirBuilderProtocol: ...

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class PresetRegistryProtocol(Protocol):
    """Registry/adapter contract for named reservoir presets."""

    def names(self) -> list[str]: ...

    def get(self, name: str, **kwargs: object) -> ReservoirSpecProtocol: ...

    def build_dimension_model(
        self, spec: ReservoirSpecProtocol
    ) -> DimensionModelProtocol: ...

    def build_hamiltonian_params(
        self, spec: ReservoirSpecProtocol
    ) -> Mapping[str, Any]: ...

    def build_qiskit_artifacts(
        self, spec: ReservoirSpecProtocol
    ) -> QiskitArtifactMap: ...


@runtime_checkable
class DynamicsInferenceProtocol(Protocol):
    """Callable contract for resolving user dynamics input into builder state."""

    def __call__(
        self, value: DynamicsLike, *, default_preset: str
    ) -> tuple[DynamicsSpecProtocol | None, Mapping[str, Any]]: ...

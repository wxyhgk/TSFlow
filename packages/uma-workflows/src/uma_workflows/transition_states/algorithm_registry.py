from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Literal, TypeVar, cast

from uma_workflows.transition_states.algorithm_api import (
    AlgorithmMetadata,
    PathCandidate,
    PathInitializer,
    PathOptimizer,
)


AlgorithmRole = Literal["path_initializer", "path_optimizer"]
ConfigT = TypeVar("ConfigT")
AlgorithmT = TypeVar("AlgorithmT")


@dataclass(frozen=True)
class AlgorithmCompatibility:
    accepts_unparameterized: bool = True
    accepted_parameterizations: frozenset[str] | None = None
    produced_parameterization: str | None = None


@dataclass(frozen=True)
class AlgorithmRegistration(Generic[ConfigT, AlgorithmT]):
    role: AlgorithmRole
    metadata: AlgorithmMetadata
    config_type: type[ConfigT]
    factory: Callable[[ConfigT], AlgorithmT]
    compatibility: AlgorithmCompatibility = AlgorithmCompatibility()
    capabilities: tuple[str, ...] = ()


class AlgorithmRegistry:
    """Deterministic whitelist of numerical TS algorithms."""

    def __init__(self) -> None:
        self._registrations: dict[str, AlgorithmRegistration] = {}

    def register(
        self,
        registration: AlgorithmRegistration,
    ) -> None:
        algorithm_id = registration.metadata.algorithm_id
        if algorithm_id in self._registrations:
            raise ValueError(f"algorithm_id is already registered: {algorithm_id}")
        self._registrations[algorithm_id] = registration

    def describe(
        self,
        role: AlgorithmRole | None = None,
    ) -> tuple[AlgorithmRegistration, ...]:
        return tuple(
            registration
            for registration in self._registrations.values()
            if role is None or registration.role == role
        )

    def create_initializer(self, algorithm_id: str, config) -> PathInitializer:
        registration = self._require(algorithm_id, "path_initializer", config)
        return cast(PathInitializer, registration.factory(config))

    def create_optimizer(
        self,
        algorithm_id: str,
        config,
        *,
        candidate: PathCandidate | None = None,
    ) -> PathOptimizer:
        registration = self._require(algorithm_id, "path_optimizer", config)
        if candidate is not None:
            self._validate_parameterization(registration, candidate)
        return cast(PathOptimizer, registration.factory(config))

    def validate_optimizer_candidate(
        self,
        algorithm_id: str,
        candidate: PathCandidate,
    ) -> AlgorithmRegistration:
        registration = self._registrations.get(algorithm_id)
        if registration is None:
            raise ValueError(f"unknown path_optimizer algorithm_id: {algorithm_id}")
        if registration.role != "path_optimizer":
            raise ValueError(
                f"algorithm {algorithm_id} has role {registration.role}, "
                "not path_optimizer"
            )
        self._validate_parameterization(registration, candidate)
        return registration

    def _require(
        self,
        algorithm_id: str,
        role: AlgorithmRole,
        config,
    ) -> AlgorithmRegistration:
        registration = self._registrations.get(algorithm_id)
        if registration is None:
            raise ValueError(f"unknown {role} algorithm_id: {algorithm_id}")
        if registration.role != role:
            raise ValueError(
                f"algorithm {algorithm_id} has role {registration.role}, not {role}"
            )
        if not isinstance(config, registration.config_type):
            raise TypeError(
                f"algorithm {algorithm_id} requires config "
                f"{registration.config_type.__name__}"
            )
        return registration

    @staticmethod
    def _validate_parameterization(
        registration: AlgorithmRegistration,
        candidate: PathCandidate,
    ) -> None:
        parameterization = candidate.parameterization
        compatibility = registration.compatibility
        if parameterization is None:
            if not compatibility.accepts_unparameterized:
                raise ValueError(
                    f"algorithm {registration.metadata.algorithm_id} requires a "
                    "parameterized path candidate"
                )
            return
        accepted = compatibility.accepted_parameterizations
        if accepted is not None and parameterization.kind not in accepted:
            raise ValueError(
                f"algorithm {registration.metadata.algorithm_id} does not accept "
                f"path parameterization {parameterization.kind!r}"
            )

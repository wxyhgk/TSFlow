from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken
from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)


@dataclass(frozen=True)
class AlgorithmMetadata:
    algorithm_id: str
    version: str
    display_name: str


@dataclass(frozen=True)
class TransitionStateProblem:
    reactant: Atoms
    product: Atoms
    reaction_coordinates: tuple[ReactionCoordinateLike, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReactionPath:
    positions_a: np.ndarray
    coordinates: np.ndarray
    tangents_a: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_a)
        coordinates = np.asarray(self.coordinates)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("reaction path positions must have shape (images, atoms, 3)")
        if coordinates.shape != (positions.shape[0],):
            raise ValueError("reaction path coordinates must match the image count")
        if self.tangents_a is not None:
            tangents = np.asarray(self.tangents_a)
            if tangents.shape != positions.shape:
                raise ValueError("reaction path tangents must match positions")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(coordinates)):
            raise ValueError("reaction path must contain only finite values")
        if np.any(np.diff(coordinates) <= 0.0):
            raise ValueError("reaction path coordinates must be strictly increasing")

    def evaluate(
        self,
        coordinates,
        *,
        derivative_order: int = 0,
    ) -> np.ndarray:
        requested = np.asarray(coordinates, dtype=float)
        if derivative_order not in {0, 1}:
            raise ValueError("sampled reaction paths support derivatives 0 and 1")
        if derivative_order == 1 and self.tangents_a is not None:
            values = np.asarray(self.tangents_a, dtype=float)
        elif derivative_order == 1:
            values = np.gradient(
                np.asarray(self.positions_a, dtype=float),
                np.asarray(self.coordinates, dtype=float),
                axis=0,
            )
        else:
            values = np.asarray(self.positions_a, dtype=float)
        flattened = values.reshape(values.shape[0], -1)
        interpolated = np.stack(
            [
                np.interp(requested, self.coordinates, flattened[:, column])
                for column in range(flattened.shape[1])
            ],
            axis=1,
        )
        return interpolated.reshape((len(requested),) + values.shape[1:])

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions_a": np.asarray(self.positions_a, dtype=float).tolist(),
            "coordinates": np.asarray(self.coordinates, dtype=float).tolist(),
            "tangents_a": (
                np.asarray(self.tangents_a, dtype=float).tolist()
                if self.tangents_a is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PathParameterization:
    kind: str
    coefficients: tuple[tuple[tuple[float, float, float], ...], ...]

    @classmethod
    def from_array(cls, kind: str, values: np.ndarray) -> "PathParameterization":
        coefficients = tuple(
            tuple(
                tuple(float(component) for component in vector)
                for vector in control
            )
            for control in np.asarray(values, dtype=float)
        )
        return cls(kind=kind, coefficients=coefficients)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.coefficients, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "coefficients": self.coefficients,
        }


@dataclass(frozen=True)
class PathCandidate:
    path: ReactionPath
    score: float | None = None
    source: str = ""
    candidate_index: int | None = None
    parameterization: PathParameterization | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathInitializationResult:
    candidates: tuple[PathCandidate, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathOptimizationResult:
    path: ReactionPath
    converged: bool
    iterations: int
    optimizer_status: str
    status_code: int | None = None
    adaptive_sampling_enabled: bool | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "converged": self.converged,
            "iterations": self.iterations,
            "optimizer_status": self.optimizer_status,
            "status_code": self.status_code,
            "adaptive_sampling_enabled": self.adaptive_sampling_enabled,
            "warnings": self.warnings,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    passed: bool
    message: str
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathProgressSnapshot:
    path: ReactionPath
    energy_ev: float | None = None
    path_energies_ev: tuple[float, ...] = ()
    max_force_ev_per_a: float | None = None


PathProgressCallback = Callable[
    [str, int, int, PathProgressSnapshot | None],
    None,
]


class PathInitializer(Protocol):
    metadata: AlgorithmMetadata

    def generate(
        self,
        problem: TransitionStateProblem,
        *,
        calculator,
        cancel_token: CancellationToken,
        progress: PathProgressCallback | None = None,
    ) -> PathInitializationResult: ...


class PathOptimizer(Protocol):
    metadata: AlgorithmMetadata

    def optimize(
        self,
        problem: TransitionStateProblem,
        initial_path: PathCandidate,
        *,
        calculator,
        cancel_token: CancellationToken,
        progress: PathProgressCallback | None = None,
    ) -> PathOptimizationResult: ...


class TransitionStateValidator(Protocol):
    metadata: AlgorithmMetadata

    def validate(
        self,
        problem: TransitionStateProblem,
        path: ReactionPath,
        *,
        calculator,
        cancel_token: CancellationToken,
    ) -> Sequence[ValidationFinding]: ...

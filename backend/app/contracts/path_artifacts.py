from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.models import TaskName
from app.contracts.structures import ReactionCoordinate
from app.contracts.transition_states import CollectiveCoordinateDefinition


class _StrictArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReactionPathArtifactV1(_StrictArtifactModel):
    positions_a: list[list[tuple[float, float, float]]] = Field(
        min_length=3,
        max_length=201,
    )
    coordinates: list[float] = Field(min_length=3, max_length=201)
    tangents_a: list[list[tuple[float, float, float]]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_path_shape_and_values(self) -> "ReactionPathArtifactV1":
        frame_count = len(self.positions_a)
        if len(self.coordinates) != frame_count:
            raise ValueError("path coordinates must match the frame count")
        atom_count = len(self.positions_a[0])
        if atom_count < 1:
            raise ValueError("path frames must contain at least one atom")
        if any(len(frame) != atom_count for frame in self.positions_a):
            raise ValueError("all path frames must have the same atom count")
        if self.tangents_a is not None:
            if len(self.tangents_a) != frame_count or any(
                len(frame) != atom_count for frame in self.tangents_a
            ):
                raise ValueError("path tangents must match the path shape")
        if not all(math.isfinite(value) for value in self.coordinates):
            raise ValueError("path coordinates must be finite")
        if any(
            current <= previous
            for previous, current in zip(
                self.coordinates,
                self.coordinates[1:],
            )
        ):
            raise ValueError("path coordinates must be strictly increasing")
        if not math.isclose(self.coordinates[0], 0.0, abs_tol=1.0e-9):
            raise ValueError("path coordinates must start at zero")
        if not math.isclose(self.coordinates[-1], 1.0, abs_tol=1.0e-9):
            raise ValueError("path coordinates must end at one")
        if not _vectors_are_finite(self.positions_a):
            raise ValueError("path positions must be finite")
        if self.tangents_a is not None and not _vectors_are_finite(
            self.tangents_a
        ):
            raise ValueError("path tangents must be finite")
        return self


class PathParameterizationArtifactV1(_StrictArtifactModel):
    kind: str = Field(min_length=1, max_length=120)
    coefficients: list[list[tuple[float, float, float]]] = Field(
        min_length=1,
        max_length=201,
    )

    @model_validator(mode="after")
    def validate_coefficients(self) -> "PathParameterizationArtifactV1":
        atom_count = len(self.coefficients[0])
        if atom_count < 1:
            raise ValueError("path coefficients must contain atoms")
        if any(len(control) != atom_count for control in self.coefficients):
            raise ValueError("all path controls must have the same atom count")
        if not _vectors_are_finite(self.coefficients):
            raise ValueError("path coefficients must be finite")
        return self


class PathArtifactV1(_StrictArtifactModel):
    schema_version: Literal[1] = 1
    model_name: str = Field(min_length=1, max_length=120)
    task_name: TaskName
    reactant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    product_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    atom_symbols: list[str] = Field(min_length=1, max_length=2_000)
    charge: int
    spin: int = Field(ge=1)
    initializer_algorithm_id: str = Field(min_length=1, max_length=120)
    initializer_algorithm_version: str = Field(min_length=1, max_length=80)
    candidate_index: int = Field(ge=0)
    score_ev: float | None = None
    source: str = Field(max_length=120)
    path: ReactionPathArtifactV1
    parameterization: PathParameterizationArtifactV1 | None = None
    reaction_coordinates: list[ReactionCoordinate] = Field(
        default_factory=list,
        max_length=64,
    )
    warnings: list[str] = Field(default_factory=list, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_cross_field_shapes(self) -> "PathArtifactV1":
        atom_count = len(self.atom_symbols)
        if any(len(frame) != atom_count for frame in self.path.positions_a):
            raise ValueError("path atom count does not match atom_symbols")
        if self.parameterization is not None and any(
            len(control) != atom_count
            for control in self.parameterization.coefficients
        ):
            raise ValueError(
                "path parameterization atom count does not match atom_symbols"
            )
        if self.score_ev is not None and not math.isfinite(self.score_ev):
            raise ValueError("path candidate score must be finite")
        seen: set[tuple[int, int, str]] = set()
        for coordinate in self.reaction_coordinates:
            if coordinate.atom_j >= atom_count:
                raise ValueError("reaction coordinate atom index is out of range")
            key = (coordinate.atom_i, coordinate.atom_j, coordinate.kind)
            if key in seen:
                raise ValueError("reaction coordinates contain a duplicate")
            seen.add(key)
        return self


class OptimizedPathArtifactV1(_StrictArtifactModel):
    schema_version: Literal[1] = 1
    model_name: str = Field(min_length=1, max_length=120)
    task_name: TaskName
    reactant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    product_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    atom_symbols: list[str] = Field(min_length=1, max_length=2_000)
    charge: int
    spin: int = Field(ge=1)
    candidate_index: int = Field(ge=0)
    initializer_algorithm_id: str = Field(min_length=1, max_length=120)
    initializer_algorithm_version: str = Field(min_length=1, max_length=80)
    optimizer_algorithm_id: str = Field(min_length=1, max_length=120)
    optimizer_algorithm_version: str = Field(min_length=1, max_length=80)
    optimizer_parameters: dict[str, Any]
    converged: bool
    iterations: int = Field(ge=0)
    optimizer_status: str
    status_code: int | None = None
    adaptive_sampling_enabled: bool | None = None
    path: ReactionPathArtifactV1
    cpu_energies_ev: list[float] = Field(min_length=3, max_length=201)
    reaction_coordinates: list[ReactionCoordinate] = Field(
        default_factory=list,
        max_length=64,
    )
    barrier_score_ev: float
    coordination_penalty_ev: float
    total_score_ev: float
    same_kind_progress_imbalance: float
    has_interior_maximum: bool
    warnings: list[str] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def validate_optimized_path(self) -> "OptimizedPathArtifactV1":
        atom_count = len(self.atom_symbols)
        if any(len(frame) != atom_count for frame in self.path.positions_a):
            raise ValueError("optimized path atom count does not match atom_symbols")
        if len(self.cpu_energies_ev) != len(self.path.positions_a):
            raise ValueError("optimized path energies must match the frame count")
        numeric_values = [
            *self.cpu_energies_ev,
            self.barrier_score_ev,
            self.coordination_penalty_ev,
            self.total_score_ev,
            self.same_kind_progress_imbalance,
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("optimized path diagnostics must be finite")
        return self


class CollectiveScanPointArtifactV1(_StrictArtifactModel):
    target_value_a: float
    coordinate_value_a: float
    energy_ev: float
    derivative_ev_per_a: float
    orthogonal_fmax_ev_per_a: float = Field(ge=0.0)
    converged: bool
    steps: int = Field(ge=0)
    positions_a: list[tuple[float, float, float]] = Field(
        min_length=1,
        max_length=2_000,
    )

    @model_validator(mode="after")
    def validate_finite_values(self) -> "CollectiveScanPointArtifactV1":
        values = (
            self.target_value_a,
            self.coordinate_value_a,
            self.energy_ev,
            self.derivative_ev_per_a,
            self.orthogonal_fmax_ev_per_a,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("collective scan diagnostics must be finite")
        if not _vectors_are_finite([self.positions_a]):
            raise ValueError("collective scan positions must be finite")
        return self


class CollectiveScanArtifactV1(_StrictArtifactModel):
    schema_version: Literal[1] = 1
    model_name: str = Field(min_length=1, max_length=120)
    task_name: TaskName
    reactant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    product_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_optimized_path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    atom_symbols: list[str] = Field(min_length=1, max_length=2_000)
    charge: int
    spin: int = Field(ge=1)
    collective_coordinate: CollectiveCoordinateDefinition
    points: list[CollectiveScanPointArtifactV1] = Field(
        min_length=3,
        max_length=31,
    )
    derivative_brackets: list[tuple[int, int]] = Field(default_factory=list)
    interior_maxima: list[int] = Field(default_factory=list)
    selected_maximum_index: int | None = Field(default=None, ge=1)
    warnings: list[str] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def validate_scan(self) -> "CollectiveScanArtifactV1":
        atom_count = len(self.atom_symbols)
        if any(len(point.positions_a) != atom_count for point in self.points):
            raise ValueError("collective scan atom counts must match atom_symbols")
        if any(
            left < 0 or right != left + 1 or right >= len(self.points)
            for left, right in self.derivative_brackets
        ):
            raise ValueError("collective scan derivative bracket is invalid")
        if any(index <= 0 or index >= len(self.points) - 1 for index in self.interior_maxima):
            raise ValueError("collective scan maximum must be interior")
        if (
            self.selected_maximum_index is not None
            and self.selected_maximum_index not in self.interior_maxima
        ):
            raise ValueError("selected scan maximum must be an interior maximum")
        return self



def _vectors_are_finite(
    frames: list[list[tuple[float, float, float]]],
) -> bool:
    return all(
        math.isfinite(component)
        for frame in frames
        for vector in frame
        for component in vector
    )

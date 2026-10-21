from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts.artifacts import ArtifactReference
from app.contracts.calculations import OptimizationFrame
from app.contracts.structures import AtomicStructure
from app.contracts.transition_states import IrcEndpointRelaxationResult, IrcPathFrameResult


class OptimizePathCandidateActionResult(BaseModel):
    action_type: Literal["optimize_path_candidate"] = "optimize_path_candidate"
    source_path_artifact: ArtifactReference
    optimized_path_artifact: ArtifactReference
    candidate_index: int = Field(ge=0)
    optimizer_algorithm_id: str
    optimizer_algorithm_version: str
    converged: bool
    iterations: int = Field(ge=0)
    optimizer_status: str
    status_code: int | None = None
    adaptive_sampling_enabled: bool | None = None
    energy_profile: list[Any] = Field(default_factory=list)
    highest_path_coordinate: float
    highest_structure: AtomicStructure
    has_interior_maximum: bool
    barrier_score_ev: float
    coordination_penalty_ev: float
    total_score_ev: float
    same_kind_progress_imbalance: float
    scientific_status: Literal["optimized_path_candidate"] = "optimized_path_candidate"
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class PathCurvatureProbeSummary(BaseModel):
    frame_index: int = Field(ge=0)
    path_coordinate: float
    structure: AtomicStructure
    directional_curvature_ev_per_a2: float
    gradient_projection_ev_per_a: float
    max_force_ev_per_a: float
    has_negative_directional_curvature: bool


class ProbePathCurvatureActionResult(BaseModel):
    action_type: Literal["probe_path_curvature"] = "probe_path_curvature"
    source_optimized_path_artifact: ArtifactReference
    probes: list[PathCurvatureProbeSummary] = Field(default_factory=list)
    peak_frame_index: int = Field(ge=0)
    selected_probe_index: int | None = Field(default=None, ge=0)
    selected_structure: AtomicStructure | None = None
    force_evaluations: int = Field(ge=0)
    scientific_status: Literal[
        "negative_directional_curvature_seed",
        "no_negative_directional_curvature",
    ]
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class RunCollectiveCoordinateScanActionResult(BaseModel):
    action_type: Literal["run_collective_coordinate_scan"] = (
        "run_collective_coordinate_scan"
    )
    source_optimized_path_artifact: ArtifactReference
    scan_artifact: ArtifactReference
    collective_coordinate: Any
    points: list[Any] = Field(default_factory=list)
    interior_maxima: list[int] = Field(default_factory=list)
    selected_maximum_index: int | None = Field(default=None, ge=0)
    selected_structure: AtomicStructure | None = None
    scientific_status: Literal[
        "collective_scan_seed",
        "collective_scan_no_interior_maximum",
    ]
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class PathCandidateSummary(BaseModel):
    candidate_index: int = Field(ge=0)
    score_ev: float | None = None
    source: str
    frame_count: int = Field(ge=3)
    parameterization_kind: str | None = None
    preview_structure: AtomicStructure
    path_artifact: ArtifactReference
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratePathCandidatesActionResult(BaseModel):
    action_type: Literal["generate_path_candidates"] = "generate_path_candidates"
    initializer_algorithm_id: str
    initializer_algorithm_version: str
    candidates: list[PathCandidateSummary]
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class SaddleRefinementActionResult(BaseModel):
    action_type: Literal["refine_saddle"] = "refine_saddle"
    structure: AtomicStructure
    energy_ev: float
    forces_ev_per_a: list[tuple[float, float, float]]
    max_force_ev_per_a: float
    converged: bool
    stationary: bool
    optimization_steps: int
    trajectory: list[OptimizationFrame] = Field(default_factory=list)
    scientific_status: Literal["unvalidated_candidate"] = "unvalidated_candidate"
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class HessianModesActionResult(BaseModel):
    action_type: Literal["compute_full_hessian_and_modes"] = (
        "compute_full_hessian_and_modes"
    )
    structure: AtomicStructure
    energy_ev: float
    max_force_ev_per_a: float
    stationary: bool
    hessian_scope: Literal["full"] = "full"
    frequencies_cm1: list[float]
    imaginary_frequencies_cm1: list[float]
    has_exactly_one_imaginary_mode: bool
    hessian_method: str
    hessian_asymmetry_relative: float
    hessian_quality_valid: bool
    pes_index1_saddle_valid: bool
    hessian_artifact: ArtifactReference
    mode_artifact: ArtifactReference
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class BidirectionalIrcActionResult(BaseModel):
    action_type: Literal["run_bidirectional_irc"] = "run_bidirectional_irc"
    transition_state: AtomicStructure
    frames: list[IrcPathFrameResult] = Field(default_factory=list)
    forward_frame_count: int
    reverse_frame_count: int
    path_complete: bool
    endpoint_a: AtomicStructure
    endpoint_b: AtomicStructure
    endpoint_a_relaxation: IrcEndpointRelaxationResult
    endpoint_b_relaxation: IrcEndpointRelaxationResult
    endpoint_match_evaluated: bool
    endpoint_matches_expected: bool | None = None
    endpoint_a_role: str | None = None
    endpoint_b_role: str | None = None
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


ActionResult = (
    GeneratePathCandidatesActionResult
    | OptimizePathCandidateActionResult
    | ProbePathCurvatureActionResult
    | RunCollectiveCoordinateScanActionResult
    | SaddleRefinementActionResult
    | HessianModesActionResult
    | BidirectionalIrcActionResult
)

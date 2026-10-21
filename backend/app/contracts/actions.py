from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.artifacts import ArtifactReference, ArtifactSelector
from app.contracts.models import TaskName
from app.contracts.parameters import CalculationParameters
from app.contracts.requests import JobCreate
from app.contracts.structures import (
    AtomicStructure,
    ReactionCoordinate,
    validate_electron_parity,
)
from app.contracts.transition_states import CollectiveCoordinateDefinition


ActionType = Literal[
    "evaluate_energy_forces",
    "optimize_minimum",
    "analyze_frequencies",
    "transition_state_workflow",
    "refine_saddle",
    "compute_full_hessian_and_modes",
    "run_bidirectional_irc",
    "generate_path_candidates",
    "optimize_path_candidate",
    "probe_path_curvature",
    "run_collective_coordinate_scan",
]


class _StrictParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvaluateEnergyForcesParameters(_StrictParameters):
    pass


class OptimizeMinimumParameters(_StrictParameters):
    fmax: float = Field(default=0.05, gt=0.0, le=10.0)
    max_steps: int = Field(default=100, ge=1, le=100_000)


class AnalyzeFrequenciesParameters(_StrictParameters):
    atom_indices: list[int] | None = Field(default=None, max_length=2_000)
    delta_a: float = Field(default=0.01, gt=0.0, le=0.1)
    imaginary_threshold_cm1: float = Field(default=100.0, gt=0.0, le=1_000.0)
    hessian_method: Literal["auto", "analytic", "finite_difference"] = "auto"
    finite_difference_max_atoms: int = Field(default=30, ge=1, le=2_000)


class RefineSaddleParameters(_StrictParameters):
    fmax: float = Field(default=0.05, gt=0.0, le=2.0)
    max_steps: int = Field(default=100, ge=1, le=10_000)
    hessian_method: Literal["auto", "analytic", "finite_difference"] = "auto"
    hessian_delta_a: float = Field(default=0.01, gt=0.0, le=0.1)
    finite_difference_max_atoms: int = Field(default=30, ge=1, le=2_000)


class ComputeHessianModesParameters(_StrictParameters):
    hessian_method: Literal["auto", "analytic", "finite_difference"] = "auto"
    delta_a: float = Field(default=0.01, gt=0.0, le=0.1)
    finite_difference_max_atoms: int = Field(default=30, ge=1, le=2_000)


class RunBidirectionalIrcParameters(_StrictParameters):
    engine: Literal["geometric", "sella"] = "geometric"
    steps: int = Field(default=30, ge=2, le=1_000)
    dx: float = Field(default=0.12, gt=0.0, le=1.0)
    inner_steps: int = Field(default=10, ge=1, le=1_000)
    inner_fmax: float = Field(default=0.01, gt=0.0, le=2.0)
    fmax: float = Field(default=0.05, gt=0.0, le=2.0)
    endpoint_distance_threshold_a: float = Field(default=0.02, ge=0.0, le=5.0)
    relax_endpoints: bool = True
    endpoint_relaxation_fmax: float = Field(default=0.05, gt=0.0, le=2.0)
    endpoint_relaxation_max_steps: int = Field(default=100, ge=1, le=10_000)


class GeneratePathCandidatesParameters(_StrictParameters):
    initializer_id: str = "fbenm.correlated"
    nmove: int = Field(default=10, ge=3, le=20)
    ensemble_size: int = Field(default=3, ge=1, le=32)
    evaluation_device: Literal["cpu", "mps"] = "cpu"
    path_bias_mode: Literal["native", "asynchronous", "legacy_synchronous"] = "asynchronous"


class OptimizePathCandidateParameters(_StrictParameters):
    optimizer_id: str = "dmf.direct_maxflux"
    expected_optimizer_version: str = "1"
    nmove: int = Field(default=10, ge=3, le=20)
    max_iterations: int = Field(default=200, ge=1, le=10_000)
    convergence: Literal["tight", "middle", "loose"] = "middle"
    path_points: int = Field(default=31, ge=3, le=201)
    update_teval: bool = True
    accept_every_trial_step: bool = False
    path_bias_mode: Literal["native", "asynchronous", "legacy_synchronous"] = "asynchronous"
    evaluation_device: Literal["cpu", "mps"] = "cpu"


class ProbePathCurvatureParameters(_StrictParameters):
    displacement_a: float = Field(default=0.01, gt=0.0, le=0.2)
    negative_threshold_ev_per_a2: float = Field(default=-0.05, lt=0.0, ge=-100.0)
    max_frames: int = Field(default=9, ge=3, le=201)


class CollectiveCoordinateScanParameters(_StrictParameters):
    points: int = Field(default=9, ge=3, le=31)
    target_values_a: list[float] | None = Field(default=None, min_length=3, max_length=31)
    range_padding_a: float = Field(default=0.25, ge=0.0, le=10.0)
    fmax: float = Field(default=0.05, gt=0.0, le=2.0)
    max_steps_per_point: int = Field(default=100, ge=1, le=10_000)
    restraint_strength_ev_per_a2: float = Field(default=50.0, gt=0.0, le=10_000.0)


class ActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    model_name: str = "uma-s-1p2p1"
    task_name: TaskName = "omol"
    structure: AtomicStructure


class EvaluateEnergyForcesAction(ActionBase):
    action_type: Literal["evaluate_energy_forces"]
    parameters: EvaluateEnergyForcesParameters = Field(
        default_factory=EvaluateEnergyForcesParameters
    )


class OptimizeMinimumAction(ActionBase):
    action_type: Literal["optimize_minimum"]
    parameters: OptimizeMinimumParameters = Field(
        default_factory=OptimizeMinimumParameters
    )


class AnalyzeFrequenciesAction(ActionBase):
    action_type: Literal["analyze_frequencies"]
    parameters: AnalyzeFrequenciesParameters = Field(
        default_factory=AnalyzeFrequenciesParameters
    )


class GeneratePathCandidatesAction(ActionBase):
    action_type: Literal["generate_path_candidates"]
    product_structure: AtomicStructure
    reaction_coordinates: list[ReactionCoordinate] = Field(default_factory=list, max_length=64)
    parameters: GeneratePathCandidatesParameters = Field(default_factory=GeneratePathCandidatesParameters)


class OptimizePathCandidateAction(ActionBase):
    action_type: Literal["optimize_path_candidate"]
    product_structure: AtomicStructure
    reaction_coordinates: list[ReactionCoordinate] = Field(default_factory=list, max_length=64)
    path_candidate: ArtifactSelector
    parameters: OptimizePathCandidateParameters = Field(default_factory=OptimizePathCandidateParameters)


class ProbePathCurvatureAction(ActionBase):
    action_type: Literal["probe_path_curvature"]
    product_structure: AtomicStructure
    reaction_coordinates: list[ReactionCoordinate] = Field(default_factory=list, max_length=64)
    optimized_path: ArtifactSelector
    parameters: ProbePathCurvatureParameters = Field(default_factory=ProbePathCurvatureParameters)


class RunCollectiveCoordinateScanAction(ActionBase):
    action_type: Literal["run_collective_coordinate_scan"]
    product_structure: AtomicStructure
    reaction_coordinates: list[ReactionCoordinate] = Field(default_factory=list, max_length=64)
    optimized_path: ArtifactSelector
    collective_coordinate: CollectiveCoordinateDefinition | None = None
    parameters: CollectiveCoordinateScanParameters = Field(default_factory=CollectiveCoordinateScanParameters)


class TransitionStateWorkflowAction(ActionBase):
    action_type: Literal["transition_state_workflow"]
    product_structure: AtomicStructure
    transition_state_guess: AtomicStructure | None = None
    reaction_coordinates: list[ReactionCoordinate] = Field(
        default_factory=list,
        max_length=64,
    )
    parameters: CalculationParameters = Field(default_factory=CalculationParameters)
    validation_profile_id: Literal["uma-ts-strict-v1"] = "uma-ts-strict-v1"


class RefineSaddleAction(ActionBase):
    action_type: Literal["refine_saddle"]
    parameters: RefineSaddleParameters = Field(default_factory=RefineSaddleParameters)
    validation_profile_id: Literal["uma-ts-strict-v1"] = "uma-ts-strict-v1"


class ComputeFullHessianAndModesAction(ActionBase):
    action_type: Literal["compute_full_hessian_and_modes"]
    parameters: ComputeHessianModesParameters = Field(
        default_factory=ComputeHessianModesParameters
    )
    validation_profile_id: Literal["uma-ts-strict-v1"] = "uma-ts-strict-v1"


class RunBidirectionalIrcAction(ActionBase):
    action_type: Literal["run_bidirectional_irc"]
    hessian_artifact: ArtifactReference
    mode_artifact: ArtifactReference
    expected_reactant: AtomicStructure | None = None
    expected_product: AtomicStructure | None = None
    parameters: RunBidirectionalIrcParameters = Field(
        default_factory=RunBidirectionalIrcParameters
    )
    validation_profile_id: Literal["uma-ts-strict-v1"] = "uma-ts-strict-v1"

    @model_validator(mode="after")
    def validate_expected_endpoints(self) -> "RunBidirectionalIrcAction":
        if (self.expected_reactant is None) != (self.expected_product is None):
            raise ValueError(
                "expected_reactant and expected_product must be supplied together"
            )
        if self.expected_reactant is not None and self.expected_product is not None:
            symbols = [atom.symbol for atom in self.structure.atoms]
            for label, endpoint in (
                ("expected_reactant", self.expected_reactant),
                ("expected_product", self.expected_product),
            ):
                validate_electron_parity(endpoint)
                if [atom.symbol for atom in endpoint.atoms] != symbols:
                    raise ValueError(
                        f"{label} must use the transition state's atom ordering"
                    )
                if endpoint.charge != self.structure.charge:
                    raise ValueError(f"{label} must have the same charge")
                if endpoint.spin != self.structure.spin:
                    raise ValueError(f"{label} must have the same spin multiplicity")
        return self


ActionSpec = Annotated[
    EvaluateEnergyForcesAction
    | OptimizeMinimumAction
    | AnalyzeFrequenciesAction
    | GeneratePathCandidatesAction
    | OptimizePathCandidateAction
    | ProbePathCurvatureAction
    | RunCollectiveCoordinateScanAction
    | TransitionStateWorkflowAction
    | RefineSaddleAction
    | ComputeFullHessianAndModesAction
    | RunBidirectionalIrcAction,
    Field(discriminator="action_type"),
]


class LegacyJobExecution(BaseModel):
    execution_type: Literal["legacy_job"] = "legacy_job"
    request: "JobCreate"


class RecoveryExecutionContext(BaseModel):
    recovery_id: UUID
    parent_action_id: UUID
    source_optimized_path: ArtifactSelector
    step_index: int = Field(ge=0)
    saddle_refinement_attempts: int = Field(default=0, ge=0)
    collective_scan_attempts: int = Field(default=0, ge=0)
    irc_attempts: int = Field(default=0, ge=0)


class ActionExecution(BaseModel):
    execution_type: Literal["action"] = "action"
    action: ActionSpec
    recovery_context: "RecoveryExecutionContext | None" = None
    resolved_artifacts: dict[str, ArtifactReference] = Field(default_factory=dict)
    resolved_algorithms: dict[str, str] = Field(default_factory=dict)


ExecutionPayload = Annotated[
    LegacyJobExecution | ActionExecution,
    Field(discriminator="execution_type"),
]


class ExecutionEnvelope(BaseModel):
    schema_version: Literal[1] = 1
    execution_id: UUID
    payload: ExecutionPayload


class ActionRecord(BaseModel):
    id: UUID
    action_type: ActionType
    schema_version: Literal[1] = 1
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    job_id: UUID
    job_url: str
    events_url: str
    guidance_url: str
    created_at: datetime
    updated_at: datetime
    execution_backend: Literal["job_bridge"] = "job_bridge"


class ActionCapability(BaseModel):
    action_type: ActionType
    schema_version: Literal[1] = 1
    available: bool = True
    unavailable_reason: str | None = None
    execution_backend: Literal["job_bridge"] = "job_bridge"
    description: str
    input_roles: list[str]
    output_kinds: list[str]
    prerequisite_codes: list[str] = Field(default_factory=list)
    possible_next_actions: list[ActionType] = Field(default_factory=list)
    request_schema: dict[str, Any]
    validation_profile_id: str | None = None
    validation_is_server_controlled: bool = False


class ActionCapabilitiesResponse(BaseModel):
    items: list[ActionCapability]


class ActionFinding(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    values: dict[str, Any] = Field(default_factory=dict)


class NextActionDescriptor(BaseModel):
    """Advisory, directly POST-able operation exposed to an AI client."""

    action_type: ActionType
    reason_code: str
    description: str
    input_bindings: dict[str, str] = Field(default_factory=dict)
    parameter_patch: dict[str, Any] = Field(default_factory=dict)


class ActionGuidance(BaseModel):
    action_id: UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    terminal: bool
    scientific_state: Literal[
        "pending",
        "completed",
        "candidate",
        "requires_refinement",
        "index1_saddle",
        "irc_connected_saddle",
        "validated_transition_state",
        "mechanism_mismatch",
        "inconclusive",
        "execution_failed",
        "cancelled",
    ]
    advisory_only: Literal[True] = True
    findings: list[ActionFinding] = Field(default_factory=list)
    next_actions: list[NextActionDescriptor] = Field(default_factory=list)


from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.contracts.artifacts import ArtifactReference
from app.contracts.models import TaskName


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Atom(BaseModel):
    symbol: Annotated[str, Field(pattern=r"^[A-Z][a-z]?$", examples=["Fe"])]
    position: tuple[float, float, float]


class AtomicStructure(BaseModel):
    atoms: Annotated[list[Atom], Field(min_length=1, max_length=2_000)]
    charge: int
    spin: int = Field(ge=1, description="Spin multiplicity")
    name: Annotated[str | None, Field(max_length=255)] = None
    comment: Annotated[str | None, Field(max_length=1_000)] = None
    source_format: Literal["xyz", "sdf"] | None = None


class ReactionCoordinate(BaseModel):
    """User-selected pair used to define and validate a reaction centre."""

    atom_i: int = Field(ge=0, description="Zero-based atom index")
    atom_j: int = Field(ge=0, description="Zero-based atom index")
    kind: Literal["breaking", "forming", "coordination"]
    label: Annotated[str | None, Field(max_length=120)] = None

    @model_validator(mode="after")
    def validate_pair(self) -> "ReactionCoordinate":
        if self.atom_i == self.atom_j:
            raise ValueError("a reaction coordinate requires two different atoms")
        if self.atom_i > self.atom_j:
            self.atom_i, self.atom_j = self.atom_j, self.atom_i
        return self


class CalculationParameters(BaseModel):
    # Transition-state path compatibility fields.  They are retained in the
    # legacy JobCreate envelope while typed Actions use the narrower contracts.
    endpoint_preopt: bool = True
    endpoint_preopt_fmax: float = Field(default=0.10, gt=0, le=2.0)
    endpoint_preopt_max_steps: int = Field(default=50, ge=1, le=10_000)
    path_initializer_id: str = "fbenm.correlated"
    path_optimizer_id: str = "dmf.direct_maxflux"
    path_ensemble_size: int = Field(default=3, ge=1, le=32)
    path_bias_mode: Literal["native", "asynchronous", "legacy_synchronous"] = "asynchronous"
    dmf_accept_every_trial_step: bool = False
    path_repulsion_strength_ev_per_a2: float = Field(default=75.0, gt=0, le=10_000)
    path_repulsion_cutoff_scale: float = Field(default=0.80, gt=0, le=10)
    reaction_coordinate_sync_strength_ev: float = Field(default=5.0, ge=0, le=10_000)
    reaction_coordinate_range_strength_ev: float = Field(default=100.0, ge=0, le=10_000)
    reaction_coordinate_schedule_strength_ev: float = Field(default=20.0, ge=0, le=10_000)
    path_scaffold_strength_ev_per_a2: float = Field(default=5.0, ge=0, le=10_000)
    near_barrierless_threshold_ev: float = Field(default=0.05, ge=0, le=10)
    sella_path_mode_only: bool = False
    relaxed_scan_refine: bool = True
    relaxed_scan_progress: float = Field(default=0.50, ge=0, le=1)
    relaxed_scan_fmax: float = Field(default=0.05, gt=0, le=2)
    relaxed_scan_max_steps: int = Field(default=100, ge=1, le=10_000)
    saddle_automatic_fallback: bool = True
    saddle_fallback_nearby_frames: int = Field(default=7, ge=1, le=101)
    saddle_fallback_max_candidates: int = Field(default=3, ge=1, le=32)
    optimization_device: Literal["auto", "cpu", "mps"] | None = None
    fmax: float = Field(default=0.05, gt=0, le=10, description="eV/A")
    max_steps: int = Field(default=100, ge=1, le=100_000)
    gaussian_route: str = Field(
        default="#p B3LYP/6-31G(d) Opt Freq",
        min_length=1,
        max_length=500,
    )
    gaussian_nproc: int = Field(default=8, ge=1, le=1_024)
    gaussian_memory: str = Field(
        default="8GB",
        pattern=r"^[1-9][0-9]*(?:KB|MB|GB|TB)$",
    )
    nmove: int = Field(
        default=3,
        ge=3,
        le=20,
        description="Number of movable interior DMF energy-evaluation points",
    )
    update_teval: bool = Field(
        default=True,
        description="Adapt DMF energy-evaluation points toward the barrier region",
    )
    dmf_convergence: Literal["tight", "middle", "loose"] = "middle"
    dmf_max_iterations: int = Field(default=200, ge=1, le=10_000)
    path_points: int = Field(
        default=31,
        ge=3,
        le=201,
        description="Number of structures sampled from the final B-spline path",
    )
    sella_refine: bool = True
    saddle_final_refinement_engine: Literal["geometric", "sella"] = "geometric"
    sella_mps_fmax: float = Field(default=0.10, gt=0, le=2.0, description="eV/A")
    sella_cpu_refine: bool = Field(
        default=False,
        description=(
            "Run iterative CPU Sella refinement after MPS. Disabled by default for "
            "fast screening; the final energy and forces are still recomputed on CPU."
        ),
    )
    sella_cpu_fmax: float = Field(default=0.05, gt=0, le=2.0, description="eV/A")
    sella_max_steps: int = Field(default=100, ge=1, le=10_000)
    sella_active_region_hops: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Freeze atoms farther than this many covalent-graph hops from changing bonds during Sella refinement",
    )
    compute_frequencies: bool = True
    frequency_max_atoms: int = Field(default=30, ge=1, le=2_000)
    frequency_active_region_hops: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Graph hops around clicked reaction coordinates for a partial TS Hessian",
    )
    frequency_delta_a: float = Field(default=0.01, gt=0, le=0.1)
    hessian_method: Literal["auto", "analytic", "finite_difference"] = "auto"
    maximum_hessian_asymmetry: float = Field(default=0.05, ge=0.0, le=1.0)
    imaginary_frequency_threshold_cm1: float = Field(
        default=100.0,
        gt=0,
        le=1_000,
        description="Negative modes below this magnitude are treated as numerical/rotational noise",
    )
    irc_follow: bool = Field(
        default=False,
        description=(
            "Run bidirectional CPU UMA IRC after a candidate passes the "
            "full-Hessian transition-state screen"
        ),
    )
    irc_engine: Literal["geometric", "sella"] = "geometric"
    irc_steps: int = Field(default=30, ge=2, le=1_000)
    irc_dx: float = Field(default=0.12, gt=0.0, le=1.0)
    irc_inner_steps: int = Field(default=10, ge=1, le=1_000)
    irc_inner_fmax: float = Field(default=0.01, gt=0.0, le=2.0)
    irc_fmax: float = Field(default=0.05, gt=0.0, le=2.0)
    irc_endpoint_distance_threshold_a: float = Field(
        default=0.02,
        ge=0.0,
        le=5.0,
    )
    irc_require_endpoint_match: bool = True
    irc_relax_endpoints: bool = True
    irc_endpoint_relaxation_fmax: float = Field(default=0.05, gt=0.0, le=2.0)
    irc_endpoint_relaxation_max_steps: int = Field(default=100, ge=1, le=10_000)
    frequency_atom_indices: list[int] | None = Field(
        default=None,
        max_length=2_000,
        description=(
            "Zero-based atom indices included in a partial numerical Hessian; "
            "all atoms are used when omitted"
        ),
    )


class JobCreate(BaseModel):
    model_name: str = "uma-s-1p2p1"
    task_name: TaskName = "omol"
    calculation: Literal[
        "single_point",
        "geometry_optimization",
        "transition_state_search",
        "frequency_analysis",
    ]
    structure: AtomicStructure
    product_structure: AtomicStructure | None = None
    transition_state_guess: AtomicStructure | None = None
    reaction_coordinates: list[ReactionCoordinate] = Field(
        default_factory=list,
        max_length=64,
    )
    parameters: CalculationParameters = Field(default_factory=CalculationParameters)

    @model_validator(mode="after")
    def validate_task_inputs(self) -> "JobCreate":
        _validate_electron_parity(self.structure)
        if self.calculation == "transition_state_search":
            if self.product_structure is None:
                raise ValueError("transition_state_search requires product_structure")
            _validate_electron_parity(self.product_structure)
            reactant_symbols = [atom.symbol for atom in self.structure.atoms]
            product_symbols = [atom.symbol for atom in self.product_structure.atoms]
            if reactant_symbols != product_symbols:
                raise ValueError(
                    "reactant and product must have identical atom symbols in identical order"
                )
            if self.structure.charge != self.product_structure.charge:
                raise ValueError("reactant and product must have the same charge")
            if self.structure.spin != self.product_structure.spin:
                raise ValueError("reactant and product must have the same spin multiplicity")
            if self.transition_state_guess is not None:
                _validate_electron_parity(self.transition_state_guess)
                guess_symbols = [
                    atom.symbol for atom in self.transition_state_guess.atoms
                ]
                if guess_symbols != reactant_symbols:
                    raise ValueError(
                        "transition-state guess must have identical atom symbols "
                        "in identical order"
                    )
                if self.transition_state_guess.charge != self.structure.charge:
                    raise ValueError(
                        "transition-state guess must have the same charge"
                    )
                if self.transition_state_guess.spin != self.structure.spin:
                    raise ValueError(
                        "transition-state guess must have the same spin multiplicity"
                    )
            atom_count = len(self.structure.atoms)
            seen_coordinates: set[tuple[int, int, str]] = set()
            for coordinate in self.reaction_coordinates:
                if coordinate.atom_j >= atom_count:
                    raise ValueError(
                        "reaction_coordinates contains an out-of-range atom index"
                    )
                key = (coordinate.atom_i, coordinate.atom_j, coordinate.kind)
                if key in seen_coordinates:
                    raise ValueError("reaction_coordinates contains a duplicate pair")
                seen_coordinates.add(key)
        elif self.product_structure is not None or self.transition_state_guess is not None:
            raise ValueError(
                "product_structure and transition_state_guess are only valid for "
                "transition_state_search"
            )
        elif self.reaction_coordinates:
            raise ValueError(
                "reaction_coordinates are only valid for transition_state_search"
            )
        indices = self.parameters.frequency_atom_indices
        if indices is not None:
            if self.calculation != "frequency_analysis":
                raise ValueError(
                    "frequency_atom_indices is only valid for frequency_analysis"
                )
            if not indices:
                raise ValueError("frequency_atom_indices cannot be empty")
            if len(indices) != len(set(indices)):
                raise ValueError("frequency_atom_indices must be unique")
            if min(indices) < 0 or max(indices) >= len(self.structure.atoms):
                raise ValueError("frequency_atom_indices contains an out-of-range index")
        return self


class OptimizationFrame(BaseModel):
    step: int
    energy_ev: float
    max_force_ev_per_a: float
    structure: AtomicStructure


class CalculationProgress(BaseModel):
    step: int
    max_steps: int
    energy_ev: float
    max_force_ev_per_a: float
    optimization_device: Literal["cpu", "mps"]
    current_structure: AtomicStructure
    stage: Literal[
        "optimization",
        "fbenm",
        "dmf",
        "sella_mps",
        "sella_cpu",
        "geometric_cpu",
        "frequencies",
        "cpu_validation",
        "irc_forward",
        "irc_reverse",
        "irc_endpoint_a_relaxation",
        "irc_endpoint_b_relaxation",
    ] = "optimization"
    path_energies_ev: list[float] = Field(default_factory=list)
    reactant_endpoint: AtomicStructure | None = None
    product_endpoint: AtomicStructure | None = None


class CalculationResult(BaseModel):
    energy_ev: float
    initial_energy_ev: float | None = None
    forces_ev_per_a: list[tuple[float, float, float]]
    max_force_ev_per_a: float
    final_structure: AtomicStructure
    converged: bool | None = None
    optimization_steps: int = 0
    optimization_device: Literal["cpu", "mps"] | None = None
    optimizer: Literal["geomeTRIC-TRIC"] | None = None
    final_single_point_device: Literal["cpu"] = "cpu"
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)
    trajectory: list[OptimizationFrame] = Field(default_factory=list)


class FrequencyAnalysisResult(BaseModel):
    energy_ev: float
    max_force_ev_per_a: float
    structure: AtomicStructure
    active_atom_indices: list[int]
    frequencies_cm1: list[float]
    imaginary_frequencies_cm1: list[float]
    has_exactly_one_imaginary_mode: bool
    primary_imaginary_mode: list[tuple[float, float, float]] | None = None
    hessian_method: str = "finite_difference"
    hessian_asymmetry_relative: float = 0.0
    rigid_body_projection_applied: bool = False
    rigid_mode_count: int = 0
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class ReactionPathFrame(BaseModel):
    path_coordinate: float = Field(ge=0.0, le=1.0)
    structure: AtomicStructure


class EnergyProfilePoint(BaseModel):
    path_coordinate: float = Field(ge=0.0, le=1.0)
    energy_ev: float
    relative_energy_ev: float


class SaddleOptimizationFrame(BaseModel):
    stage: Literal["sella_mps", "sella_cpu", "geometric_cpu"]
    step: int
    energy_ev: float
    max_force_ev_per_a: float
    structure: AtomicStructure


class BondChangeAssessment(BaseModel):
    atom_i: int = Field(description="Zero-based atom index")
    atom_j: int = Field(description="Zero-based atom index")
    kind: Literal["breaking", "forming", "coordination", "inferred"] = "inferred"
    reactant_distance_a: float
    transition_state_distance_a: float
    product_distance_a: float
    path_fraction: float


class IrcPathFrameResult(BaseModel):
    path_coordinate: float = Field(ge=0.0, le=1.0)
    energy_ev: float
    relative_energy_ev: float
    max_force_ev_per_a: float
    converged: bool
    structure: AtomicStructure


class IrcEndpointRelaxationResult(BaseModel):
    endpoint: Literal["endpoint_a", "endpoint_b"]
    performed: bool
    converged: bool
    initial_energy_ev: float
    final_energy_ev: float
    final_max_force_ev_per_a: float
    initial_structure: AtomicStructure
    relaxed_endpoint_structure: AtomicStructure
    final_structure_label: Literal[
        "relaxed_contact_minimum",
        "unconverged_relaxation_endpoint",
    ]
    trajectory: list[OptimizationFrame] = Field(default_factory=list)
    error: str | None = None


class TransitionStateResult(BaseModel):
    scientific_outcome: str = "unvalidated_candidate"
    transition_state: AtomicStructure
    transition_state_energy_ev: float
    transition_state_forces_ev_per_a: list[tuple[float, float, float]]
    transition_state_max_force_ev_per_a: float
    reactant_energy_ev: float
    product_energy_ev: float
    forward_barrier_ev: float = Field(
        description=(
            "Validated saddle barrier when TS screening passes; otherwise the "
            "non-negative sampled-path rise"
        ),
    )
    reverse_barrier_ev: float = Field(
        description=(
            "Validated reverse saddle barrier when TS screening passes; "
            "otherwise the non-negative sampled-path rise"
        ),
    )
    barrier_source: Literal[
        "validated_transition_state",
        "converged_path_estimate",
        "unconverged_path_estimate",
        "legacy_candidate_energy",
    ] = "legacy_candidate_energy"
    barrier_is_validated: bool = False
    path: list[ReactionPathFrame]
    energy_profile: list[EnergyProfilePoint]
    highest_path_coordinate: float = Field(ge=0.0, le=1.0)
    path_has_interior_maximum: bool = Field(
        default=True,
        description=(
            "Whether the final CPU UMA path has an energy maximum away from both endpoints"
        )
    )
    dmf_iterations: int
    converged: bool = Field(description="DMF path optimization convergence only")
    transition_state_stationary: bool = Field(
        description="Whether the CPU force at the path maximum is <= 0.05 eV/A"
    )
    saddle_refinement_performed: bool
    saddle_refinement_converged: bool | None = None
    sella_mps_steps: int = 0
    sella_cpu_steps: int = 0
    saddle_trajectory: list[SaddleOptimizationFrame] = Field(default_factory=list)
    frequency_analysis_performed: bool = False
    frequency_scope: Literal["none", "partial", "full"] = "none"
    frequency_active_atom_indices: list[int] = Field(default_factory=list)
    frequencies_cm1: list[float] = Field(default_factory=list)
    imaginary_frequencies_cm1: list[float] = Field(default_factory=list)
    has_exactly_one_imaginary_mode: bool | None = None
    primary_imaginary_mode: list[tuple[float, float, float]] | None = None
    hessian_method: str | None = None
    hessian_asymmetry_relative: float | None = None
    rigid_body_projection_applied: bool | None = None
    rigid_mode_count: int | None = None
    imaginary_mode_reaction_overlap: float | None = None
    imaginary_mode_matches_reaction: bool | None = None
    reaction_definition_valid: bool | None = None
    geometry_progress_consistent: bool | None = None
    reaction_coordinate_consistent: bool | None = None
    bond_changes: list[BondChangeAssessment] = Field(default_factory=list)
    hessian_quality_valid: bool | None = None
    local_saddle_valid: bool | None = None
    irc_eligible: bool | None = None
    passes_uma_ts_screening: bool | None = None
    screening_diagnostic_codes: list[str] = Field(default_factory=list)
    screening_failure_code: str | None = None
    irc_performed: bool = False
    irc_validation_passed: bool | None = None
    irc_error: str | None = None
    irc_failure_code: str | None = None
    retry_actions: list[str] = Field(default_factory=list)
    irc_frames: list[IrcPathFrameResult] = Field(default_factory=list)
    irc_forward_frame_count: int = 0
    irc_reverse_frame_count: int = 0
    irc_forward_inner_loop_failed: bool | None = None
    irc_reverse_inner_loop_failed: bool | None = None
    irc_path_complete: bool | None = None
    irc_endpoint_distance_delta_a: float | None = None
    irc_relaxed_endpoint_distance_delta_a: float | None = None
    irc_raw_endpoint_matches_expected: bool | None = None
    irc_endpoint_matches_expected: bool | None = None
    irc_endpoint_assignment_final: bool = False
    irc_endpoint_a_role: str | None = None
    irc_endpoint_b_role: str | None = None
    irc_endpoint_a_relaxation: IrcEndpointRelaxationResult | None = None
    irc_endpoint_b_relaxation: IrcEndpointRelaxationResult | None = None
    separated_asymptote_status: Literal["not_computed", "computed"] = "not_computed"
    separated_fragment_energy_ev: float | None = None
    frequency_elapsed_seconds: float | None = None
    optimizer_status: str
    nmove: int
    update_teval: bool
    convergence: Literal["tight", "middle", "loose"]
    optimization_device: Literal["cpu", "mps"]
    final_single_point_device: Literal["cpu"] = "cpu"
    gaussian_input: str
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os "
    "Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md "
    "No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: index for index, symbol in enumerate(ELEMENT_SYMBOLS, start=1)}


def _atomic_number(symbol: str) -> int:
    number = ATOMIC_NUMBERS.get(symbol)
    if number is None:
        raise ValueError(f"unsupported element symbol: {symbol}")
    return number


def _validate_electron_parity(structure: AtomicStructure) -> None:
    electron_count = sum(_atomic_number(atom.symbol) for atom in structure.atoms)
    electron_count -= structure.charge
    unpaired_electrons = structure.spin - 1
    if electron_count <= 0:
        raise ValueError("charge leaves the structure with no electrons")
    if electron_count % 2 != unpaired_electrons % 2:
        raise ValueError("charge and spin multiplicity have inconsistent electron parity")


class Job(BaseModel):
    id: UUID
    status: JobStatus
    request: JobCreate
    created_at: datetime
    updated_at: datetime
    execution_envelope: ExecutionEnvelope | None = None
    idempotency_key: str | None = None
    canonical_request_hash: str | None = None
    service_managed: bool = False
    recovery_count: int = Field(default=0, ge=0)
    error: str | None = None
    result: CalculationResult | TransitionStateResult | FrequencyAnalysisResult | ActionResult | None = None
    progress: CalculationProgress | None = None
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    artifact_error: str | None = None


class JobEvent(BaseModel):
    """Small SSE payload; full results are fetched once after terminal events."""

    event_type: Literal["snapshot", "progress", "terminal"]
    job_id: UUID
    status: JobStatus
    updated_at: datetime
    progress: CalculationProgress | None = None
    error: str | None = None
    result_available: bool = False
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    artifact_error: str | None = None


def job_event_from_job(job: Job, *, initial: bool = False) -> JobEvent:
    terminal = job.status in {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    event_type: Literal["snapshot", "progress", "terminal"]
    if terminal:
        event_type = "terminal"
    elif initial:
        event_type = "snapshot"
    else:
        event_type = "progress"
    return JobEvent(
        event_type=event_type,
        job_id=job.id,
        status=job.status,
        updated_at=job.updated_at,
        progress=job.progress,
        error=job.error,
        result_available=job.result is not None,
        artifacts=job.artifacts,
        artifact_error=job.artifact_error,
    )


class JobResultSummary(BaseModel):
    kind: Literal["calculation", "transition_state", "frequency_analysis"]
    energy_ev: float | None = None
    max_force_ev_per_a: float | None = None
    converged: bool | None = None
    optimization_steps: int | None = None
    forward_barrier_ev: float | None = None
    barrier_is_validated: bool | None = None
    scientific_outcome: str | None = None
    imaginary_frequency_count: int | None = None
    passes_uma_ts_screening: bool | None = None
    irc_validation_passed: bool | None = None
    has_trajectory: bool = False
    has_reaction_path: bool = False
    has_irc_trajectory: bool = False
    has_imaginary_mode: bool = False


def result_summary_from_job(job: Job) -> JobResultSummary | None:
    result = job.result
    if isinstance(result, TransitionStateResult):
        return JobResultSummary(
            kind="transition_state",
            energy_ev=result.transition_state_energy_ev,
            max_force_ev_per_a=result.transition_state_max_force_ev_per_a,
            converged=result.converged,
            forward_barrier_ev=result.forward_barrier_ev,
            barrier_is_validated=result.barrier_is_validated,
            scientific_outcome=result.scientific_outcome,
            imaginary_frequency_count=len(result.imaginary_frequencies_cm1),
            passes_uma_ts_screening=result.passes_uma_ts_screening,
            irc_validation_passed=result.irc_validation_passed,
            has_trajectory=bool(result.saddle_trajectory),
            has_reaction_path=bool(result.path),
            has_irc_trajectory=bool(result.irc_frames),
            has_imaginary_mode=result.primary_imaginary_mode is not None,
        )
    if isinstance(result, CalculationResult):
        return JobResultSummary(
            kind="calculation",
            energy_ev=result.energy_ev,
            max_force_ev_per_a=result.max_force_ev_per_a,
            converged=result.converged,
            optimization_steps=result.optimization_steps,
            has_trajectory=bool(result.trajectory),
        )
    if isinstance(result, FrequencyAnalysisResult):
        return JobResultSummary(
            kind="frequency_analysis",
            energy_ev=result.energy_ev,
            max_force_ev_per_a=result.max_force_ev_per_a,
            imaginary_frequency_count=len(result.imaginary_frequencies_cm1),
            has_imaginary_mode=result.primary_imaginary_mode is not None,
        )
    return None


class JobSummary(BaseModel):
    id: UUID
    status: JobStatus
    request: JobCreate
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    progress: CalculationProgress | None = None
    result_summary: JobResultSummary | None = None
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    artifact_error: str | None = None


def job_summary_from_job(job: Job) -> JobSummary:
    return JobSummary(
        id=job.id,
        status=job.status,
        request=job.request,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        progress=job.progress,
        result_summary=result_summary_from_job(job),
        artifacts=job.artifacts,
        artifact_error=job.artifact_error,
    )


class JobListResponse(BaseModel):
    items: list[Job]
    total: int


class JobSummaryListResponse(BaseModel):
    items: list[JobSummary]
    total: int
    limit: int
    offset: int

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.contracts.actions import (
    ActionCapability,
    ActionType,
    AnalyzeFrequenciesAction,
    ComputeFullHessianAndModesAction,
    EvaluateEnergyForcesAction,
    GeneratePathCandidatesAction,
    OptimizePathCandidateAction,
    OptimizeMinimumAction,
    ProbePathCurvatureAction,
    RunCollectiveCoordinateScanAction,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
    TransitionStateWorkflowAction,
)


def _capability(
    *,
    action_type: ActionType,
    model: type[BaseModel],
    description: str,
    input_roles: list[str],
    output_kinds: list[str],
    prerequisite_codes: list[str] | None = None,
    possible_next_actions: list[ActionType] | None = None,
    validated: bool = False,
) -> ActionCapability:
    schema: dict[str, Any] = model.model_json_schema()
    return ActionCapability(
        action_type=action_type,
        description=description,
        input_roles=input_roles,
        output_kinds=output_kinds,
        prerequisite_codes=prerequisite_codes or [],
        possible_next_actions=possible_next_actions or [],
        request_schema=schema,
        validation_profile_id="uma-ts-strict-v1" if validated else None,
        validation_is_server_controlled=validated,
    )


ACTION_CAPABILITIES = (
    _capability(
        action_type="evaluate_energy_forces",
        model=EvaluateEnergyForcesAction,
        description="Evaluate UMA electronic energy and Cartesian forces.",
        input_roles=["structure"],
        output_kinds=["calculation_result", "final_structure"],
        possible_next_actions=["optimize_minimum", "analyze_frequencies"],
    ),
    _capability(
        action_type="optimize_minimum",
        model=OptimizeMinimumAction,
        description="Run an unconstrained local minimum optimization.",
        input_roles=["structure"],
        output_kinds=["calculation_result", "optimization_trajectory"],
        prerequisite_codes=["VALID_ELECTRON_CONFIGURATION"],
        possible_next_actions=["evaluate_energy_forces", "analyze_frequencies"],
    ),
    _capability(
        action_type="analyze_frequencies",
        model=AnalyzeFrequenciesAction,
        description=(
            "Compute a full or explicitly selected partial vibrational analysis."
        ),
        input_roles=["structure"],
        output_kinds=["frequency_analysis", "imaginary_mode"],
        prerequisite_codes=["VALID_ELECTRON_CONFIGURATION"],
        possible_next_actions=["optimize_minimum", "refine_saddle"],
    ),
    _capability(
        action_type="generate_path_candidates",
        model=GeneratePathCandidatesAction,
        description=(
            "Generate inspectable endpoint-to-endpoint path candidates without "
            "optimizing a path or refining a saddle."
        ),
        input_roles=["reactant", "product"],
        output_kinds=["path_candidate", "path_parameterization"],
        prerequisite_codes=[
            "MATCHING_ATOM_ORDER",
            "MATCHING_CHARGE_AND_SPIN",
        ],
        possible_next_actions=["optimize_path_candidate"],
    ),
    _capability(
        action_type="optimize_path_candidate",
        model=OptimizePathCandidateAction,
        description=(
            "Optimize one verified path candidate and recompute an unbiased CPU "
            "energy profile without performing saddle refinement."
        ),
        input_roles=["reactant", "product", "path_candidate_artifact"],
        output_kinds=[
            "optimized_path_candidate",
            "reaction_energy_profile",
            "saddle_seed",
        ],
        prerequisite_codes=[
            "SOURCE_ACTION_SUCCEEDED",
            "PATH_ARTIFACT_BINDINGS_VALID",
            "PATH_PARAMETERIZATION_COMPATIBLE",
        ],
        possible_next_actions=["optimize_path_candidate", "probe_path_curvature"],
    ),
    _capability(
        action_type="probe_path_curvature",
        model=ProbePathCurvatureAction,
        description=(
            "Probe local optimized-path tangents with finite-difference "
            "Hessian-vector products and return non-stationary saddle seeds."
        ),
        input_roles=["reactant", "product", "optimized_path_artifact"],
        output_kinds=["path_curvature_screen", "nonstationary_saddle_seed"],
        prerequisite_codes=[
            "SOURCE_ACTION_SUCCEEDED",
            "OPTIMIZED_PATH_BINDINGS_VALID",
            "PATH_INTERIOR_MAXIMUM",
        ],
        possible_next_actions=[
            "optimize_path_candidate",
            "run_collective_coordinate_scan",
            "refine_saddle",
        ],
    ),
    _capability(
        action_type="run_collective_coordinate_scan",
        model=RunCollectiveCoordinateScanAction,
        description=(
            "Relax a bounded scan of an explicit or reaction-derived linear "
            "combination of bond distances using unbiased CPU UMA diagnostics."
        ),
        input_roles=["reactant", "product", "optimized_path_artifact"],
        output_kinds=[
            "collective_coordinate_scan",
            "reaction_energy_profile",
            "saddle_seed",
        ],
        prerequisite_codes=[
            "SOURCE_ACTION_SUCCEEDED",
            "OPTIMIZED_PATH_BINDINGS_VALID",
            "COLLECTIVE_COORDINATE_DEFINED",
        ],
        possible_next_actions=[
            "run_collective_coordinate_scan",
            "refine_saddle",
        ],
    ),
    _capability(
        action_type="transition_state_workflow",
        model=TransitionStateWorkflowAction,
        description=(
            "Run the existing endpoint-to-endpoint transition-state macro workflow."
        ),
        input_roles=["reactant", "product", "optional_ts_guess"],
        output_kinds=["transition_state_result", "reaction_path", "irc_path"],
        prerequisite_codes=[
            "MATCHING_ATOM_ORDER",
            "MATCHING_CHARGE_AND_SPIN",
        ],
        possible_next_actions=[
            "refine_saddle",
            "compute_full_hessian_and_modes",
            "run_bidirectional_irc",
        ],
        validated=True,
    ),
    _capability(
        action_type="refine_saddle",
        model=RefineSaddleAction,
        description="Refine a supplied TS candidate with CPU UMA RS-P-RFO.",
        input_roles=["candidate_structure"],
        output_kinds=["saddle_candidate", "saddle_trajectory"],
        prerequisite_codes=["VALID_ELECTRON_CONFIGURATION"],
        possible_next_actions=["compute_full_hessian_and_modes"],
        validated=True,
    ),
    _capability(
        action_type="compute_full_hessian_and_modes",
        model=ComputeFullHessianAndModesAction,
        description="Compute and persist a full Hessian and normal modes.",
        input_roles=["stationary_candidate"],
        output_kinds=["full_hessian", "normal_modes", "frequency_analysis"],
        prerequisite_codes=["STATIONARY_CANDIDATE_RECOMMENDED"],
        possible_next_actions=["refine_saddle", "run_bidirectional_irc"],
        validated=True,
    ),
    _capability(
        action_type="run_bidirectional_irc",
        model=RunBidirectionalIrcAction,
        description="Run guarded bidirectional IRC from persisted Hessian/mode artifacts.",
        input_roles=["transition_state", "full_hessian", "imaginary_mode"],
        output_kinds=["irc_path", "relaxed_endpoints", "connectivity_report"],
        prerequisite_codes=[
            "FULL_HESSIAN_ARTIFACT",
            "EXACTLY_ONE_IMAGINARY_MODE",
            "MATCHING_PES_AND_STRUCTURE",
        ],
        possible_next_actions=["run_bidirectional_irc", "transition_state_workflow"],
        validated=True,
    ),
)

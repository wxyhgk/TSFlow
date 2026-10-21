"""Composable transition-state search algorithms."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "AlgorithmMetadata": "uma_workflows.transition_states.algorithm_api",
    "PathCandidate": "uma_workflows.transition_states.algorithm_api",
    "PathInitializationResult": "uma_workflows.transition_states.algorithm_api",
    "PathOptimizationResult": "uma_workflows.transition_states.algorithm_api",
    "PathParameterization": "uma_workflows.transition_states.algorithm_api",
    "ReactionPath": "uma_workflows.transition_states.algorithm_api",
    "TransitionStateProblem": "uma_workflows.transition_states.algorithm_api",
    "DirectMaxFluxConfig": "uma_workflows.transition_states.direct_maxflux",
    "DirectMaxFluxPathOptimizer": "uma_workflows.transition_states.direct_maxflux",
    "FbenmConfig": "uma_workflows.transition_states.fbenm",
    "FbenmPathInitializer": "uma_workflows.transition_states.fbenm",
    "SaddlePipelineConfig": "uma_workflows.transition_states.saddle_pipeline",
    "refine_path_saddle": "uma_workflows.transition_states.saddle_pipeline",
    "TransitionStateScreeningConfig": "uma_workflows.transition_states.screening",
    "TransitionStateScreeningResult": "uma_workflows.transition_states.screening",
    "screen_transition_state": "uma_workflows.transition_states.screening",
    "CoordinatedPathScore": "uma_workflows.transition_states.candidate_ranking",
    "coordinated_path_score": "uma_workflows.transition_states.candidate_ranking",
    "same_kind_progress_imbalance": "uma_workflows.transition_states.candidate_ranking",
    "select_best_candidate": "uma_workflows.transition_states.candidate_ranking",
    "bond_force_components": "uma_workflows.transition_states.bond_scan",
    "relaxed_bond_scan": "uma_workflows.transition_states.bond_scan",
    "CollectiveCoordinateTerm": "uma_workflows.transition_states.collective_scan",
    "collective_coordinate_value_and_gradient": "uma_workflows.transition_states.collective_scan",
    "collective_force_components": "uma_workflows.transition_states.collective_scan",
    "relaxed_collective_coordinate_scan": "uma_workflows.transition_states.collective_scan",
    "build_builtin_algorithm_registry": "uma_workflows.transition_states.builtin_algorithms",
    "analyze_mode_reaction_overlap": "uma_workflows.transition_states.mode_overlap",
    "classify_transition_outcome": "uma_workflows.transition_states.outcomes",
    "PathBias": "uma_workflows.transition_states.path_bias",
    "PathBiasedCalculator": "uma_workflows.transition_states.path_bias",
    "ReactionCoordinateKind": "uma_workflows.transition_states.reaction_coordinates",
    "ReactionCoordinateLike": "uma_workflows.transition_states.reaction_coordinates",
    "build_path_bias": "uma_workflows.transition_states.path_bias",
    "assess_bond_changes": "uma_workflows.transition_states.reaction_analysis",
    "reaction_active_region": "uma_workflows.transition_states.reaction_analysis",
    "align_product_to_stable_scaffold": "uma_workflows.transition_states.reference_paths",
    "generate_directional_reference_paths": "uma_workflows.transition_states.reference_paths",
    "coarse_path_barrier_score": "uma_workflows.transition_states.candidate_ranking",
    "dense_path_bias_value_and_gradient": "uma_workflows.transition_states.path_bias",
    "evaluate_path_energies": "uma_workflows.transition_states.path_analysis",
    "linear_path_has_collision": "uma_workflows.transition_states.path_geometry",
    "minimum_covalent_ratio": "uma_workflows.transition_states.path_geometry",
    "select_path_maximum": "uma_workflows.transition_states.path_analysis",
    "validate_path_geometry": "uma_workflows.transition_states.path_geometry",
    "IrcConfig": "uma_workflows.transition_states.workflow.models",
    "PathConfig": "uma_workflows.transition_states.workflow.models",
    "SaddleConfig": "uma_workflows.transition_states.workflow.models",
    "ScreeningConfig": "uma_workflows.transition_states.workflow.models",
    "TransitionSearchConfig": "uma_workflows.transition_states.workflow.models",
    "TransitionSearchWorkflowResult": "uma_workflows.transition_states.workflow.runner",
    "run_transition_state_search": "uma_workflows.transition_states.workflow.runner",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

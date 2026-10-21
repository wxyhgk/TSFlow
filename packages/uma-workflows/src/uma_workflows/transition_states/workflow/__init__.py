"""Reusable, API-independent transition-state workflow."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "Device": "uma_workflows.transition_states.workflow.models",
    "EndpointConfig": "uma_workflows.transition_states.workflow.models",
    "IrcConfig": "uma_workflows.transition_states.workflow.models",
    "IrcStageResult": "uma_workflows.transition_states.workflow.irc",
    "SaddleFallbackAttempt": "uma_workflows.transition_states.workflow.fallback",
    "SaddleFallbackResult": "uma_workflows.transition_states.workflow.fallback",
    "PathConfig": "uma_workflows.transition_states.workflow.models",
    "OptimizedPathCandidateResult": (
        "uma_workflows.transition_states.workflow.path_candidate"
    ),
    "PathCurvatureProbe": (
        "uma_workflows.transition_states.workflow.path_curvature"
    ),
    "PathCurvatureProbeResult": (
        "uma_workflows.transition_states.workflow.path_curvature"
    ),
    "SaddleConfig": "uma_workflows.transition_states.workflow.models",
    "ScreeningConfig": "uma_workflows.transition_states.workflow.models",
    "TransitionSearchConfig": "uma_workflows.transition_states.workflow.models",
    "TransitionSearchWorkflowResult": (
        "uma_workflows.transition_states.workflow.runner"
    ),
    "WorkflowProgress": "uma_workflows.transition_states.workflow.models",
    "WorkflowProgressCallback": (
        "uma_workflows.transition_states.workflow.models"
    ),
    "run_transition_state_search": (
        "uma_workflows.transition_states.workflow.runner"
    ),
    "build_supplied_guess_reference_path": (
        "uma_workflows.transition_states.workflow.path"
    ),
    "refine_saddle_guess": "uma_workflows.transition_states.workflow.saddle",
    "optimize_path_candidate": (
        "uma_workflows.transition_states.workflow.path_candidate"
    ),
    "probe_path_curvature": (
        "uma_workflows.transition_states.workflow.path_curvature"
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

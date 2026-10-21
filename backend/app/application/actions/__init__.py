"""Public action helpers; optional workflow code loads only on demand."""

from app.application.actions.mapping import action_record_from_job, action_to_job


def guidance_from_job(job):
    from app.application.actions.guidance import guidance_from_job as impl

    return impl(job)


SUPPORTED_ACTIONS = {
    "evaluate_energy_forces",
    "optimize_minimum",
    "analyze_frequencies",
    "transition_state_workflow",
    "generate_path_candidates",
    "optimize_path_candidate",
    "probe_path_curvature",
    "run_collective_coordinate_scan",
}


def _capabilities():
    from app.application.actions.capabilities import ACTION_CAPABILITIES as capabilities
    return capabilities


ACTION_CAPABILITIES = _capabilities()

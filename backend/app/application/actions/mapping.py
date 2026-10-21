from app.application.actions.validation_profiles import UMA_TS_STRICT_V1
from app.contracts.actions import (
    ActionRecord,
    ActionSpec,
    AnalyzeFrequenciesAction,
    ComputeFullHessianAndModesAction,
    EvaluateEnergyForcesAction,
    GeneratePathCandidatesAction,
    OptimizePathCandidateAction,
    OptimizeMinimumAction,
    ProbePathCurvatureAction,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
    RunCollectiveCoordinateScanAction,
    TransitionStateWorkflowAction,
)
from app.contracts.jobs import Job, JobCreate
from app.contracts.parameters import CalculationParameters


def action_to_job(action: ActionSpec) -> JobCreate:
    common = {
        "model_name": action.model_name,
        "task_name": action.task_name,
        "structure": action.structure,
    }
    if isinstance(action, EvaluateEnergyForcesAction):
        return JobCreate(calculation="single_point", **common)
    if isinstance(action, OptimizeMinimumAction):
        return JobCreate(
            calculation="geometry_optimization",
            parameters={
                "fmax": action.parameters.fmax,
                "max_steps": action.parameters.max_steps,
            },
            **common,
        )
    if isinstance(action, AnalyzeFrequenciesAction):
        return JobCreate(
            calculation="frequency_analysis",
            parameters={
                "frequency_atom_indices": action.parameters.atom_indices,
                "frequency_delta_a": action.parameters.delta_a,
                "imaginary_frequency_threshold_cm1": (
                    action.parameters.imaginary_threshold_cm1
                ),
                "hessian_method": action.parameters.hessian_method,
                "frequency_max_atoms": (action.parameters.finite_difference_max_atoms),
            },
            **common,
        )
    if isinstance(action, TransitionStateWorkflowAction):
        parameters = action.parameters.model_copy(
            update={
                "sella_cpu_fmax": UMA_TS_STRICT_V1.stationary_fmax_ev_per_a,
                "imaginary_frequency_threshold_cm1": (
                    UMA_TS_STRICT_V1.imaginary_frequency_threshold_cm1
                ),
                "maximum_hessian_asymmetry": (
                    UMA_TS_STRICT_V1.maximum_hessian_asymmetry
                ),
                "irc_fmax": UMA_TS_STRICT_V1.irc_path_fmax_ev_per_a,
                "irc_inner_fmax": UMA_TS_STRICT_V1.irc_inner_fmax_ev_per_a,
                "irc_endpoint_relaxation_fmax": (
                    UMA_TS_STRICT_V1.irc_endpoint_fmax_ev_per_a
                ),
            }
        )
        return JobCreate(
            calculation="transition_state_search",
            product_structure=action.product_structure,
            transition_state_guess=action.transition_state_guess,
            reaction_coordinates=action.reaction_coordinates,
            parameters=parameters,
            **common,
        )
    if isinstance(action, GeneratePathCandidatesAction):
        parameters = CalculationParameters(
            path_initializer_id=action.parameters.initializer_id,
            nmove=action.parameters.nmove,
            path_ensemble_size=action.parameters.ensemble_size,
            path_bias_mode=action.parameters.path_bias_mode,
        )
        return JobCreate(
            calculation="transition_state_search",
            product_structure=action.product_structure,
            reaction_coordinates=action.reaction_coordinates,
            parameters=parameters,
            **common,
        )
    if isinstance(action, OptimizePathCandidateAction):
        parameters = CalculationParameters(
            path_optimizer_id=action.parameters.optimizer_id,
            nmove=action.parameters.nmove,
            dmf_max_iterations=action.parameters.max_iterations,
            dmf_convergence=action.parameters.convergence,
            path_points=action.parameters.path_points,
            update_teval=action.parameters.update_teval,
            dmf_accept_every_trial_step=action.parameters.accept_every_trial_step,
            path_bias_mode=action.parameters.path_bias_mode,
        )
        return JobCreate(
            calculation="transition_state_search",
            product_structure=action.product_structure,
            reaction_coordinates=action.reaction_coordinates,
            parameters=parameters,
            **common,
        )
    if isinstance(action, (ProbePathCurvatureAction, RunCollectiveCoordinateScanAction)):
        return JobCreate(
            calculation="transition_state_search",
            product_structure=action.product_structure,
            reaction_coordinates=action.reaction_coordinates,
            **common,
        )
    if isinstance(action, ComputeFullHessianAndModesAction):
        return JobCreate(
            calculation="frequency_analysis",
            parameters={
                "frequency_delta_a": action.parameters.delta_a,
                "imaginary_frequency_threshold_cm1": (
                    action.parameters.imaginary_threshold_cm1
                ),
                "hessian_method": action.parameters.hessian_method,
                "frequency_max_atoms": (action.parameters.finite_difference_max_atoms),
                "maximum_hessian_asymmetry": (
                    action.parameters.maximum_hessian_asymmetry
                ),
            },
            **common,
        )
    if isinstance(action, (RefineSaddleAction, RunBidirectionalIrcAction)):
        # Compatibility request used for model lookup and legacy job views;
        # the typed ExecutionEnvelope selects the primitive executor.
        return JobCreate(calculation="single_point", **common)
    raise TypeError(f"unsupported action type: {type(action).__name__}")


def action_record_from_job(job: Job, *, api_prefix: str) -> ActionRecord:
    envelope = job.execution_envelope
    if envelope is not None and envelope.payload.execution_type == "action":
        action_type = envelope.payload.action.action_type
    else:
        action_type = {
            "single_point": "evaluate_energy_forces",
            "geometry_optimization": "optimize_minimum",
            "frequency_analysis": "analyze_frequencies",
            "transition_state_search": "transition_state_workflow",
        }[job.request.calculation]
    return ActionRecord(
        id=job.id,
        action_type=action_type,
        status=job.status,
        job_id=job.id,
        job_url=f"{api_prefix}/jobs/{job.id}",
        events_url=f"{api_prefix}/jobs/{job.id}/events",
        guidance_url=f"{api_prefix}/actions/{job.id}/guidance",
        created_at=job.created_at,
        updated_at=job.updated_at,
    )

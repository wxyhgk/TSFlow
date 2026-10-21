from __future__ import annotations

from app.application.actions.guidance_common import finding, guidance, next_action
from app.contracts.action_results import (
    BidirectionalIrcActionResult,
    HessianModesActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.actions import ActionGuidance
from app.contracts.job_records import Job
from app.contracts.transition_states import TransitionStateResult


def saddle_guidance(job: Job, result: SaddleRefinementActionResult) -> ActionGuidance:
    if result.stationary:
        return guidance(
            job,
            "candidate",
            findings=[finding(
                "STATIONARY_CANDIDATE",
                "info",
                "The force threshold is met, but a full Hessian is still required.",
                max_force_ev_per_a=result.max_force_ev_per_a,
            )],
            next_actions=[next_action(
                "compute_full_hessian_and_modes",
                "FULL_HESSIAN_REQUIRED",
                "Compute a full Hessian before classifying or following the saddle.",
                structure="result.structure",
            )],
        )
    return guidance(
        job,
        "requires_refinement",
        findings=[finding(
            "TS_NOT_STATIONARY",
            "warning",
            "The candidate remains above the server force threshold.",
            max_force_ev_per_a=result.max_force_ev_per_a,
        )],
        next_actions=[next_action(
            "refine_saddle",
            "CONTINUE_SADDLE_REFINEMENT",
            "Continue from the latest candidate with a bounded refinement budget.",
            structure="result.structure",
        )],
    )


def hessian_guidance(job: Job, result: HessianModesActionResult) -> ActionGuidance:
    if result.pes_index1_saddle_valid:
        return guidance(
            job,
            "index1_saddle",
            findings=[finding(
                "INDEX1_SADDLE_VALID",
                "info",
                "The full Hessian supports a stationary first-order saddle on this PES.",
                imaginary_frequency_count=len(result.imaginary_frequencies_cm1),
            )],
            next_actions=[next_action(
                "run_bidirectional_irc",
                "IRC_CONNECTIVITY_REQUIRED",
                "Follow both directions to discover or validate endpoint connectivity.",
                structure="result.structure",
                hessian_artifact="result.hessian_artifact",
                mode_artifact="result.mode_artifact",
            )],
        )
    code = _hessian_failure_code(result)
    return guidance(
        job,
        "requires_refinement",
        findings=[finding(
            code,
            "warning",
            "The candidate does not satisfy the complete index-1 saddle profile.",
            stationary=result.stationary,
            imaginary_frequency_count=len(result.imaginary_frequencies_cm1),
            hessian_quality_valid=result.hessian_quality_valid,
        )],
        next_actions=[next_action(
            "refine_saddle",
            "TRY_ALTERNATIVE_OR_CONTINUED_REFINEMENT",
            "Refine a selected candidate or alternative seed before recomputing the Hessian.",
            structure="result.structure",
        )],
    )


def irc_guidance(job: Job, result: BidirectionalIrcActionResult) -> ActionGuidance:
    endpoints_converged = (
        result.endpoint_a_relaxation.converged
        and result.endpoint_b_relaxation.converged
    )
    if result.path_complete and endpoints_converged:
        if result.endpoint_match_evaluated and result.endpoint_matches_expected is True:
            state = "validated_transition_state"
            code = "VALIDATED_TRANSITION_STATE"
        elif result.endpoint_match_evaluated and result.endpoint_matches_expected is False:
            state = "mechanism_mismatch"
            code = "IRC_TOPOLOGY_MISMATCH"
        else:
            state = "irc_connected_saddle"
            code = "IRC_CONNECTED_SADDLE"
        severity = "warning" if state == "mechanism_mismatch" else "info"
        return guidance(job, state, findings=[finding(
            code,
            severity,
            "IRC endpoint roles are separate from local saddle validity.",
        )])
    return guidance(
        job,
        "inconclusive",
        findings=[finding(
            "IRC_INCOMPLETE" if not result.path_complete else "IRC_ENDPOINT_UNCONVERGED",
            "warning",
            "Incomplete IRC evidence must not be labelled final endpoint validation.",
        )],
        next_actions=[next_action(
            "run_bidirectional_irc",
            "RETRY_IRC_WITH_BOUNDED_BUDGET",
            "Retry from the same validated Hessian and mode with a larger bounded budget.",
            structure="action.structure",
            hessian_artifact="action.hessian_artifact",
            mode_artifact="action.mode_artifact",
        )],
    )


def transition_workflow_guidance(
    job: Job,
    result: TransitionStateResult,
) -> ActionGuidance:
    findings = [
        finding(code, "warning", "Transition-state workflow diagnostic.")
        for code in result.screening_diagnostic_codes
    ]
    if result.scientific_outcome == "validated_transition_state":
        return guidance(job, "validated_transition_state", findings=findings)
    if result.local_saddle_valid:
        findings.append(finding(
            "INDEX1_SADDLE_VALID",
            "info",
            "Persist a bound Hessian and mode artifact before stepwise IRC.",
        ))
        return guidance(
            job,
            "index1_saddle",
            findings=findings,
            next_actions=[next_action(
                "compute_full_hessian_and_modes",
                "PERSIST_REUSABLE_HESSIAN",
                "Persist reusable Hessian and mode artifacts for independent IRC.",
                structure="result.transition_state",
            )],
        )
    if result.scientific_outcome == "near_barrierless":
        findings.append(finding(
            "NEAR_BARRIERLESS_PATH", "info", result.scientific_outcome_reason
        ))
        return guidance(job, "completed", findings=findings)
    return guidance(
        job,
        "requires_refinement",
        findings=findings,
        next_actions=[next_action(
            "refine_saddle",
            result.screening_failure_code or "REFINE_SELECTED_CANDIDATE",
            "Refine the reported candidate or an agent-selected alternative seed.",
            structure="result.transition_state",
        )],
    )


def _hessian_failure_code(result: HessianModesActionResult) -> str:
    if not result.stationary:
        return "TS_NOT_STATIONARY"
    if not result.hessian_quality_valid:
        return "HESSIAN_QUALITY_INVALID"
    count = len(result.imaginary_frequencies_cm1)
    if count == 0:
        return "SADDLE_COLLAPSED_TO_MINIMUM"
    if count > 1:
        return "MULTIPLE_IMAGINARY_MODES"
    return "INDEX1_SADDLE_INVALID"

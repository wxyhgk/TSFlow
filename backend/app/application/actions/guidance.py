from __future__ import annotations

from app.application.actions.guidance_common import finding, guidance, next_action
from app.application.actions.ts_guidance import (
    hessian_guidance,
    irc_guidance,
    saddle_guidance,
    transition_workflow_guidance,
)
from app.contracts.action_results import (
    BidirectionalIrcActionResult,
    GeneratePathCandidatesActionResult,
    HessianModesActionResult,
    OptimizePathCandidateActionResult,
    ProbePathCurvatureActionResult,
    RunCollectiveCoordinateScanActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.actions import ActionGuidance
from app.contracts.calculations import CalculationResult
from app.contracts.frequencies import FrequencyAnalysisResult
from app.contracts.job_records import Job, JobStatus
from app.contracts.transition_states import TransitionStateResult


def guidance_from_job(job: Job) -> ActionGuidance:
    if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        return guidance(job, "pending", findings=[finding(
            "ACTION_PENDING", "info", "The action has not reached a terminal state."
        )])
    if job.status == JobStatus.CANCELLED:
        return guidance(job, "cancelled", findings=[finding(
            "ACTION_CANCELLED", "warning", "The action was cancelled."
        )])
    if job.status == JobStatus.FAILED:
        return guidance(job, "execution_failed", findings=[finding(
            "ACTION_EXECUTION_FAILED",
            "error",
            "Execution failed before a scientific result was produced.",
            error=job.error,
        )])
    result = job.result
    if isinstance(result, GeneratePathCandidatesActionResult):
        return guidance(
            job,
            "candidate",
            findings=[finding(
                "PATH_CANDIDATES_READY",
                "info",
                "Path candidates are ready for explicit inspection and selection.",
                candidate_count=len(result.candidates),
            )],
        )
    if isinstance(result, OptimizePathCandidateActionResult):
        if result.has_interior_maximum:
            return guidance(job, "candidate", findings=[finding(
                "PATH_INTERIOR_MAXIMUM",
                "info",
                "The optimized path contains an interior energy maximum suitable for curvature screening.",
                path_coordinate=result.highest_path_coordinate,
            )], next_actions=[next_action(
                "probe_path_curvature",
                "PATH_CURVATURE_PROBE",
                "Probe the optimized path around its interior maximum.",
                structure="result.highest_structure",
                optimized_path="action.optimized_path_artifact",
            )])
        return guidance(job, "inconclusive", findings=[finding(
            "PATH_NO_INTERIOR_MAXIMUM", "warning", "The optimized path has no interior energy maximum."
        )], next_actions=[next_action(
            "run_collective_coordinate_scan",
            "COLLECTIVE_SCAN_FALLBACK",
            "Run a bounded collective-coordinate scan.",
            optimized_path="action.optimized_path_artifact",
        )])
    if isinstance(result, ProbePathCurvatureActionResult):
        if result.selected_structure is not None:
            return guidance(job, "candidate", next_actions=[next_action(
                "refine_saddle", "NEGATIVE_CURVATURE_SEED",
                "Refine the selected negative-curvature seed.",
                structure="result.selected_structure",
            )])
        return guidance(job, "inconclusive", next_actions=[next_action(
            "run_collective_coordinate_scan", "CURVATURE_SCAN_FALLBACK",
            "Run a bounded collective-coordinate scan.",
            optimized_path="action.source_optimized_path_artifact",
        )])
    if isinstance(result, RunCollectiveCoordinateScanActionResult):
        if result.selected_structure is not None:
            return guidance(job, "candidate", next_actions=[next_action(
                "refine_saddle", "COLLECTIVE_SCAN_SEED",
                "Refine the selected scan maximum.",
                structure="result.selected_structure",
            )])
        return guidance(job, "inconclusive", next_actions=[next_action(
            "run_collective_coordinate_scan", "COLLECTIVE_SCAN_RETRY",
            "Adjust the collective coordinate or scan targets.",
            optimized_path="result.source_optimized_path_artifact",
        )])
    if isinstance(result, SaddleRefinementActionResult):
        return saddle_guidance(job, result)
    if isinstance(result, HessianModesActionResult):
        return hessian_guidance(job, result)
    if isinstance(result, BidirectionalIrcActionResult):
        return irc_guidance(job, result)
    if isinstance(result, TransitionStateResult):
        return transition_workflow_guidance(job, result)
    if isinstance(result, CalculationResult):
        return _calculation_guidance(job, result)
    if isinstance(result, FrequencyAnalysisResult):
        code = (
            "EXACTLY_ONE_IMAGINARY_MODE"
            if result.has_exactly_one_imaginary_mode
            else "FREQUENCY_ANALYSIS_COMPLETE"
        )
        return guidance(job, "candidate", findings=[finding(
            code, "info", "Frequency evidence is available for agent interpretation."
        )])
    return guidance(job, "inconclusive", findings=[finding(
        "ACTION_RESULT_UNAVAILABLE",
        "warning",
        "The action succeeded but no typed scientific result is available.",
    )])


def _calculation_guidance(job: Job, result: CalculationResult) -> ActionGuidance:
    if result.converged is False:
        return guidance(
            job,
            "requires_refinement",
            findings=[finding(
                "MINIMUM_NOT_CONVERGED", "warning", "Optimization did not converge."
            )],
            next_actions=[next_action(
                "optimize_minimum",
                "CONTINUE_MINIMUM_OPTIMIZATION",
                "Continue optimization from the latest structure.",
                structure="result.final_structure",
            )],
        )
    return guidance(job, "completed")

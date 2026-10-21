"""Decode and forward messages emitted by the calculation child process."""

import asyncio
import queue
from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.contracts.jobs import (
    CalculationProgress,
    CalculationResult,
    FrequencyAnalysisResult,
    TransitionStateResult,
)
from app.contracts.action_results import (
    ActionResult,
    BidirectionalIrcActionResult,
    GeneratePathCandidatesActionResult,
    OptimizePathCandidateActionResult,
    ProbePathCurvatureActionResult,
    RunCollectiveCoordinateScanActionResult,
    HessianModesActionResult,
    SaddleRefinementActionResult,
)
from app.infrastructure.persistence.protocol import JobRepository


WorkerResult = (
    CalculationResult | FrequencyAnalysisResult | TransitionStateResult | ActionResult
)
_POLL_SECONDS = 0.25
_RESULT_TYPES = {
    "CalculationResult": CalculationResult,
    "FrequencyAnalysisResult": FrequencyAnalysisResult,
    "TransitionStateResult": TransitionStateResult,
    "SaddleRefinementActionResult": SaddleRefinementActionResult,
    "HessianModesActionResult": HessianModesActionResult,
    "BidirectionalIrcActionResult": BidirectionalIrcActionResult,
    "GeneratePathCandidatesActionResult": GeneratePathCandidatesActionResult,
    "OptimizePathCandidateActionResult": OptimizePathCandidateActionResult,
    "ProbePathCurvatureActionResult": ProbePathCurvatureActionResult,
    "RunCollectiveCoordinateScanActionResult": RunCollectiveCoordinateScanActionResult,
}


async def receive_worker_result(
    output_queue: Any,
    process_is_alive: Callable[[], bool],
    repository: JobRepository,
    job_id: UUID,
) -> WorkerResult:
    while True:
        try:
            message = await asyncio.to_thread(
                output_queue.get,
                True,
                _POLL_SECONDS,
            )
        except queue.Empty:
            if not process_is_alive():
                raise RuntimeError("UMA calculation process exited without a result")
            continue
        kind = message[0]
        if kind == "progress":
            progress = CalculationProgress.model_validate_json(message[1])
            await repository.update_progress(job_id, progress)
            continue
        if kind == "error":
            raise RuntimeError(message[1].strip())
        if kind == "result":
            result_type = _RESULT_TYPES.get(message[1])
            if result_type is None:
                raise RuntimeError(f"unknown UMA result type: {message[1]}")
            return result_type.model_validate_json(message[2])
        raise RuntimeError(f"unknown UMA worker message: {kind}")

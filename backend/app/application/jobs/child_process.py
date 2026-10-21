"""Entry point that owns Torch and UMA allocations in a child process."""

import queue
import traceback
from pathlib import Path
from typing import Any

from app.contracts.actions import ExecutionEnvelope
from app.contracts.jobs import CalculationProgress


def calculation_process_main(
    input_queue: Any,
    output_queue: Any,
    cancel_event: Any,
    reference_dir: str,
    optimization_device: str,
    max_tasks: int,
    idle_timeout_seconds: float,
    artifact_root: str | None,
) -> None:
    from app.services.platform_executor import PlatformExecutor

    executor = PlatformExecutor(
        Path(reference_dir),
        optimization_device,  # type: ignore[arg-type]
        artifact_root=Path(artifact_root) if artifact_root else None,
    )
    completed = 0
    # max_tasks=0 means keep the process (and its UMA predictor cache) alive
    # until the idle timeout or an explicit shutdown signal.
    while max_tasks <= 0 or completed < max_tasks:
        # Parent owns the idle deadline so an arriving job cannot race with
        # a child independently exiting just before receiving the command.
        command = input_queue.get()
        if command is None:
            return
        envelope_data, checkpoint = command
        cancel_event.clear()
        try:
            envelope = ExecutionEnvelope.model_validate(envelope_data)

            def report_progress(progress: CalculationProgress) -> None:
                output_queue.put(("progress", progress.model_dump_json()))

            result = executor.execute_envelope(
                envelope,
                Path(checkpoint) if checkpoint else None,
                cancel_event,
                report_progress,
            )
            output_queue.put(
                ("result", type(result).__name__, result.model_dump_json())
            )
        except BaseException:
            output_queue.put(("error", traceback.format_exc()))
            return
        completed += 1

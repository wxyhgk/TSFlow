from __future__ import annotations

from pathlib import Path
from threading import Event
from collections.abc import Callable
from typing import Any

from app.contracts.jobs import (
    CalculationProgress,
    CalculationResult,
    FrequencyAnalysisResult,
    JobCreate,
    TransitionStateResult,
)
from app.contracts.action_results import ActionResult
from app.contracts.actions import (
    ComputeFullHessianAndModesAction,
    ExecutionEnvelope,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
)
from app.application.actions import action_to_job
from app.infrastructure.artifacts import FileArtifactStore
from uma_workflows.calculators import OptimizationDevice

from app.adapters.uma import UmaExecutor
from app.calculators.models import ModelRegistry


class PlatformExecutor:
    """Route jobs to the registered UMA, xTB, or Gaussian provider."""

    def __init__(
        self,
        reference_dir: Path,
        optimization_device: OptimizationDevice,
        *,
        artifact_root: Path | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._model_registry = model_registry or ModelRegistry(
            reference_dir.parent / "checkpoints"
        )
        self._uma = UmaExecutor(reference_dir, optimization_device)
        self._providers = {"uma": self._uma}
        self._dmf: dict[str, object] = {}
        self._artifact_root = artifact_root
        self._primitive = None

    def execute_envelope(
        self,
        envelope: ExecutionEnvelope,
        checkpoint: Path | None,
        cancel_event: Event,
        progress_callback: Callable[[CalculationProgress], None] | None = None,
    ) -> (
        CalculationResult
        | TransitionStateResult
        | FrequencyAnalysisResult
        | ActionResult
    ):
        payload = envelope.payload
        if payload.execution_type == "legacy_job":
            return self.execute(
                payload.request,
                checkpoint,
                cancel_event,
                progress_callback,
            )
        action = payload.action
        if action.action_type in {
            "evaluate_energy_forces",
            "optimize_minimum",
            "analyze_frequencies",
            "transition_state_workflow",
        }:
            return self.execute(
                action_to_job(action), checkpoint, cancel_event, progress_callback
            )
        from app.contracts.actions import (
            GeneratePathCandidatesAction,
            OptimizePathCandidateAction,
            ProbePathCurvatureAction,
            RunCollectiveCoordinateScanAction,
        )

        if isinstance(action, GeneratePathCandidatesAction):
            return self._require_primitive(action.model_name).generate_path_candidates(
                envelope.execution_id,
                action,
                checkpoint,
                cancel_event,
                progress_callback,
            )
        if isinstance(action, OptimizePathCandidateAction):
            reference = getattr(envelope.payload, "resolved_artifacts", {}).get(
                "path_candidate"
            )
            if reference is None:
                raise RuntimeError("path candidate artifact was not resolved")
            return self._require_primitive(action.model_name).optimize_path_candidate(
                envelope.execution_id,
                action,
                reference,
                checkpoint,
                cancel_event,
                progress_callback,
            )
        if isinstance(action, ProbePathCurvatureAction):
            reference = getattr(envelope.payload, "resolved_artifacts", {}).get(
                "optimized_path"
            )
            if reference is None:
                raise RuntimeError("optimized path artifact was not resolved")
            return self._require_primitive(action.model_name).probe_path_curvature(
                envelope.execution_id, action, reference, checkpoint, cancel_event, progress_callback
            )
        if isinstance(action, RunCollectiveCoordinateScanAction):
            reference = getattr(envelope.payload, "resolved_artifacts", {}).get(
                "optimized_path"
            )
            if reference is None:
                raise RuntimeError("optimized path artifact was not resolved")
            return self._require_primitive(action.model_name).run_collective_coordinate_scan(
                envelope.execution_id, action, reference, checkpoint, cancel_event, progress_callback
            )
        if isinstance(action, RefineSaddleAction):
            return self._require_primitive(action.model_name).refine_saddle(
                envelope.execution_id,
                action,
                checkpoint,
                cancel_event,
                progress_callback,
            )
        if isinstance(action, ComputeFullHessianAndModesAction):
            return self._require_primitive(
                action.model_name
            ).compute_full_hessian_and_modes(
                envelope.execution_id,
                action,
                checkpoint,
                cancel_event,
                progress_callback,
            )
        if isinstance(action, RunBidirectionalIrcAction):
            return self._require_primitive(action.model_name).run_bidirectional_irc(
                envelope.execution_id,
                action,
                checkpoint,
                cancel_event,
                progress_callback,
            )
        return self.execute(
            action_to_job(action),
            checkpoint,
            cancel_event,
            progress_callback,
        )

    def _require_primitive(self, model_name: str = "uma-s-1p2p1") -> Any:
        definition = self._model_registry.get_definition(model_name)
        if definition is not None and definition.engine != "uma":
            raise RuntimeError(
                f"primitive action execution is not implemented for {definition.engine}; "
                "use a standard calculation or select a UMA model"
            )
        if self._primitive is None and self._artifact_root is not None:
            from app.adapters.uma.primitives import UmaPrimitiveExecutor

            self._primitive = UmaPrimitiveExecutor(
                self._uma, FileArtifactStore(self._artifact_root)
            )
        if self._primitive is None:
            raise RuntimeError(
                "primitive Action execution requires a configured artifact store"
            )
        return self._primitive

    def execute(
        self,
        request: JobCreate,
        checkpoint: Path | None,
        cancel_event: Event,
        progress_callback: Callable[[CalculationProgress], None] | None = None,
    ) -> CalculationResult | TransitionStateResult | FrequencyAnalysisResult:
        executor = self._executor_for(request.model_name)
        if (
            request.calculation == "transition_state_search"
            and request.transition_state_guess is not None
        ):
            from app.adapters.uma.supplied_ts import execute_supplied_ts

            return execute_supplied_ts(
                executor, request, checkpoint, cancel_event, progress_callback
            )
        if request.calculation == "transition_state_search":
            engine = getattr(executor, "engine", "uma")
            if engine not in self._dmf:
                from app.application.transition_states import DmfExecutor

                self._dmf[engine] = DmfExecutor(executor)
            return self._dmf[engine].execute(
                request, checkpoint, cancel_event, progress_callback
            )
        return executor.execute(request, checkpoint, cancel_event, progress_callback)

    def _executor_for(self, model_name: str):
        definition = self._model_registry.get_definition(model_name)
        if definition is None:
            raise ValueError(f"unknown model_name: {model_name}")
        engine = definition.engine
        if engine not in self._providers:
            if engine == "xtb":
                from app.adapters.xtb import XtbExecutor

                self._providers[engine] = XtbExecutor()
            elif engine == "gaussian":
                from app.adapters.gaussian import GaussianExecutor

                self._providers[engine] = GaussianExecutor()
            else:
                raise ValueError(f"unsupported calculator engine: {engine}")
        return self._providers[engine]

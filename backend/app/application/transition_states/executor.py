"""Map API jobs to the API-independent transition-state workflow."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Callable, Literal

from app.adapters.uma import UmaExecutor
from app.contracts.jobs import (
    CalculationProgress,
    JobCreate,
    TransitionStateResult,
)
from app.services.transition_state_search.request_adapter import config_from_request
from app.services.transition_state_search.result_builder import build_result
from app.services.transition_state_search.structures import (
    progress_to_schema,
    to_atoms,
)
from uma_workflows.transition_states.workflow import run_transition_state_search


class TransitionStateExecutor:
    """Backend adapter for path, saddle, screening, and IRC stages.

    Scientific algorithms remain in ``uma_workflows``. This class only handles
    API structure conversion, calculator selection, progress conversion, and
    durable result serialization.
    """

    def __init__(self, calculator_provider: UmaExecutor) -> None:
        self._calculator_provider = calculator_provider

    def execute(
        self,
        request: JobCreate,
        checkpoint: Path | None,
        cancel_event: Event,
        progress_callback: Callable[[CalculationProgress], None] | None = None,
    ) -> TransitionStateResult:
        self._validate_request(request)
        if checkpoint is None:
            raise ValueError("transition-state search requires a model checkpoint")
        assert request.product_structure is not None

        started = time.perf_counter()
        device = self._calculator_provider.resolve_optimization_device()
        config = config_from_request(request)
        reactant = to_atoms(request.structure)
        product = to_atoms(request.product_structure)
        calculator = self._calculator_provider.calculator(
            checkpoint, device, request.task_name
        )
        cpu_calculator = self._calculator_provider.calculator(
            checkpoint, "cpu", request.task_name
        )
        analytic_hessian_calculator = self._analytic_hessian(
            request, checkpoint
        )
        guess = (
            to_atoms(request.transition_state_guess)
            if request.transition_state_guess is not None
            else None
        )

        def report(event) -> None:
            if progress_callback is not None:
                progress_callback(progress_to_schema(request.structure, event))

        workflow = run_transition_state_search(
            reactant,
            product,
            transition_state_guess=guess,
            reaction_coordinates=tuple(request.reaction_coordinates),
            config=config,
            device=device,
            calculator=calculator,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=report,
        )
        warnings = list(workflow.warnings)
        if device == "mps":
            warnings.append(
                "UMA path stages used experimental MPS acceleration; final "
                "energies and forces were recomputed on CPU."
            )
        return build_result(
            request,
            workflow.endpoints,
            workflow.path,
            workflow.saddle,
            workflow.screening,
            workflow.irc,
            workflow.outcome,
            device=device,
            started=started,
            warnings=warnings,
        )

    def _analytic_hessian(
        self,
        request: JobCreate,
        checkpoint: Path,
    ):
        if request.parameters.hessian_method == "finite_difference":
            return None
        try:
            return self._calculator_provider.hessian_calculator(
                checkpoint,
                request.task_name,
                use_vmap=len(request.structure.atoms) <= 30,
            )
        except Exception:
            if request.parameters.hessian_method == "analytic":
                raise
            return None

    @staticmethod
    def _validate_request(request: JobCreate) -> None:
        if request.calculation != "transition_state_search":
            raise ValueError(
                "TransitionStateExecutor only accepts transition_state_search jobs"
            )
        if request.product_structure is None:
            raise ValueError("transition-state search requires a product structure")


# Keep the historical class name available to callers while new code uses the
# domain-oriented name above.
DmfExecutor = TransitionStateExecutor

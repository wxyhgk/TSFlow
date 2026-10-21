from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from uma_workflows.cancellation import CancellationToken
from uma_workflows.irc import (
    BidirectionalIrcConfig,
    BidirectionalIrcResult,
    IrcValidationError,
    run_bidirectional_irc,
)
from uma_workflows.transition_states.screening import (
    TransitionStateScreeningResult,
)
from uma_workflows.transition_states.workflow.models import (
    IrcConfig,
    WorkflowProgress,
    WorkflowProgressCallback,
)

if TYPE_CHECKING:
    from uma_workflows.transition_states.workflow.path import PathStageResult
    from uma_workflows.transition_states.workflow.saddle import SaddleStageResult


@dataclass(frozen=True)
class IrcStageResult:
    performed: bool
    validation_passed: bool | None
    result: BidirectionalIrcResult | None
    error: str | None
    warnings: tuple[str, ...]
    failure_code: str | None = None
    retry_actions: tuple[str, ...] = ()


def follow_and_validate_irc(
    path: PathStageResult,
    saddle: SaddleStageResult,
    screening: TransitionStateScreeningResult,
    *,
    config: IrcConfig,
    cpu_calculator,
    cancel_event: CancellationToken,
    progress_callback: WorkflowProgressCallback | None,
) -> IrcStageResult:
    if not config.enabled:
        return IrcStageResult(False, None, None, None, ())
    if (
        hasattr(screening, "irc_eligible")
        and not screening.irc_eligible
    ) or getattr(screening, "frequency_scope", None) != "full":
        warning = (
            "IRC was requested but skipped because the candidate did not pass "
            "the full-Hessian first-order-saddle screen."
        )
        return IrcStageResult(False, False, None, warning, (warning,))
    frequency = screening.frequency_analysis
    if frequency is None:
        warning = "IRC was requested but no full Cartesian Hessian is available."
        return IrcStageResult(False, False, None, warning, (warning,))
    if frequency.primary_imaginary_mode is None:
        warning = "IRC was requested but no validated imaginary mode is available."
        return IrcStageResult(False, False, None, warning, (warning,))

    def report(direction: str, frame) -> None:
        if progress_callback is None:
            return
        endpoint_relaxation = direction.endswith("_relaxation")
        progress_callback(
            WorkflowProgress(
                stage=f"irc_{direction}",
                step=frame.step,
                max_steps=(
                    config.endpoint_relaxation_max_steps
                    if endpoint_relaxation
                    else config.steps
                ),
                energy_ev=frame.energy_ev,
                max_force_ev_per_a=frame.max_force_ev_per_a,
                device="cpu",
                positions_a=frame.positions_a.copy(),
                name=(
                    f"UMA IRC endpoint relaxation: {direction}"
                    if endpoint_relaxation
                    else f"UMA/{config.engine} IRC: {direction}"
                ),
            )
        )

    try:
        result = run_bidirectional_irc(
            saddle.atoms,
            frequency.hessian_ev_per_a2,
            validated_cartesian_mode=frequency.primary_imaginary_mode,
            calculator=cpu_calculator,
            expected_reactant=path.problem.reactant,
            expected_product=path.problem.product,
            config=BidirectionalIrcConfig(
                engine=config.engine,
                steps=config.steps,
                dx=config.dx,
                inner_steps=config.inner_steps,
                inner_fmax=config.inner_fmax,
                fmax=config.fmax,
                ts_fmax=path.problem.metadata.get(
                    "stationary_force_threshold_ev_per_a",
                    0.05,
                ),
                endpoint_distance_threshold_a=(config.endpoint_distance_threshold_a),
                require_endpoint_match=config.require_endpoint_match,
                relax_endpoints=config.relax_endpoints,
                endpoint_relaxation_fmax=config.endpoint_relaxation_fmax,
                endpoint_relaxation_max_steps=config.endpoint_relaxation_max_steps,
            ),
            cancel_event=cancel_event,
            on_frame=report,
        )
    except IrcValidationError as exc:
        message = str(exc)
        return IrcStageResult(True, False, None, message, (message,))
    validated = bool(
        result.path_complete
        and result.endpoint_a_relaxation.converged
        and result.endpoint_b_relaxation.converged
        and getattr(result.endpoint_topology, "matches_expected_endpoints", False)
        and getattr(result, "relaxed_endpoint_distance_delta_a", config.endpoint_distance_threshold_a)
        >= config.endpoint_distance_threshold_a
    )
    error = (
        None if validated else next(iter(result.warnings), "IRC_VALIDATION_INCOMPLETE")
    )
    return IrcStageResult(
        True,
        validated,
        result,
        error,
        result.warnings,
        error,
        () if validated else ("inspect_irc_diagnostics",),
    )

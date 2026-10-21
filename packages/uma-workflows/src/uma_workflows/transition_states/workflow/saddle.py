"""Saddle-stage adapters for path and supplied-guess workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.frequencies.workflow import numerical_frequency_analysis
from uma_workflows.optimization.geometric_saddle import refine_geometric_saddle
from uma_workflows.transition_states.saddle_pipeline import (
    SaddlePipelineConfig,
    refine_path_saddle,
)
from uma_workflows.transition_states.workflow.models import (
    Device,
    SaddleConfig,
    WorkflowProgress,
    WorkflowProgressCallback,
)
from uma_workflows.transition_states.workflow.path import PathStageResult


@dataclass(frozen=True)
class SaddleStageResult:
    atoms: Atoms
    refinement: object | None
    trajectory: tuple[object, ...]
    warnings: tuple[str, ...]


def refine_saddle(
    path_result: PathStageResult,
    transition_atoms: Atoms,
    *,
    config: SaddleConfig,
    device: Device,
    mps_calculator,
    cpu_calculator,
    analytic_hessian_calculator=None,
    cancel_event: CancellationToken,
    progress_callback: WorkflowProgressCallback | None,
) -> SaddleStageResult:
    # The path saddle implementation uses the path tangent and Sella stages;
    # analytic Hessians are consumed by the supplied-guess refinement and the
    # later screening stage. Keep this optional argument for one shared
    # fallback call contract.
    del analytic_hessian_calculator
    if not config.enabled or not path_result.has_interior_maximum:
        return SaddleStageResult(transition_atoms, None, (), ())
    trajectory: list[object] = []

    def report(frame) -> None:
        trajectory.append(frame)
        if progress_callback is None:
            return
        force = float(np.linalg.norm(frame.forces_ev_per_a, axis=1).max(initial=0.0))
        progress_callback(
            WorkflowProgress(
                stage=frame.stage,
                step=frame.step,
                max_steps=config.max_steps,
                energy_ev=frame.energy_ev,
                max_force_ev_per_a=force,
                device="mps" if frame.stage == "sella_mps" else "cpu",
                positions_a=frame.positions_a.copy(),
                name=f"UMA saddle refinement: {frame.stage}",
            )
        )

    result = refine_path_saddle(
        path_result.problem,
        transition_atoms,
        path=path_result.path,
        highest_coordinate=path_result.highest_coordinate,
        path_bias=path_result.bias,
        mps_calculator=mps_calculator,
        cpu_calculator=cpu_calculator,
        config=SaddlePipelineConfig(
            optimization_device=device,
            mps_fmax=config.mps_fmax,
            cpu_fmax=config.cpu_fmax,
            max_steps=config.max_steps,
            cpu_refine=config.cpu_refine,
            use_path_mode_hessian=config.path_mode_only,
            active_region_hops=config.active_region_hops,
            relaxed_scan=config.relaxed_scan,
            relaxed_scan_progress=config.relaxed_scan_progress,
            relaxed_scan_fmax=config.relaxed_scan_fmax,
            relaxed_scan_max_steps=config.relaxed_scan_max_steps,
        ),
        cancel_token=cancel_event,
        on_frame=report,
    )
    return SaddleStageResult(result.atoms, result.refinement, tuple(trajectory), result.warnings)


def refine_saddle_guess(
    atoms: Atoms,
    *,
    config: SaddleConfig,
    cpu_calculator,
    analytic_hessian_calculator=None,
    cancel_event: CancellationToken,
    progress_callback: WorkflowProgressCallback | None,
    hessian_method: str = "auto",
    finite_difference_max_atoms: int = 30,
) -> SaddleStageResult:
    """Refine a supplied geometry using a CPU Hessian and geomeTRIC."""
    raise_if_cancelled(cancel_event)
    candidate = atoms.copy()
    candidate.set_constraint()
    initial = numerical_frequency_analysis(
        candidate,
        calculator=cpu_calculator,
        analytic_calculator=analytic_hessian_calculator,
        hessian_method=hessian_method,
        delta_a=0.01,
        imaginary_threshold_cm1=100.0,
        cancel_event=cancel_event,
        allow_finite_difference_fallback=len(candidate) <= finite_difference_max_atoms,
    )
    warnings: list[str] = []
    if initial.hessian_fallback_reason:
        warnings.append(
            "Initial Hessian fell back to finite differences: "
            + initial.hessian_fallback_reason
        )
    trajectory: list[object] = []

    def report(frame) -> None:
        trajectory.append(frame)
        if progress_callback is not None:
            progress_callback(
                WorkflowProgress(
                    stage=frame.stage,
                    step=frame.step,
                    max_steps=config.max_steps,
                    energy_ev=frame.energy_ev,
                    max_force_ev_per_a=float(
                        np.linalg.norm(frame.forces_ev_per_a, axis=1).max(initial=0.0)
                    ),
                    device="cpu",
                    positions_a=frame.positions_a.copy(),
                    name="CPU geometric saddle refinement",
                )
            )

    refinement = refine_geometric_saddle(
        candidate,
        initial.hessian_ev_per_a2,
        calculator=cpu_calculator,
        fmax=config.cpu_fmax,
        max_steps=config.max_steps,
        cancel_event=cancel_event,
        on_frame=report,
    )
    if not refinement.converged:
        warnings.append("Supplied-guess saddle refinement did not converge.")
    return SaddleStageResult(candidate, refinement, tuple(trajectory), tuple(warnings))

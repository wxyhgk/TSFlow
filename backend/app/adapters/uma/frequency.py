"""UMA-backed frequency-analysis execution."""

from pathlib import Path
from threading import Event
from typing import Any, Callable

from uma_workflows.calculators import ANALYTIC_HESSIAN_VMAP_MAX_ATOMS
from uma_workflows.frequencies import numerical_frequency_analysis

from app.adapters.uma.structures import max_force, structure_from_atoms, to_atoms
from app.contracts.jobs import (
    CalculationProgress,
    FrequencyAnalysisResult,
    JobCreate,
)


def run_frequency_analysis(
    executor: Any,
    request: JobCreate,
    checkpoint: Path,
    cancel_event: Event,
    progress_callback: Callable[[CalculationProgress], None] | None,
) -> FrequencyAnalysisResult:
    atoms = to_atoms(request)
    warnings: list[str] = []
    active_indices = request.parameters.frequency_atom_indices
    if active_indices is None:
        active_indices = list(range(len(atoms)))
    if len(active_indices) < len(atoms):
        warnings.append(
            f"Partial Hessian used {len(active_indices)} active atoms; "
            f"{len(atoms) - len(active_indices)} environment atoms were fixed."
        )
    analytic_calculator = None
    if request.parameters.hessian_method != "finite_difference":
        try:
            analytic_calculator = executor.hessian_calculator(
                checkpoint,
                request.task_name,
                use_vmap=len(atoms) <= ANALYTIC_HESSIAN_VMAP_MAX_ATOMS,
            )
        except Exception as exc:
            if request.parameters.hessian_method == "analytic":
                raise
            warnings.append(
                "Analytic Hessian predictor could not be initialized; "
                f"finite differences will be used: {type(exc).__name__}: {exc}"
            )
    atoms.calc = analytic_calculator or executor._calculator(
        checkpoint,
        "cpu",
        request.task_name,
    )
    energy = float(atoms.get_potential_energy())
    force = max_force(atoms.get_forces())

    def report_frequency(coordinate: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                CalculationProgress(
                    step=coordinate,
                    max_steps=total,
                    energy_ev=energy,
                    max_force_ev_per_a=force,
                    optimization_device="cpu",
                    current_structure=structure_from_atoms(request, atoms),
                    stage="frequencies",
                )
            )

    analysis = numerical_frequency_analysis(
        atoms,
        calculator=atoms.calc,
        delta_a=request.parameters.frequency_delta_a,
        imaginary_threshold_cm1=request.parameters.imaginary_frequency_threshold_cm1,
        cancel_event=cancel_event,
        active_indices=active_indices,
        on_coordinate=report_frequency,
        analytic_calculator=analytic_calculator,
        hessian_method=request.parameters.hessian_method,
    )
    if analysis.hessian_fallback_reason:
        warnings.append(
            "Analytic Hessian fell back to finite differences: "
            f"{analysis.hessian_fallback_reason}"
        )
    if analysis.hessian_asymmetry_relative > request.parameters.maximum_hessian_asymmetry:
        warnings.append(
            "Hessian symmetry residual exceeds the configured quality limit "
            f"({analysis.hessian_asymmetry_relative:.3e})."
        )
    return FrequencyAnalysisResult(
        energy_ev=energy,
        max_force_ev_per_a=force,
        structure=structure_from_atoms(request, atoms),
        active_atom_indices=active_indices,
        frequencies_cm1=analysis.frequencies_cm1,
        imaginary_frequencies_cm1=analysis.imaginary_frequencies_cm1,
        has_exactly_one_imaginary_mode=len(analysis.imaginary_frequencies_cm1) == 1,
        primary_imaginary_mode=(
            [tuple(float(value) for value in row) for row in analysis.primary_imaginary_mode]
            if analysis.primary_imaginary_mode is not None
            else None
        ),
        hessian_method=analysis.hessian_method,
        hessian_asymmetry_relative=analysis.hessian_asymmetry_relative,
        rigid_body_projection_applied=analysis.rigid_body_projection_applied,
        rigid_mode_count=analysis.rigid_mode_count,
        elapsed_seconds=analysis.elapsed_seconds,
        warnings=warnings,
    )

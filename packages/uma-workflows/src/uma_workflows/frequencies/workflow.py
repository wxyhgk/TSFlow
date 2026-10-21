from __future__ import annotations

from typing import Callable

from ase import Atoms

from uma_workflows.cancellation import CancellationToken
from uma_workflows.frequencies.analysis import analyze_vibrational_hessian
from uma_workflows.frequencies.hessian import compute_numerical_hessian
from uma_workflows.frequencies.hessian import compute_analytic_hessian
from uma_workflows.frequencies.models import FrequencyAnalysis


def numerical_frequency_analysis(
    atoms: Atoms,
    *,
    calculator,
    delta_a: float,
    imaginary_threshold_cm1: float,
    cancel_event: CancellationToken,
    active_indices: list[int] | None = None,
    on_coordinate: Callable[[int, int], None] | None = None,
    analytic_calculator=None,
    hessian_method: str = "auto",
    project_rigid_body: bool | None = None,
    allow_finite_difference_fallback: bool = True,
) -> FrequencyAnalysis:
    """Compute a numerical Hessian and immediately analyze its spectrum."""
    if hessian_method not in {"auto", "analytic", "finite_difference"}:
        raise ValueError(f"unsupported Hessian method: {hessian_method}")
    if hessian_method == "analytic" and analytic_calculator is None:
        raise ValueError("analytic Hessian requested but no calculator is available")
    fallback_reason = None
    hessian = None
    if hessian_method != "finite_difference" and analytic_calculator is not None:
        try:
            hessian = compute_analytic_hessian(
                atoms,
                calculator=analytic_calculator,
                active_indices=active_indices,
            )
            if on_coordinate is not None:
                on_coordinate(1, 1)
        except Exception as exc:
            if hessian_method == "analytic":
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
    if hessian is None:
        if not allow_finite_difference_fallback:
            detail = (
                f" after analytic failure: {fallback_reason}"
                if fallback_reason
                else ""
            )
            raise RuntimeError(
                "finite-difference Hessian fallback is disabled" + detail
            )
        hessian = compute_numerical_hessian(
            atoms,
            calculator=calculator,
            delta_a=delta_a,
            cancel_event=cancel_event,
            active_indices=active_indices,
            on_coordinate=on_coordinate,
        )
    return analyze_vibrational_hessian(
        atoms,
        hessian.hessian_ev_per_a2,
        imaginary_threshold_cm1=imaginary_threshold_cm1,
        active_indices=hessian.active_atom_indices,
        elapsed_seconds=hessian.elapsed_seconds,
        hessian_method=hessian.method,
        hessian_asymmetry_relative=hessian.asymmetry_relative,
        hessian_fallback_reason=fallback_reason,
        project_rigid_body=project_rigid_body,
    )

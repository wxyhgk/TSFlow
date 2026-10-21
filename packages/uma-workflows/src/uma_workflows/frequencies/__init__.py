"""Numerical Hessian generation and vibrational analysis."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "FrequencyAnalysis": "uma_workflows.frequencies.models",
    "NumericalHessian": "uma_workflows.frequencies.models",
    "analyze_vibrational_hessian": "uma_workflows.frequencies.analysis",
    "compute_analytic_hessian": "uma_workflows.frequencies.hessian",
    "compute_numerical_hessian": "uma_workflows.frequencies.hessian",
    "numerical_frequency_analysis": "uma_workflows.frequencies.workflow",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

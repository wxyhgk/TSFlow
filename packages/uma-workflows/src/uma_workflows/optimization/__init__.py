"""Minimum and first-order saddle optimization workflows."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "FrequencyAnalysis": "uma_workflows.frequencies.models",
    "MinimumRefinement": "uma_workflows.optimization.models",
    "NumericalHessian": "uma_workflows.frequencies.models",
    "SaddleRefinement": "uma_workflows.optimization.models",
    "SellaFrame": "uma_workflows.optimization.models",
    "SellaStage": "uma_workflows.optimization.models",
    "OptimizationStage": "uma_workflows.optimization.models",
    "OptimizationStageResult": "uma_workflows.optimization.models",
    "analyze_vibrational_hessian": "uma_workflows.frequencies.analysis",
    "compute_numerical_hessian": "uma_workflows.frequencies.hessian",
    "numerical_frequency_analysis": "uma_workflows.frequencies.workflow",
    "refine_first_order_saddle": "uma_workflows.optimization.saddle",
    "refine_saddle_stages": "uma_workflows.optimization.saddle",
    "refine_geometric_saddle": "uma_workflows.optimization.geometric_saddle",
    "relax_constrained_minimum": "uma_workflows.optimization.minimum",
    "optimize_minimum_stages": "uma_workflows.optimization.minimum",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

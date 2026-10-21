"""UMA calculator construction and accelerator integration."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "ANALYTIC_HESSIAN_VMAP_MAX_ATOMS": "uma_workflows.calculators.uma",
    "OptimizationDevice": "uma_workflows.calculators.uma",
    "UmaCalculatorFactory": "uma_workflows.calculators.uma",
    "enable_experimental_mps": "uma_workflows.calculators.mps",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

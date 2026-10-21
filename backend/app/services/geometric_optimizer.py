"""Compatibility imports; use :mod:`app.adapters.uma.geometric` for new code."""

from app.adapters.uma.geometric import (
    EvaluationCallback,
    GeometricFrame,
    GeometricResult,
    UmaGeometricEngine,
    optimize_with_geometric,
)

__all__ = [
    "EvaluationCallback",
    "GeometricFrame",
    "GeometricResult",
    "UmaGeometricEngine",
    "optimize_with_geometric",
]

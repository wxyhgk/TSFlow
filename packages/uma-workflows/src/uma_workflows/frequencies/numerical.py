"""Compatibility imports for the former combined numerical frequency module."""

from uma_workflows.frequencies.analysis import analyze_vibrational_hessian
from uma_workflows.frequencies.hessian import (
    compute_analytic_hessian,
    compute_numerical_hessian,
)
from uma_workflows.frequencies.models import FrequencyAnalysis, NumericalHessian
from uma_workflows.frequencies.workflow import numerical_frequency_analysis

__all__ = [
    "FrequencyAnalysis",
    "NumericalHessian",
    "analyze_vibrational_hessian",
    "compute_analytic_hessian",
    "compute_numerical_hessian",
    "numerical_frequency_analysis",
]

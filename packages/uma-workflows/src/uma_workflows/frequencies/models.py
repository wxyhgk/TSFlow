from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NumericalHessian:
    hessian_ev_per_a2: np.ndarray
    active_atom_indices: list[int]
    elapsed_seconds: float
    method: str = "finite_difference"
    asymmetry_relative: float = 0.0
    fallback_reason: str | None = None


@dataclass(frozen=True)
class FrequencyAnalysis:
    frequencies_cm1: list[float]
    imaginary_frequencies_cm1: list[float]
    primary_imaginary_mode: np.ndarray | None
    imaginary_modes: list[np.ndarray]
    hessian_ev_per_a2: np.ndarray
    elapsed_seconds: float
    hessian_method: str = "finite_difference"
    hessian_asymmetry_relative: float = 0.0
    rigid_body_projection_applied: bool = False
    rigid_mode_count: int = 0
    hessian_fallback_reason: str | None = None

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.vibrations.data import VibrationsData

from uma_workflows.frequencies.models import FrequencyAnalysis
from uma_workflows.frequencies.geometric_adapter import (
    geometric_vibrational_modes,
)
from uma_workflows.frequencies.selection import validate_active_indices


def analyze_vibrational_hessian(
    atoms: Atoms,
    hessian_ev_per_a2: np.ndarray,
    *,
    imaginary_threshold_cm1: float,
    active_indices: list[int] | None = None,
    elapsed_seconds: float = 0.0,
    hessian_method: str = "finite_difference",
    hessian_asymmetry_relative: float = 0.0,
    hessian_fallback_reason: str | None = None,
    project_rigid_body: bool | None = None,
) -> FrequencyAnalysis:
    """Convert a Cartesian Hessian into frequencies and full-structure modes."""
    selected = validate_active_indices(atoms, active_indices)
    hessian = np.asarray(hessian_ev_per_a2, dtype=float)
    expected_shape = (3 * len(selected), 3 * len(selected))
    if hessian.shape != expected_shape:
        raise ValueError(
            f"Hessian shape {hessian.shape} does not match active coordinates "
            f"{expected_shape}"
        )
    hessian = 0.5 * (hessian + hessian.T)
    projection_applied = (
        len(selected) == len(atoms)
        if project_rigid_body is None
        else project_rigid_body
    )
    if projection_applied and len(selected) != len(atoms):
        raise ValueError("rigid-body projection requires a full-system Hessian")
    if projection_applied:
        frequencies, modes, rigid_mode_count = geometric_vibrational_modes(
            atoms,
            hessian,
        )
    else:
        vibration_data = VibrationsData.from_2d(atoms[selected], hessian)
        complex_frequencies = vibration_data.get_frequencies()
        frequencies = [
            -abs(float(value.imag)) if abs(value.imag) > 1.0e-8 else float(value.real)
            for value in complex_frequencies
        ]
        modes = vibration_data.get_modes(all_atoms=True)
        rigid_mode_count = 0
    imaginary_indices = [
        index
        for index, frequency in enumerate(frequencies)
        if frequency <= -imaginary_threshold_cm1
    ]
    imaginary_modes: list[np.ndarray] = []
    for imaginary_index in imaginary_indices:
        full_mode = np.zeros((len(atoms), 3), dtype=float)
        if projection_applied:
            full_mode = np.asarray(modes[imaginary_index], dtype=float)
        else:
            full_mode[selected] = np.asarray(modes[imaginary_index], dtype=float)
        imaginary_modes.append(full_mode)
    return FrequencyAnalysis(
        frequencies_cm1=frequencies,
        imaginary_frequencies_cm1=[frequencies[index] for index in imaginary_indices],
        primary_imaginary_mode=imaginary_modes[0] if imaginary_modes else None,
        imaginary_modes=imaginary_modes,
        hessian_ev_per_a2=hessian,
        elapsed_seconds=elapsed_seconds,
        hessian_method=hessian_method,
        hessian_asymmetry_relative=hessian_asymmetry_relative,
        rigid_body_projection_applied=projection_applied,
        rigid_mode_count=rigid_mode_count,
        hessian_fallback_reason=hessian_fallback_reason,
    )

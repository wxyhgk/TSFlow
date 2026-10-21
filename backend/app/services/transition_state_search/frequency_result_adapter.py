from __future__ import annotations


def frequency_result_fields(screening) -> dict:
    frequency = screening.frequency_analysis
    return {
        "frequency_analysis_performed": frequency is not None,
        "frequency_scope": screening.frequency_scope,
        "frequency_active_atom_indices": (
            list(screening.frequency_active_atom_indices) if frequency else []
        ),
        "frequencies_cm1": frequency.frequencies_cm1 if frequency else [],
        "imaginary_frequencies_cm1": (
            frequency.imaginary_frequencies_cm1 if frequency else []
        ),
        "has_exactly_one_imaginary_mode": (
            len(frequency.imaginary_frequencies_cm1) == 1 if frequency else None
        ),
        "primary_imaginary_mode": (
            [
                tuple(float(value) for value in row)
                for row in frequency.primary_imaginary_mode
            ]
            if frequency and frequency.primary_imaginary_mode is not None
            else None
        ),
        "hessian_method": frequency.hessian_method if frequency else None,
        "hessian_asymmetry_relative": (
            frequency.hessian_asymmetry_relative if frequency else None
        ),
        "rigid_body_projection_applied": (
            frequency.rigid_body_projection_applied if frequency else None
        ),
        "rigid_mode_count": frequency.rigid_mode_count if frequency else None,
        "frequency_elapsed_seconds": (
            frequency.elapsed_seconds if frequency else None
        ),
    }

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled


def evaluate_path_energies(
    positions: Iterable[np.ndarray],
    template: Atoms,
    calculator,
    *,
    cancel_event: CancellationToken,
) -> list[float]:
    """Evaluate path images on an unbiased validation calculator."""
    energies: list[float] = []
    for coordinates in positions:
        raise_if_cancelled(cancel_event)
        image = template.copy()
        image.set_positions(coordinates)
        image.calc = calculator
        energies.append(float(image.get_potential_energy()))
    return energies


def select_path_maximum(
    energies: list[float],
    *,
    minimum_prominence_ev: float = 1.0e-3,
) -> tuple[int, bool]:
    """Return the highest resolvable interior peak, including submerged barriers."""
    if len(energies) < 3:
        raise ValueError("path validation requires at least three energy points")
    values = np.asarray(energies, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CPU path validation produced non-finite energies")
    candidates: list[int] = []
    for index in range(1, len(values) - 1):
        if values[index] < values[index - 1] or values[index] < values[index + 1]:
            continue
        left_minimum = float(np.min(values[: index + 1]))
        right_minimum = float(np.min(values[index:]))
        prominence = min(
            float(values[index] - left_minimum),
            float(values[index] - right_minimum),
        )
        if prominence >= minimum_prominence_ev:
            candidates.append(index)
    if candidates:
        return max(candidates, key=lambda index: (values[index], -index)), True
    return int(np.argmax(values)), False

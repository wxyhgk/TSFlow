from __future__ import annotations

import numpy as np
from ase.data import atomic_numbers, covalent_radii


def minimum_covalent_ratio(
    positions,
    symbols: list[str],
) -> tuple[float, tuple[int, int], float]:
    coordinates = np.asarray(positions, dtype=float)
    if not np.isfinite(coordinates).all():
        return float("nan"), (0, 0), float("nan")
    radii = np.asarray(
        [covalent_radii[atomic_numbers[symbol]] for symbol in symbols]
    )
    best_ratio = float("inf")
    best_pair = (0, 0)
    best_distance = float("inf")
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            distances = np.linalg.norm(
                coordinates[:, first] - coordinates[:, second],
                axis=1,
            )
            distance = float(distances[int(np.argmin(distances))])
            ratio = distance / float(radii[first] + radii[second])
            if ratio < best_ratio:
                best_ratio = ratio
                best_pair = (first, second)
                best_distance = distance
    return best_ratio, best_pair, best_distance


def linear_path_has_collision(
    start,
    finish,
    symbols: list[str],
    *,
    collision_ratio: float = 0.70,
) -> bool:
    positions = np.asarray(
        [
            (1.0 - value) * start + value * finish
            for value in np.linspace(0.1, 0.9, 9)
        ]
    )
    return minimum_covalent_ratio(positions, symbols)[0] < collision_ratio


def validate_path_geometry(
    positions,
    symbols: list[str],
    *,
    stage: str,
    collision_ratio: float = 0.45,
) -> None:
    minimum_ratio, atom_pair, minimum_distance = minimum_covalent_ratio(
        positions,
        symbols,
    )
    if not np.isfinite(minimum_ratio):
        raise RuntimeError(f"{stage} path contains non-finite atomic coordinates")
    if minimum_ratio < collision_ratio:
        first, second = atom_pair
        raise RuntimeError(
            f"{stage} path contains an atomic collision between atoms "
            f"{first + 1} ({symbols[first]}) and {second + 1} ({symbols[second]}): "
            f"minimum distance {minimum_distance:.4f} A"
        )

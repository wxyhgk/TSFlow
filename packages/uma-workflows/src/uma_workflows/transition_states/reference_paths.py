from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.transition_states.path_geometry import (
    linear_path_has_collision,
)


@dataclass(frozen=True)
class ReferencePathEnsemble:
    candidates: list[list[Atoms]]
    warnings: tuple[str, ...]


def align_product_to_stable_scaffold(
    reactant: Atoms,
    product: Atoms,
) -> np.ndarray:
    """Rigidly align product atoms using the least-changing scaffold."""
    start = reactant.positions
    finish = product.positions
    atom_count = len(reactant)
    if atom_count == 1:
        return finish - finish[0] + start[0]

    start_distances = np.linalg.norm(
        start[:, None, :] - start[None, :, :],
        axis=-1,
    )
    finish_distances = np.linalg.norm(
        finish[:, None, :] - finish[None, :, :],
        axis=-1,
    )
    change_scores = np.mean(np.abs(start_distances - finish_distances), axis=1)
    hydrogen_penalty = np.asarray(
        [
            0.15 if symbol == "H" else 0.0
            for symbol in reactant.get_chemical_symbols()
        ]
    )
    stable_count = min(atom_count, max(2, int(np.ceil(atom_count * 0.6))))
    stable_indices = np.argsort(change_scores + hydrogen_penalty)[:stable_count]
    masses = reactant.get_masses()[stable_indices]
    start_center = np.average(start[stable_indices], axis=0, weights=masses)
    finish_center = np.average(finish[stable_indices], axis=0, weights=masses)
    start_vectors = start[stable_indices] - start_center
    finish_vectors = finish[stable_indices] - finish_center
    if (
        np.linalg.norm(start_vectors) < 1.0e-10
        or np.linalg.norm(finish_vectors) < 1.0e-10
    ):
        return finish - finish_center + start_center
    covariance = (finish_vectors * masses[:, None]).T @ start_vectors
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return (finish - finish_center) @ rotation + start_center


def generate_directional_reference_paths(
    reactant: Atoms,
    product: Atoms,
    *,
    ensemble_size: int,
) -> ReferencePathEnsemble:
    """Align endpoints and generate deterministic curved midpoint candidates."""
    if ensemble_size < 1:
        raise ValueError("reference path ensemble size must be positive")
    aligned_reactant = reactant.copy()
    aligned_product = product.copy()
    aligned_product.set_positions(
        align_product_to_stable_scaffold(aligned_reactant, aligned_product)
    )
    start = aligned_reactant.positions.copy()
    finish = aligned_product.positions.copy()
    base_midpoint = 0.5 * (start + finish)
    displacement = finish - start
    magnitudes = np.linalg.norm(displacement, axis=1)
    moving_index = int(np.argmax(magnitudes))
    maximum_displacement = float(magnitudes[moving_index])
    collision = linear_path_has_collision(
        start,
        finish,
        aligned_reactant.get_chemical_symbols(),
    )

    directions: list[np.ndarray | None] = [None]
    weights = np.zeros(len(aligned_reactant), dtype=float)
    amplitude = 0.0
    if maximum_displacement > 1.0e-8:
        direction = displacement[moving_index] / maximum_displacement
        axes = np.identity(3)
        helper_axis = axes[int(np.argmin(np.abs(axes @ direction)))]
        perpendicular_one = np.cross(direction, helper_axis)
        perpendicular_one /= np.linalg.norm(perpendicular_one)
        perpendicular_two = np.cross(direction, perpendicular_one)
        perpendicular_two /= np.linalg.norm(perpendicular_two)
        curved = [
            perpendicular_one,
            -perpendicular_one,
            perpendicular_two,
            -perpendicular_two,
        ]
        directions = curved if collision else [None, *curved]
        amplitude = min(1.5, max(0.60, 0.35 * maximum_displacement))
        weights = (magnitudes / maximum_displacement) ** 2

    candidates: list[list[Atoms]] = []
    for bend_direction in directions[:ensemble_size]:
        midpoint_positions = base_midpoint.copy()
        if bend_direction is not None:
            midpoint_positions += (
                weights[:, None] * amplitude * bend_direction[None, :]
            )
        midpoint = aligned_reactant.copy()
        midpoint.set_positions(midpoint_positions)
        candidates.append(
            [aligned_reactant.copy(), midpoint, aligned_product.copy()]
        )

    warnings: list[str] = []
    if collision:
        warnings.append(
            "Curved midpoint candidates were inserted because the aligned "
            "linear endpoint path contained an atomic-collision risk."
        )
    if len(candidates) > 1:
        warnings.append(
            f"Generated {len(candidates)} deterministic directional path "
            "candidates for a disconnected reaction endpoint."
        )
    return ReferencePathEnsemble(candidates=candidates, warnings=tuple(warnings))

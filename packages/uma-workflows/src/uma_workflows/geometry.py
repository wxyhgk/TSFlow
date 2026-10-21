from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from ase import Atoms


def set_nonperiodic_graph_cell(
    images: Iterable[Atoms],
    *,
    padding_a: float = 40.0,
) -> None:
    """Give non-periodic structures one shared graph-generation cell."""
    image_list = list(images)
    if not image_list:
        raise ValueError("at least one structure is required")
    all_positions = np.concatenate([image.positions for image in image_list], axis=0)
    span = all_positions.max(axis=0) - all_positions.min(axis=0)
    cell = tuple(float(value + padding_a) for value in span)
    for image in image_list:
        image.set_cell(cell)

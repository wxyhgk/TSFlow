from __future__ import annotations

from ase import Atoms


def validate_active_indices(
    atoms: Atoms,
    active_indices: list[int] | None,
) -> list[int]:
    selected = list(range(len(atoms))) if active_indices is None else list(active_indices)
    if not selected:
        raise ValueError("frequency active atom selection cannot be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("frequency active atom indices must be unique")
    if min(selected) < 0 or max(selected) >= len(atoms):
        raise ValueError("frequency active atom index is out of range")
    return selected

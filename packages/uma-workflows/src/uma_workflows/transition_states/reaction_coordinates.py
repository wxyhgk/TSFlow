from __future__ import annotations

from typing import Literal, Protocol


ReactionCoordinateKind = Literal["forming", "breaking", "coordination"]


class ReactionCoordinateLike(Protocol):
    """Structural interface accepted by transition-state algorithms."""

    atom_i: int
    atom_j: int
    kind: str

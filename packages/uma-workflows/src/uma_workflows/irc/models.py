from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IrcFrame:
    step: int
    energy_ev: float
    max_force_ev_per_a: float
    positions_a: np.ndarray
    converged: bool


@dataclass(frozen=True)
class IrcDirectionResult:
    frames: list[IrcFrame]
    inner_loop_failed: bool

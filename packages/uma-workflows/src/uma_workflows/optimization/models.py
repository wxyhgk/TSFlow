from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SellaStage = str


@dataclass(frozen=True)
class OptimizationStage:
    """One calculator/device stage in a reusable optimization pipeline."""

    name: str
    calculator: Any
    device: str = "cpu"
    fmax: float = 0.05
    max_steps: int = 100
    synchronize_after: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("optimization stage name cannot be empty")
        if self.fmax <= 0:
            raise ValueError("optimization stage fmax must be positive")
        if self.max_steps < 0:
            raise ValueError("optimization stage max_steps cannot be negative")


@dataclass(frozen=True)
class OptimizationStageResult:
    name: str
    device: str
    steps: int
    converged: bool


@dataclass(frozen=True)
class SellaFrame:
    stage: SellaStage
    step: int
    energy_ev: float
    forces_ev_per_a: np.ndarray
    positions_a: np.ndarray


@dataclass(frozen=True)
class SaddleRefinement:
    converged: bool
    mps_steps: int
    cpu_steps: int
    frames: list[SellaFrame]
    stage_results: tuple[OptimizationStageResult, ...] = ()


@dataclass(frozen=True)
class MinimumRefinement:
    converged: bool
    mps_steps: int
    cpu_steps: int
    stage_results: tuple[OptimizationStageResult, ...] = ()

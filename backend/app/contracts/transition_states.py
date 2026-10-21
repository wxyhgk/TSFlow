"""Transition-state result and collective-coordinate contracts."""

from typing import Annotated

from pydantic import BaseModel, Field

from app.contracts.jobs import (
    BondChangeAssessment,
    EnergyProfilePoint,
    IrcEndpointRelaxationResult,
    IrcPathFrameResult,
    ReactionPathFrame,
    SaddleOptimizationFrame,
    TransitionStateResult,
)


class CollectiveCoordinateTermInput(BaseModel):
    atom_i: int = Field(ge=0)
    atom_j: int = Field(ge=0)
    coefficient: float


class CollectiveCoordinateDefinition(BaseModel):
    terms: list[CollectiveCoordinateTermInput] = Field(min_length=1, max_length=64)


class CollectiveScanPointResult(BaseModel):
    target_value_a: float
    coordinate_value_a: float
    energy_ev: float
    relative_energy_ev: float = 0.0
    derivative_ev_per_a: float
    orthogonal_fmax_ev_per_a: float = Field(ge=0.0)
    converged: bool
    steps: int = Field(ge=0)


__all__ = [
    "BondChangeAssessment",
    "CollectiveCoordinateDefinition",
    "CollectiveCoordinateTermInput",
    "CollectiveScanPointResult",
    "EnergyProfilePoint",
    "IrcEndpointRelaxationResult",
    "IrcPathFrameResult",
    "ReactionPathFrame",
    "SaddleOptimizationFrame",
    "TransitionStateResult",
]

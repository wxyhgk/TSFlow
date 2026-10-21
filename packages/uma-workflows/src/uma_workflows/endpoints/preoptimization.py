"""Endpoint preoptimization helpers shared by transition-state workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.data import atomic_numbers, covalent_radii

from uma_workflows.cancellation import CancellationToken
from uma_workflows.optimization.models import OptimizationStage
from uma_workflows.optimization.minimum import optimize_minimum_stages


@dataclass(frozen=True)
class EndpointPreoptimization:
    converged: bool
    fragment_count: int
    mps_steps: int
    cpu_steps: int
    constraints_released: bool
    stage_results: tuple[Any, ...] = ()


def covalent_fragments(atoms: Atoms) -> list[list[int]]:
    """Return connected components using a conservative covalent graph."""
    symbols = atoms.get_chemical_symbols()
    radii = np.asarray(
        [covalent_radii[atomic_numbers[symbol]] for symbol in symbols],
        dtype=float,
    )
    distances = atoms.get_all_distances(mic=False)
    parent = list(range(len(atoms)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            if distances[first, second] <= 1.25 * (radii[first] + radii[second]):
                union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(atoms)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def preoptimize_endpoint_stages(
    atoms: Atoms,
    *,
    stages: Sequence[OptimizationStage],
    cancel_event: CancellationToken,
    preserved_bonds: Sequence[tuple[int, int]] = (),
    on_step: Callable[[OptimizationStage, int, int, Atoms], None] | None = None,
) -> EndpointPreoptimization:
    """Optimize an endpoint while temporarily preserving selected bonds."""
    original_constraints = list(atoms.constraints)
    if preserved_bonds:
        atoms.set_constraint(
            [FixBondLengths([(int(first), int(second)) for first, second in preserved_bonds])]
        )
    try:
        result = optimize_minimum_stages(
            atoms,
            stages,
            cancel_event=cancel_event,
            on_step=on_step,
        )
    finally:
        atoms.set_constraint(original_constraints)
    return EndpointPreoptimization(
        converged=result.converged,
        fragment_count=len(covalent_fragments(atoms)),
        mps_steps=result.mps_steps,
        cpu_steps=result.cpu_steps,
        constraints_released=not bool(atoms.constraints),
        stage_results=result.stage_results,
    )


def preoptimize_endpoint(
    atoms: Atoms,
    *,
    optimization_device: Literal["cpu", "mps"],
    mps_calculator,
    cpu_calculator,
    fmax: float,
    max_steps: int,
    cancel_event: CancellationToken,
    preserved_bonds: Sequence[tuple[int, int]] = (),
    on_step: Callable[[OptimizationStage, int, int, Atoms], None] | None = None,
) -> EndpointPreoptimization:
    """Compatibility adapter for the standard MPS then CPU endpoint flow."""
    stages: list[OptimizationStage] = []
    if optimization_device == "mps":
        stages.append(
            OptimizationStage(
                name="endpoint_mps",
                device="mps",
                calculator=mps_calculator,
                fmax=fmax,
                max_steps=max_steps,
            )
        )
    stages.append(
        OptimizationStage(
            name="endpoint_cpu",
            device="cpu",
            calculator=cpu_calculator,
            fmax=fmax,
            max_steps=max_steps,
        )
    )
    return preoptimize_endpoint_stages(
        atoms,
        stages=stages,
        cancel_event=cancel_event,
        preserved_bonds=preserved_bonds,
        on_step=on_step,
    )

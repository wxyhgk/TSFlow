from __future__ import annotations

from threading import Event

import numpy as np
from ase import Atoms

from app.contracts.jobs import BondChangeAssessment, ReactionCoordinate
from app.services.transition_state_search.structures import (
    max_force,
    progress_snapshot,
    schema_bond_changes,
    structure_from_atoms,
    structure_from_positions,
    to_atoms,
)
from app.services.uma_executor import CalculationCancelled
from uma_workflows.geometry import set_nonperiodic_graph_cell
from uma_workflows.transition_states import (
    PathBias,
    align_product_to_stable_scaffold,
    build_path_bias,
    coarse_path_barrier_score,
    evaluate_path_energies,
    generate_directional_reference_paths,
    linear_path_has_collision,
    minimum_covalent_ratio,
    reaction_active_region,
    select_path_maximum,
    validate_path_geometry,
)


class DmfCompatibilityMixin:
    """Temporary adapters for callers that used executor implementation details."""

    _to_atoms = staticmethod(to_atoms)
    _set_nonperiodic_graph_cell = staticmethod(set_nonperiodic_graph_cell)
    _structure_from_positions = staticmethod(structure_from_positions)
    _structure_from_atoms = staticmethod(structure_from_atoms)
    _progress = staticmethod(progress_snapshot)
    _max_force = staticmethod(max_force)

    @classmethod
    def _prepare_reference_images(
        cls,
        reactant: Atoms,
        product: Atoms,
    ) -> tuple[list[Atoms], list[str]]:
        candidates, warnings = cls._prepare_reference_candidates(
            reactant,
            product,
            ensemble_size=1,
        )
        return candidates[0], warnings

    @classmethod
    def _prepare_reference_candidates(
        cls,
        reactant: Atoms,
        product: Atoms,
        *,
        ensemble_size: int,
    ) -> tuple[list[list[Atoms]], list[str]]:
        ensemble = generate_directional_reference_paths(
            reactant,
            product,
            ensemble_size=ensemble_size,
        )
        return ensemble.candidates, list(ensemble.warnings)

    @classmethod
    def _initial_path_energy_score(
        cls,
        path,
        template: Atoms,
        calculator,
    ) -> float:
        return coarse_path_barrier_score(
            path.get_positions(t=np.linspace(0.0, 1.0, 7)),
            template,
            calculator,
        )

    @staticmethod
    def _build_path_bias(
        reactant: Atoms,
        product: Atoms,
        reaction_coordinates: list[ReactionCoordinate],
        **strengths,
    ) -> PathBias:
        return build_path_bias(
            reactant,
            product,
            reaction_coordinates,
            **strengths,
        )

    _evaluate_path_energies = staticmethod(evaluate_path_energies)
    _select_path_maximum = staticmethod(select_path_maximum)

    @staticmethod
    def _assess_bond_changes(
        reactant: Atoms,
        transition: Atoms,
        product: Atoms,
        expected_coordinates: list[ReactionCoordinate] | None = None,
    ) -> tuple[list[BondChangeAssessment], bool | None]:
        return schema_bond_changes(
            reactant,
            transition,
            product,
            expected_coordinates,
        )

    _reaction_active_region = staticmethod(reaction_active_region)
    _align_product_to_stable_scaffold = staticmethod(
        align_product_to_stable_scaffold
    )
    _linear_path_has_collision = staticmethod(linear_path_has_collision)
    _validate_path_geometry = staticmethod(validate_path_geometry)
    _minimum_covalent_ratio = staticmethod(minimum_covalent_ratio)

    @staticmethod
    def _validate_finite_path_state(path) -> None:
        for name, values in (
            ("coordinates", getattr(path, "coefs", None)),
            ("energies", getattr(path, "energies", None)),
            ("forces", getattr(path, "forces", None)),
        ):
            if values is None or not np.isfinite(values).all():
                raise RuntimeError(f"DMF produced non-finite {name}")

    @staticmethod
    def _raise_if_cancelled(cancel_event: Event) -> None:
        if cancel_event.is_set():
            raise CalculationCancelled("calculation cancelled")

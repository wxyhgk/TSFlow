from dataclasses import dataclass
from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.runtime import CalculationCancelled
from uma_workflows.transition_states import (
    FbenmConfig,
    FbenmPathInitializer,
    PathCandidate,
    PathBias,
    ReactionPath,
    TransitionStateProblem,
    analyze_mode_reaction_overlap,
    assess_bond_changes,
    build_path_bias,
    coordinated_path_score,
    dense_path_bias_value_and_gradient,
    generate_directional_reference_paths,
    select_best_candidate,
    select_path_maximum,
    same_kind_progress_imbalance,
)
from uma_workflows.transition_states.workflow.models import PathConfig
from uma_workflows.transition_states.workflow.path_bias import prepare_path_bias


@dataclass(frozen=True)
class Coordinate:
    atom_i: int
    atom_j: int
    kind: str


class MarkerEnergyCalculator(Calculator):
    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = float(atoms.positions[0, 1])


def test_path_bias_accepts_structural_reaction_coordinate() -> None:
    reactant = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    product = Atoms("CC", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

    bias = build_path_bias(
        reactant,
        product,
        [Coordinate(0, 1, "forming")],
        repulsion_strength_ev_per_a2=0.0,
        repulsion_cutoff_scale=0.8,
        synchronization_strength_ev=0.0,
        coordinate_range_strength_ev=100.0,
        coordinate_schedule_strength_ev=0.0,
        scaffold_strength_ev_per_a2=0.0,
    )

    assert bias.synchronized_coordinates == ((0, 1, 3.0, 1.5),)


def test_asynchronous_bias_does_not_exclude_ts2_like_progress() -> None:
    # UMA-refined TS2 is deliberately asynchronous: the two forming bonds are
    # nearly complete while the two breaking bonds have only just started.
    progresses = (0.108, 0.222, 0.959, 0.808)
    starts = (1.0, 1.0, 3.0, 3.0)
    finishes = (3.0, 3.0, 1.0, 1.0)
    reactant_positions = []
    product_positions = []
    ts_positions = []
    coordinates = []
    for index, (progress, start, finish) in enumerate(
        zip(progresses, starts, finishes, strict=True)
    ):
        origin = 10.0 * index
        reactant_positions.extend(
            ((origin, 0.0, 0.0), (origin + start, 0.0, 0.0))
        )
        product_positions.extend(
            ((origin, 0.0, 0.0), (origin + finish, 0.0, 0.0))
        )
        distance = start + progress * (finish - start)
        ts_positions.extend(((origin, 0.0, 0.0), (origin + distance, 0.0, 0.0)))
        coordinates.append(
            Coordinate(
                2 * index,
                2 * index + 1,
                "breaking" if index < 2 else "forming",
            )
        )

    reactant = Atoms("C8", positions=reactant_positions)
    product = Atoms("C8", positions=product_positions)
    common = dict(
        repulsion_strength_ev_per_a2=0.0,
        coordinate_range_strength_ev=100.0,
    )
    asynchronous, _ = prepare_path_bias(
        reactant,
        product,
        tuple(coordinates),
        PathConfig(bias_mode="asynchronous", **common),
    )
    legacy, _ = prepare_path_bias(
        reactant,
        product,
        tuple(coordinates),
        PathConfig(bias_mode="legacy_synchronous", **common),
    )

    asynchronous_energy, _ = asynchronous.energy_and_forces(
        np.asarray(ts_positions), target_progress=0.5
    )
    legacy_energy, _ = legacy.energy_and_forces(
        np.asarray(ts_positions), target_progress=0.5
    )

    assert asynchronous_energy < 0.10
    assert asynchronous.synchronization_strength_ev == 5.0
    assert asynchronous.synchronization_groups == ((2, 3), (0, 1))
    assert asynchronous.coordinate_schedule_strength_ev == 0.0
    assert asynchronous.scaffold_strength_ev_per_a2 == 0.0
    assert legacy_energy > 6.0


def test_native_bias_disables_every_custom_term() -> None:
    reactant = Atoms("CCC", positions=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]])
    product = Atoms("CCC", positions=[[0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    bias, _ = prepare_path_bias(
        reactant,
        product,
        (Coordinate(0, 1, "breaking"),),
        PathConfig(bias_mode="native"),
    )

    assert bias.repulsion_strength_ev_per_a2 == 0.0
    assert bias.synchronization_strength_ev == 0.0
    assert bias.coordinate_range_strength_ev == 0.0
    assert bias.coordinate_schedule_strength_ev == 0.0
    assert bias.scaffold_strength_ev_per_a2 == 0.0


def test_dense_path_bias_gradient_matches_finite_difference() -> None:
    bias = PathBias(
        symbols=("C", "C"),
        repulsion_cutoffs_a=((0, 1, 1.2),),
        synchronized_coordinates=(),
        scaffold_distances=(),
        repulsion_strength_ev_per_a2=75.0,
        repulsion_cutoff_scale=0.8,
        synchronization_strength_ev=0.0,
        coordinate_range_strength_ev=0.0,
        coordinate_schedule_strength_ev=0.0,
        scaffold_strength_ev_per_a2=0.0,
    )
    basis = np.asarray([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    controls = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.60, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.40, 0.0, 0.0]],
        ]
    )
    positions = np.tensordot(basis.T, controls, axes=1)
    value, gradient = dense_path_bias_value_and_gradient(bias, positions, basis)
    displaced = controls.copy()
    displaced[0, 1, 0] += 1.0e-6
    displaced_value, _ = dense_path_bias_value_and_gradient(
        bias,
        np.tensordot(basis.T, displaced, axes=1),
        basis,
    )

    assert gradient[0, 1, 0] == pytest.approx(
        (displaced_value - value) / 1.0e-6,
        rel=1.0e-4,
    )


def test_path_analysis_and_reaction_assessment_are_schema_independent() -> None:
    index, interior = select_path_maximum([0.0, 1.0, 0.5])
    reactant = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    transition = Atoms("CC", positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
    product = Atoms("CC", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    assessments, consistent = assess_bond_changes(
        reactant,
        transition,
        product,
        [Coordinate(0, 1, "forming")],
    )

    assert (index, interior) == (1, True)
    assert consistent is True
    assert assessments[0].kind == "forming"


def test_reaction_path_validates_image_and_coordinate_shapes() -> None:
    with pytest.raises(ValueError):
        ReactionPath(
            positions_a=np.zeros((2, 3)),
            coordinates=np.zeros(2),
        )


def test_directional_reference_paths_preserve_endpoints() -> None:
    reactant = Atoms(
        "HCO",
        positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.0, 0.0]],
    )
    product = reactant.copy()
    product.positions[2] = [1.1, 1.1, 0.0]

    ensemble = generate_directional_reference_paths(
        reactant,
        product,
        ensemble_size=3,
    )

    assert len(ensemble.candidates) == 3
    product_distances = product.get_all_distances()
    aligned_product_positions = ensemble.candidates[0][-1].positions
    for images in ensemble.candidates:
        assert np.allclose(images[0].positions, reactant.positions)
        assert np.allclose(images[-1].positions, aligned_product_positions)
        assert np.allclose(images[-1].get_all_distances(), product_distances)


def test_candidate_ranking_is_stable_for_equal_scores() -> None:
    path = ReactionPath(
        positions_a=np.zeros((2, 1, 3)),
        coordinates=np.asarray([0.0, 1.0]),
    )
    candidates = (
        PathCandidate(path=path, score=2.0, source="second"),
        PathCandidate(path=path, score=1.0, source="first-best"),
        PathCandidate(path=path, score=1.0, source="second-best"),
    )

    index, selected = select_best_candidate(candidates)

    assert index == 1
    assert selected.source == "first-best"


def test_candidate_ranking_requires_scores() -> None:
    path = ReactionPath(
        positions_a=np.zeros((2, 1, 3)),
        coordinates=np.asarray([0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="must be scored"):
        select_best_candidate((PathCandidate(path=path),))


def test_same_kind_progress_imbalance_allows_forming_breaking_asynchrony() -> None:
    reactant = Atoms(
        "C8",
        positions=[
            [0, 0, 0], [1, 0, 0],
            [10, 0, 0], [11, 0, 0],
            [20, 0, 0], [23, 0, 0],
            [30, 0, 0], [33, 0, 0],
        ],
    )
    product = Atoms(
        "C8",
        positions=[
            [0, 0, 0], [3, 0, 0],
            [10, 0, 0], [13, 0, 0],
            [20, 0, 0], [21, 0, 0],
            [30, 0, 0], [31, 0, 0],
        ],
    )
    transition = reactant.positions.copy()
    progresses = (0.108, 0.222, 0.959, 0.808)
    for index, progress in enumerate(progresses):
        start = reactant.get_distance(2 * index, 2 * index + 1)
        finish = product.get_distance(2 * index, 2 * index + 1)
        transition[2 * index + 1, 0] = 10 * index + start + progress * (finish - start)
    coordinates = (
        Coordinate(0, 1, "breaking"),
        Coordinate(2, 3, "breaking"),
        Coordinate(4, 5, "forming"),
        Coordinate(6, 7, "forming"),
    )

    imbalance = same_kind_progress_imbalance(
        transition, reactant, product, coordinates
    )

    assert imbalance == pytest.approx(0.151)


def test_coordinated_score_penalizes_partial_two_bond_path() -> None:
    reactant = Atoms(
        "C4",
        positions=[[0, 0, 0], [3, 0, 0], [10, 0, 0], [13, 0, 0]],
    )
    product = Atoms(
        "C4",
        positions=[[0, 0, 0], [1.5, 0, 0], [10, 0, 0], [11.5, 0, 0]],
    )
    balanced = 0.5 * (reactant.positions + product.positions)
    partial = reactant.positions.copy()
    partial[1] = product.positions[1]
    balanced[:, 1] = 1.0
    partial[:, 1] = 1.0
    paths = {
        "balanced": np.stack((reactant.positions, balanced, product.positions)),
        "partial": np.stack((reactant.positions, partial, product.positions)),
    }
    coordinates = (
        Coordinate(0, 1, "forming"),
        Coordinate(2, 3, "forming"),
    )

    scores = {
        name: coordinated_path_score(
            images,
            reactant,
            MarkerEnergyCalculator(),
            reactant=reactant,
            product=product,
            reaction_coordinates=coordinates,
        )
        for name, images in paths.items()
    }

    assert scores["balanced"].same_kind_progress_imbalance == pytest.approx(0.0)
    assert scores["balanced"].coordination_penalty_ev == pytest.approx(0.0)
    assert scores["partial"].same_kind_progress_imbalance == pytest.approx(1.0)
    assert scores["partial"].coordination_penalty_ev == pytest.approx(1.5)


def test_fbenm_preserves_cancellation_from_progress(monkeypatch) -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    token = Event()
    interpolation_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal interpolation_called
        interpolation_called = True
        raise AssertionError("interpolation must not start after cancellation")

    monkeypatch.setattr(
        "uma_workflows.transition_states.fbenm.interpolate_fbenm",
        fail_if_called,
    )

    with pytest.raises(CalculationCancelled):
        FbenmPathInitializer(FbenmConfig(nmove=3)).generate(
            TransitionStateProblem(reactant=reactant, product=product),
            calculator=None,
            cancel_token=token,
            progress=lambda *_args: token.set(),
        )

    assert interpolation_called is False


def test_mode_overlap_handles_arbitrary_eigenvector_sign() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = analyze_mode_reaction_overlap(
        atoms,
        mode,
        [Coordinate(0, 1, "forming")],
    )

    assert result.matches is True
    assert result.score == pytest.approx(1.0)
    assert result.projections[0].bond_length_derivative < 0.0

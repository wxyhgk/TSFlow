from pathlib import Path
from threading import Event

from ase import Atoms
from ase.calculators.emt import EMT
import numpy as np

from app.schemas.jobs import JobCreate, ReactionCoordinate
from app.services.dmf_executor import (
    DmfExecutor,
    _PathBias,
    _dense_path_bias_value_and_gradient,
)


class CheapUma:
    def resolve_optimization_device(self):
        return "cpu"

    def calculator(self, _checkpoint, _device, _task_name):
        return EMT()


def test_path_bias_repels_nonbonded_collision_with_consistent_force() -> None:
    bias = _PathBias(
        symbols=("C", "C"),
        repulsion_cutoffs_a=((0, 1, 1.20),),
        synchronized_coordinates=(),
        repulsion_strength_ev_per_a2=75.0,
        repulsion_cutoff_scale=0.80,
        synchronization_strength_ev=0.0,
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [0.50, 0.0, 0.0]])

    energy, forces = bias.energy_and_forces(positions)
    displaced = positions.copy()
    displaced[0, 0] += 1.0e-6
    displaced_energy, _ = bias.energy_and_forces(displaced)
    numerical_force = -(displaced_energy - energy) / 1.0e-6

    assert energy > 0.0
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0
    assert np.isclose(forces[0, 0], numerical_force, rtol=1.0e-4)


def test_path_bias_synchronizes_two_forming_bond_progresses() -> None:
    bias = _PathBias(
        symbols=("C", "C", "C", "C"),
        repulsion_cutoffs_a=(),
        synchronized_coordinates=(
            (0, 2, 3.0, 1.5),
            (1, 3, 3.0, 1.5),
        ),
        repulsion_strength_ev_per_a2=0.0,
        repulsion_cutoff_scale=0.80,
        synchronization_strength_ev=5.0,
    )
    synchronous = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.25, 0.0, 0.0], [2.25, 2.0, 0.0]]
    )
    asynchronous = synchronous.copy()
    asynchronous[2, 0] = 1.50

    synchronous_energy, synchronous_forces = bias.energy_and_forces(synchronous)
    asynchronous_energy, asynchronous_forces = bias.energy_and_forces(asynchronous)

    assert np.isclose(synchronous_energy, 0.0)
    assert np.allclose(synchronous_forces, 0.0)
    assert asynchronous_energy > 0.0
    assert asynchronous_forces[0, 0] < 0.0
    assert asynchronous_forces[2, 0] > 0.0


def test_path_bias_prevents_forming_bond_overshoot() -> None:
    bias = _PathBias(
        symbols=("C", "C"),
        repulsion_cutoffs_a=(),
        synchronized_coordinates=((0, 1, 3.0, 1.5),),
        repulsion_strength_ev_per_a2=0.0,
        repulsion_cutoff_scale=0.80,
        synchronization_strength_ev=0.0,
        coordinate_range_strength_ev=100.0,
    )
    inside = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    overshot = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])

    inside_energy, _ = bias.energy_and_forces(inside)
    overshot_energy, overshot_forces = bias.energy_and_forces(overshot)

    assert np.isclose(inside_energy, 0.0)
    assert overshot_energy > 0.0
    assert overshot_forces[0, 0] < 0.0
    assert overshot_forces[1, 0] > 0.0


def test_path_bias_tracks_requested_reaction_progress() -> None:
    bias = _PathBias(
        symbols=("C", "C"),
        repulsion_cutoffs_a=(),
        synchronized_coordinates=((0, 1, 3.0, 1.5),),
        repulsion_strength_ev_per_a2=0.0,
        repulsion_cutoff_scale=0.80,
        synchronization_strength_ev=0.0,
        coordinate_range_strength_ev=0.0,
        coordinate_schedule_strength_ev=20.0,
    )
    on_schedule = np.asarray([[0.0, 0.0, 0.0], [2.25, 0.0, 0.0]])
    late = np.asarray([[0.0, 0.0, 0.0], [2.85, 0.0, 0.0]])

    scheduled_energy, _ = bias.energy_and_forces(
        on_schedule,
        target_progress=0.5,
    )
    late_energy, late_forces = bias.energy_and_forces(
        late,
        target_progress=0.5,
    )

    assert np.isclose(scheduled_energy, 0.0)
    assert late_energy > 0.0
    assert late_forces[0, 0] > 0.0
    assert late_forces[1, 0] < 0.0


def test_path_bias_preserves_interpolated_scaffold_distance() -> None:
    bias = _PathBias(
        symbols=("C", "C"),
        repulsion_cutoffs_a=(),
        synchronized_coordinates=(),
        scaffold_distances=((0, 1, 1.4, 1.6),),
        repulsion_strength_ev_per_a2=0.0,
        repulsion_cutoff_scale=0.80,
        synchronization_strength_ev=0.0,
        coordinate_range_strength_ev=0.0,
        coordinate_schedule_strength_ev=0.0,
        scaffold_strength_ev_per_a2=5.0,
    )
    expected = np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    stretched = np.asarray([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])

    expected_energy, _ = bias.energy_and_forces(
        expected,
        target_progress=0.5,
    )
    stretched_energy, stretched_forces = bias.energy_and_forces(
        stretched,
        target_progress=0.5,
    )

    assert np.isclose(expected_energy, 0.0)
    assert stretched_energy > 0.0
    assert stretched_forces[0, 0] > 0.0
    assert stretched_forces[1, 0] < 0.0


def test_dense_path_bias_gradient_matches_control_point_finite_difference() -> None:
    bias = _PathBias(
        symbols=("C", "C"),
        repulsion_cutoffs_a=((0, 1, 1.20),),
        synchronized_coordinates=(),
        repulsion_strength_ev_per_a2=75.0,
        repulsion_cutoff_scale=0.80,
        synchronization_strength_ev=0.0,
    )
    basis = np.asarray(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        ]
    )
    control_points = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.60, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.40, 0.0, 0.0]],
        ]
    )
    positions = np.tensordot(basis.T, control_points, axes=1)
    value, gradient = _dense_path_bias_value_and_gradient(
        bias,
        positions,
        basis,
    )
    displaced = control_points.copy()
    displaced[0, 1, 0] += 1.0e-6
    displaced_positions = np.tensordot(basis.T, displaced, axes=1)
    displaced_value, _ = _dense_path_bias_value_and_gradient(
        bias,
        displaced_positions,
        basis,
    )
    numerical_gradient = (displaced_value - value) / 1.0e-6

    assert np.isclose(gradient[0, 1, 0], numerical_gradient, rtol=1.0e-4)


def test_dmf_executor_runs_fbenm_path_and_generates_gaussian_input() -> None:
    request = JobCreate.model_validate(
        {
            "calculation": "transition_state_search",
            "structure": {
                "atoms": [
                    {"symbol": "Cu", "position": [0.0, 0.0, 0.0]},
                    {"symbol": "Cu", "position": [2.5, 0.0, 0.0]},
                    {"symbol": "Cu", "position": [1.25, 2.0, 0.0]},
                ],
                "charge": 0,
                "spin": 2,
            },
            "product_structure": {
                "atoms": [
                    {"symbol": "Cu", "position": [0.0, 0.0, 0.0]},
                    {"symbol": "Cu", "position": [2.5, 0.0, 0.0]},
                    {"symbol": "Cu", "position": [5.0, 0.0, 0.0]},
                ],
                "charge": 0,
                "spin": 2,
            },
            "parameters": {
                "nmove": 3,
                "dmf_max_iterations": 5,
                "dmf_convergence": "loose",
                "path_points": 5,
                "sella_refine": False,
                "compute_frequencies": False,
            },
        }
    )
    progress = []
    result = DmfExecutor(CheapUma()).execute(  # type: ignore[arg-type]
        request, Path("/tmp/not-used.pt"), Event(), progress.append
    )

    assert len(result.path) == 5
    assert len(result.energy_profile) == 5
    assert 0.0 <= result.highest_path_coordinate <= 1.0
    assert result.dmf_iterations > 0
    assert result.scientific_outcome in {
        "validated_transition_state",
        "screened_first_order_saddle",
        "near_barrierless",
        "unvalidated_candidate",
        "path_search_inconclusive",
    }
    assert result.forward_path_rise_ev is not None
    assert result.reverse_path_rise_ev is not None
    assert result.forward_barrier_ev >= 0.0
    assert result.reverse_barrier_ev >= 0.0
    assert result.barrier_source in {
        "validated_transition_state",
        "screened_first_order_saddle",
        "converged_path_estimate",
        "unconverged_path_estimate",
    }
    assert result.barrier_is_validated is False
    assert {item.stage for item in progress} == {
        "endpoint_preoptimization",
        "fbenm",
        "dmf",
        "cpu_validation",
    }
    assert "Opt=(TS,CalcFC) Freq" in result.gaussian_input


def test_reference_path_aligns_scaffold_and_bends_hcn_hnc_migration() -> None:
    request = JobCreate.model_validate(
        {
            "calculation": "transition_state_search",
            "structure": {
                "atoms": [
                    {"symbol": "H", "position": [-1.06, 0.0, 0.0]},
                    {"symbol": "C", "position": [0.0, 0.0, 0.0]},
                    {"symbol": "N", "position": [1.16, 0.0, 0.0]},
                ],
                "charge": 0,
                "spin": 1,
            },
            "product_structure": {
                "atoms": [
                    {"symbol": "H", "position": [2.16, 0.0, 0.0]},
                    {"symbol": "C", "position": [0.0, 0.0, 0.0]},
                    {"symbol": "N", "position": [1.16, 0.0, 0.0]},
                ],
                "charge": 0,
                "spin": 1,
            },
        }
    )
    reactant = DmfExecutor._to_atoms(request.structure)
    product = DmfExecutor._to_atoms(request.product_structure)  # type: ignore[arg-type]
    references, warnings = DmfExecutor._prepare_reference_images(reactant, product)

    assert warnings
    assert np.allclose(references[0].positions[1:], references[-1].positions[1:])
    assert np.linalg.norm(references[1].positions[0, 1:]) >= 0.75
    ratio, _, _ = DmfExecutor._minimum_covalent_ratio(
        np.asarray([image.positions for image in references]), ["H", "C", "N"]
    )
    assert ratio > 0.70


def test_bond_change_screen_rejects_endpoint_like_fragment_mode() -> None:
    reactant = DmfExecutor._to_atoms(
        JobCreate.model_validate(
            {
                "calculation": "transition_state_search",
                "structure": {
                    "atoms": [
                        {"symbol": "H", "position": [0.15, 0.0, 0.0]},
                        {"symbol": "F", "position": [0.0, 3.0, 0.0]},
                        {"symbol": "H", "position": [-0.15, -0.68, 0.0]},
                    ],
                    "charge": 0,
                    "spin": 2,
                },
                "product_structure": {
                    "atoms": [
                        {"symbol": "H", "position": [0.15, 0.0, 0.0]},
                        {"symbol": "F", "position": [0.0, 0.92, 0.0]},
                        {"symbol": "H", "position": [-0.15, -3.0, 0.0]},
                    ],
                    "charge": 0,
                    "spin": 2,
                },
            }
        ).structure
    )
    product = reactant.copy()
    product.set_positions([[0.15, 0.0, 0.0], [0.0, 0.92, 0.0], [-0.15, -3.0, 0.0]])
    endpoint_like = reactant.copy()
    endpoint_like.set_positions([[0.21, 0.31, 0.0], [-0.05, 2.92, 0.0], [-0.28, -0.24, 0.0]])

    changes, consistent = DmfExecutor._assess_bond_changes(
        reactant, endpoint_like, product
    )

    assert len(changes) == 2
    assert consistent is False
    assert any(not 0.10 <= item.path_fraction <= 0.90 for item in changes)


def test_bond_change_screen_accepts_early_but_chemically_advanced_h_transfer() -> None:
    reactant = Atoms(
        "CHH",
        positions=[[0.0, 0.0, 0.0], [0.0, -2.26, 0.0], [0.0, -3.01, 0.0]],
    )
    transition = Atoms(
        "CHH",
        positions=[[0.0, 0.0, 0.0], [0.0, -1.44, 0.0], [0.0, -2.30, 0.0]],
    )
    product = Atoms(
        "CHH",
        positions=[[0.0, 0.0, 0.0], [0.0, -1.09, 0.0], [0.0, -3.27, 0.0]],
    )

    changes, consistent = DmfExecutor._assess_bond_changes(
        reactant, transition, product
    )

    assert len(changes) == 2
    assert consistent is True


def test_clicked_coordinates_override_unrelated_inferred_distance_changes() -> None:
    reactant = Atoms(
        "HClLi",
        positions=[[0.0, 0.0, 0.0], [1.30, 0.0, 0.0], [3.2, 0.0, 0.0]],
    )
    transition = Atoms(
        "HClLi",
        positions=[[0.0, 0.0, 0.0], [1.75, 0.0, 0.0], [2.25, 0.0, 0.0]],
    )
    product = Atoms(
        "HClLi",
        positions=[[0.0, 0.0, 0.0], [3.2, 0.0, 0.0], [1.60, 0.0, 0.0]],
    )
    expected = [
        ReactionCoordinate(atom_i=0, atom_j=1, kind="breaking"),
        ReactionCoordinate(atom_i=0, atom_j=2, kind="forming"),
    ]

    changes, consistent = DmfExecutor._assess_bond_changes(
        reactant,
        transition,
        product,
        expected_coordinates=expected,
    )

    assert [(item.atom_i, item.atom_j, item.kind) for item in changes] == [
        (0, 1, "breaking"),
        (0, 2, "forming"),
    ]
    assert consistent is True


def test_path_maximum_rejects_endpoint_as_transition_state_guess() -> None:
    index, is_interior = DmfExecutor._select_path_maximum(
        [0.0, -0.02, -0.15, -0.30]
    )

    assert index == 0
    assert is_interior is False


def test_path_maximum_requires_resolvable_interior_prominence() -> None:
    index, is_interior = DmfExecutor._select_path_maximum(
        [0.0, 0.0004, -0.10]
    )

    assert index == 1
    assert is_interior is False


def test_path_maximum_accepts_interior_barrier() -> None:
    index, is_interior = DmfExecutor._select_path_maximum(
        [0.0, 0.15, -0.10]
    )

    assert index == 1
    assert is_interior is True


def test_path_maximum_accepts_submerged_interior_barrier() -> None:
    index, is_interior = DmfExecutor._select_path_maximum(
        [0.0, -0.008, -0.004, -0.050, -0.40]
    )

    assert index == 2
    assert is_interior is True

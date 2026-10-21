from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.frequencies import (
    analyze_vibrational_hessian,
    compute_numerical_hessian,
    numerical_frequency_analysis,
)
from uma_workflows.optimization import (
    OptimizationStage,
    refine_first_order_saddle,
    refine_saddle_stages,
)
from uma_workflows.transition_states import (
    PathBias,
    SaddlePipelineConfig,
    TransitionStateProblem,
    TransitionStateScreeningConfig,
    refine_path_saddle,
    screen_transition_state,
)
from uma_workflows.runtime import CalculationCancelled


class AnalyticSaddleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        coordinates = np.asarray(self.atoms.positions).reshape(-1)
        curvatures = np.asarray([-1.0, 2.0, 3.0])
        self.results["energy"] = float(0.5 * np.sum(curvatures * coordinates**2))
        self.results["forces"] = (-curvatures * coordinates).reshape((-1, 3))


class BondSaddleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = self.atoms.positions[1] - self.atoms.positions[0]
        distance = np.linalg.norm(displacement)
        coordinate = distance - 1.0
        force_on_second = coordinate * displacement / distance
        self.results["energy"] = float(-0.5 * coordinate**2)
        self.results["forces"] = np.asarray([-force_on_second, force_on_second])


class LinearPath:
    def __init__(self, start: Atoms, finish: Atoms) -> None:
        self.start = start.positions.copy()
        self.finish = finish.positions.copy()

    def evaluate(self, coordinates, *, derivative_order: int = 0):
        values = np.asarray(coordinates, dtype=float)
        displacement = self.finish - self.start
        if derivative_order == 1:
            return np.repeat(displacement[None, :, :], len(values), axis=0)
        return self.start + values[:, None, None] * displacement


class Coordinate:
    atom_i = 0
    atom_j = 1
    kind = "forming"


def zero_path_bias() -> PathBias:
    return PathBias(
        symbols=("H", "H"),
        repulsion_cutoffs_a=(),
        synchronized_coordinates=(),
        scaffold_distances=(),
        repulsion_strength_ev_per_a2=0.0,
        repulsion_cutoff_scale=0.8,
        synchronization_strength_ev=0.0,
        coordinate_range_strength_ev=0.0,
        coordinate_schedule_strength_ev=0.0,
        scaffold_strength_ev_per_a2=0.0,
    )


def test_numerical_frequency_analysis_finds_one_imaginary_mode() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = AnalyticSaddleCalculator()
    progress: list[tuple[int, int]] = []

    result = numerical_frequency_analysis(
        atoms,
        calculator=calculator,
        delta_a=0.01,
        imaginary_threshold_cm1=50.0,
        cancel_event=Event(),
        on_coordinate=lambda step, total: progress.append((step, total)),
        project_rigid_body=False,
    )

    assert len(result.imaginary_frequencies_cm1) == 1
    assert result.primary_imaginary_mode is not None
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_hessian_generation_and_frequency_analysis_are_independent() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = AnalyticSaddleCalculator()

    numerical_hessian = compute_numerical_hessian(
        atoms,
        calculator=calculator,
        delta_a=0.01,
        cancel_event=Event(),
    )
    result = analyze_vibrational_hessian(
        atoms,
        numerical_hessian.hessian_ev_per_a2,
        imaginary_threshold_cm1=50.0,
        project_rigid_body=False,
    )

    assert numerical_hessian.hessian_ev_per_a2 == pytest.approx(
        np.diag([-1.0, 2.0, 3.0])
    )
    assert len(result.imaginary_frequencies_cm1) == 1


def test_sella_refines_first_order_bond_saddle() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    calculator = BondSaddleCalculator()

    result = refine_first_order_saddle(
        atoms,
        path_tangent=np.asarray([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        optimization_device="cpu",
        mps_calculator=calculator,
        cpu_calculator=calculator,
        mps_fmax=0.1,
        cpu_fmax=1.0e-4,
        max_steps=20,
        cancel_event=Event(),
    )

    assert result.converged is True
    assert atoms.get_distance(0, 1) == pytest.approx(1.0, abs=1.0e-5)


def test_generic_saddle_stages_preserve_custom_stage_identity() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    calculator = BondSaddleCalculator()

    result = refine_saddle_stages(
        atoms,
        path_tangent=np.asarray([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        stages=[
            OptimizationStage(
                name="custom_cpu_refinement",
                device="cpu",
                calculator=calculator,
                fmax=1.0e-4,
                max_steps=20,
            )
        ],
        cancel_event=Event(),
    )

    assert result.converged is True
    assert result.stage_results[0].name == "custom_cpu_refinement"
    assert all(frame.stage == "custom_cpu_refinement" for frame in result.frames)


def test_saddle_pipeline_accepts_a_generic_parametric_path() -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    candidate = reactant.copy()
    candidate.set_distance(0, 1, 1.2)
    calculator = BondSaddleCalculator()

    result = refine_path_saddle(
        TransitionStateProblem(reactant=reactant, product=product),
        candidate,
        path=LinearPath(reactant, product),
        highest_coordinate=0.5,
        path_bias=zero_path_bias(),
        mps_calculator=calculator,
        cpu_calculator=calculator,
        config=SaddlePipelineConfig(
            optimization_device="cpu",
            mps_fmax=0.1,
            cpu_fmax=1.0e-4,
            max_steps=20,
        ),
        cancel_token=Event(),
    )

    assert result.refinement.converged is True
    assert result.atoms.get_distance(0, 1) == pytest.approx(1.0, abs=1.0e-5)


def test_corridor_seed_is_followed_by_unbiased_cpu_sella() -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    candidate = reactant.copy()
    calculator = BondSaddleCalculator()
    bias = zero_path_bias()
    bias = PathBias(
        **{
            **bias.__dict__,
            "synchronized_coordinates": ((0, 1, 1.4, 0.6),),
            "coordinate_range_strength_ev": 20.0,
        }
    )

    result = refine_path_saddle(
        TransitionStateProblem(reactant=reactant, product=product),
        candidate,
        path=LinearPath(reactant, product),
        highest_coordinate=0.5,
        path_bias=bias,
        mps_calculator=calculator,
        cpu_calculator=calculator,
        config=SaddlePipelineConfig(
            optimization_device="cpu",
            mps_fmax=0.1,
            cpu_fmax=1.0e-4,
            max_steps=10,
            cpu_refine=False,
        ),
        cancel_token=Event(),
    )

    assert len(result.refinement.stage_results) == 2
    assert all(
        stage.device == "cpu" for stage in result.refinement.stage_results
    )
    assert result.atoms.calc is None
    assert any("removed the bias" in warning for warning in result.warnings)


def test_transition_state_screening_is_independent_of_api_schemas() -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    candidate = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = screen_transition_state(
        TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=(Coordinate(),),
        ),
        candidate,
        calculator=BondSaddleCalculator(),
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=1.0e-4,
            compute_frequencies=False,
            frequency_active_region_hops=1,
            frequency_max_atoms=10,
            frequency_delta_a=0.01,
            imaginary_frequency_threshold_cm1=50.0,
        ),
        cancel_token=Event(),
    )

    assert result.stationary is True
    assert result.reaction_coordinate_consistent is True
    assert result.frequency_analysis is None
    assert result.passes_screening is False


def test_transition_state_screening_honors_entry_cancellation() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    token = Event()
    token.set()

    with pytest.raises(CalculationCancelled):
        screen_transition_state(
            TransitionStateProblem(reactant=atoms, product=atoms.copy()),
            atoms,
            calculator=BondSaddleCalculator(),
            config=TransitionStateScreeningConfig(
                stationary_force_threshold_ev_per_a=0.1,
                compute_frequencies=False,
                frequency_active_region_hops=1,
                frequency_max_atoms=10,
                frequency_delta_a=0.01,
                imaginary_frequency_threshold_cm1=50.0,
            ),
            cancel_token=token,
        )


def test_full_hessian_and_matching_mode_pass_ts_screening() -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    candidate = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = screen_transition_state(
        TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=(Coordinate(),),
        ),
        candidate,
        calculator=BondSaddleCalculator(),
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=1.0e-4,
            compute_frequencies=True,
            frequency_active_region_hops=1,
            frequency_max_atoms=10,
            frequency_delta_a=0.005,
            imaginary_frequency_threshold_cm1=50.0,
        ),
        cancel_token=Event(),
    )

    assert result.frequency_scope == "full"
    assert result.mode_reaction_overlap is not None
    assert result.mode_reaction_overlap.matches is True
    assert result.passes_screening is True


def test_early_transition_state_remains_eligible_for_irc() -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.02, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.60, 0.0, 0.0]])
    candidate = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.00, 0.0, 0.0]])

    result = screen_transition_state(
        TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=(Coordinate(),),
        ),
        candidate,
        calculator=BondSaddleCalculator(),
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=1.0e-4,
            compute_frequencies=True,
            frequency_active_region_hops=1,
            frequency_max_atoms=10,
            frequency_delta_a=0.005,
            imaginary_frequency_threshold_cm1=50.0,
        ),
        cancel_token=Event(),
    )

    assert result.local_saddle_valid is True
    assert result.reaction_definition_valid is True
    assert result.geometry_progress_consistent is False
    assert result.irc_eligible is True
    assert result.passes_screening is True
    assert result.failure_code is None
    assert result.retry_actions == ()
    assert "GEOMETRY_PROGRESS_ADVISORY" in result.diagnostic_codes


def test_inconsistent_reaction_definition_is_advisory_for_irc() -> None:
    class BreakingCoordinate:
        atom_i = 0
        atom_j = 1
        kind = "breaking"

    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    candidate = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = screen_transition_state(
        TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=(BreakingCoordinate(),),
        ),
        candidate,
        calculator=BondSaddleCalculator(),
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=1.0e-4,
            compute_frequencies=True,
            frequency_active_region_hops=1,
            frequency_max_atoms=10,
            frequency_delta_a=0.005,
            imaginary_frequency_threshold_cm1=50.0,
        ),
        cancel_token=Event(),
    )

    assert result.local_saddle_valid is True
    assert result.pes_index1_saddle_valid is True
    assert result.reaction_definition_valid is False
    assert result.irc_eligible is True
    assert result.passes_screening is True
    assert result.failure_code is None
    assert "REACTION_DEFINITION_MISMATCH" in result.diagnostic_codes


def test_mode_hypothesis_mismatch_does_not_block_index1_irc(
    monkeypatch,
) -> None:
    reactant = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    product = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    candidate = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    analysis = SimpleNamespace(
        imaginary_frequencies_cm1=[-250.0],
        hessian_asymmetry_relative=0.0,
    )
    monkeypatch.setattr(
        "uma_workflows.transition_states.screening.screen_frequencies",
        lambda *args, **kwargs: SimpleNamespace(
            analysis=analysis,
            scope="full",
            overlap=SimpleNamespace(matches=False, score=0.01),
            active_indices=(0, 1),
            warnings=(),
        ),
    )

    result = screen_transition_state(
        TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=(Coordinate(),),
        ),
        candidate,
        calculator=BondSaddleCalculator(),
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=1.0e-4,
            compute_frequencies=True,
            frequency_active_region_hops=1,
            frequency_max_atoms=10,
            frequency_delta_a=0.005,
            imaginary_frequency_threshold_cm1=50.0,
        ),
        cancel_token=Event(),
    )

    assert result.pes_index1_saddle_valid is True
    assert result.mechanism_mode_consistent is False
    assert result.irc_eligible is True
    assert result.failure_code is None
    assert "MODE_REACTION_MISMATCH" in result.diagnostic_codes

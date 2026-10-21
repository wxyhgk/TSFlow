from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.irc import BidirectionalIrcConfig, IrcFrame, run_bidirectional_irc
from uma_workflows.irc.workflow import _relax_endpoint
from uma_workflows.transition_states import IrcConfig, TransitionSearchConfig
from uma_workflows.transition_states.workflow.models import PathConfig, SaddleConfig
from uma_workflows.transition_states.workflow.path import (
    build_supplied_guess_reference_path,
)
from uma_workflows.transition_states.workflow.saddle import refine_saddle_guess
from uma_workflows.transition_states.workflow.irc import follow_and_validate_irc
from uma_workflows.transition_states.workflow import irc as irc_workflow
from uma_workflows.transition_states.workflow import runner as transition_runner


class HarmonicEndpointCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        self.results["energy"] = float(0.5 * np.sum(positions**2))
        self.results["forces"] = -positions


def test_transition_search_workflow_has_standalone_defaults() -> None:
    config = TransitionSearchConfig()

    assert config.endpoint.enabled is True
    assert config.path.initializer_id == "fbenm.correlated"
    assert config.path.optimizer_id == "dmf.direct_maxflux"
    assert config.path.nmove == 10
    assert config.path.bias_mode == "asynchronous"
    assert config.saddle.cpu_fmax == 0.05
    assert config.screening.compute_frequencies is True
    assert config.irc.enabled is False


def test_supplied_guess_reference_path_does_not_run_dmf() -> None:
    reactant = Atoms("H", positions=[[-1.0, 0.0, 0.0]])
    guess = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    product = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    calculator = HarmonicEndpointCalculator([[0.0, 0.0, 0.0]])

    result = build_supplied_guess_reference_path(
        reactant,
        product,
        guess,
        reaction_coordinates=(),
        config=PathConfig(),
        cpu_calculator=calculator,
        cancel_event=Event(),
    )

    assert result.optimization.iterations == 0
    assert result.optimization.optimizer_status == "supplied_transition_state_guess"
    assert result.candidates_evaluated == 0
    assert result.sampled_positions.shape == (3, 1, 3)
    assert any("no FB-ENM or DMF" in warning for warning in result.warnings)


def test_supplied_guess_refinement_falls_back_to_finite_difference(
    monkeypatch,
) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    cpu_calculator = HarmonicEndpointCalculator([[0.0, 0.0, 0.0]])
    captured = {}

    class BrokenAnalyticCalculator:
        def get_property(self, *_args, **_kwargs):
            raise RuntimeError("analytic unavailable")

    def fake_refinement(candidate, hessian, **_kwargs):
        captured["hessian"] = hessian
        return SimpleNamespace(converged=True)

    monkeypatch.setattr(
        "uma_workflows.transition_states.workflow.saddle.refine_geometric_saddle",
        fake_refinement,
    )
    result = refine_saddle_guess(
        atoms,
        config=SaddleConfig(),
        cpu_calculator=cpu_calculator,
        analytic_hessian_calculator=BrokenAnalyticCalculator(),
        cancel_event=Event(),
        progress_callback=None,
        hessian_method="auto",
        finite_difference_max_atoms=2,
    )

    assert captured["hessian"].shape == (3, 3)
    assert any("fell back to finite differences" in warning for warning in result.warnings)


def test_supplied_guess_workflow_bypasses_endpoint_and_dmf_work(
    monkeypatch,
) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    endpoint = SimpleNamespace(warnings=())
    path = SimpleNamespace(
        warnings=("reference path",),
        cpu_energies_ev=[0.0, 1.0, 0.0],
        optimization=SimpleNamespace(converged=False),
    )
    saddle = SimpleNamespace(warnings=(), refinement=object(), atoms=atoms)
    screening = SimpleNamespace(
        warnings=(),
        frequency_analysis=SimpleNamespace(imaginary_frequencies_cm1=[-100.0]),
        frequency_scope="full",
        stationary=True,
        passes_screening=True,
    )
    irc = SimpleNamespace(warnings=(), validation_passed=None)
    outcome = SimpleNamespace(kind="screened_first_order_saddle")

    def fake_prepare(_reactant, _product, *, config, **_kwargs):
        assert config.enabled is False
        return endpoint

    monkeypatch.setattr(transition_runner, "prepare_endpoints", fake_prepare)
    monkeypatch.setattr(
        transition_runner,
        "optimize_reaction_path",
        lambda *args, **kwargs: pytest.fail("DMF path search must be bypassed"),
    )
    monkeypatch.setattr(
        transition_runner,
        "build_supplied_guess_reference_path",
        lambda *args, **kwargs: path,
    )
    monkeypatch.setattr(
        transition_runner,
        "refine_saddle_guess",
        lambda *args, **kwargs: saddle,
    )
    monkeypatch.setattr(transition_runner, "screen_saddle", lambda *a, **k: screening)
    monkeypatch.setattr(
        transition_runner,
        "follow_and_validate_irc",
        lambda *args, **kwargs: irc,
    )
    monkeypatch.setattr(
        transition_runner,
        "classify_transition_outcome",
        lambda *args, **kwargs: outcome,
    )

    result = transition_runner.run_transition_state_search(
        atoms.copy(),
        atoms.copy(),
        transition_state_guess=atoms.copy(),
        reaction_coordinates=(),
        config=TransitionSearchConfig(),
        device="cpu",
        calculator=None,
        cpu_calculator=None,
        analytic_hessian_calculator=None,
        cancel_event=Event(),
    )

    assert result.endpoints is endpoint
    assert result.path is path
    assert result.saddle is saddle


def test_requested_irc_is_skipped_without_definitive_ts_screen() -> None:
    class Screening:
        passes_screening = False
        frequency_scope = "partial"
        frequency_analysis = None

    config = TransitionSearchConfig()
    irc_config = type(config.irc)(enabled=True)
    result = follow_and_validate_irc(
        path=None,  # type: ignore[arg-type]
        saddle=None,  # type: ignore[arg-type]
        screening=Screening(),  # type: ignore[arg-type]
        config=irc_config,
        cpu_calculator=None,
        cancel_event=None,  # type: ignore[arg-type]
        progress_callback=None,
    )

    assert result.performed is False
    assert result.validation_passed is False
    assert "full-Hessian" in (result.error or "")


def test_early_ts_geometry_advisory_does_not_skip_irc(monkeypatch) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    relaxation = SimpleNamespace(converged=True)
    expected_result = SimpleNamespace(
        warnings=(),
        endpoint_a_relaxation=relaxation,
        endpoint_b_relaxation=relaxation,
        path_complete=True,
        endpoint_topology=SimpleNamespace(matches_expected_endpoints=True),
    )
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return expected_result

    monkeypatch.setattr(irc_workflow, "run_bidirectional_irc", fake_run)
    screening = SimpleNamespace(
        irc_eligible=True,
        passes_screening=True,
        geometry_progress_consistent=False,
        frequency_scope="full",
        frequency_analysis=SimpleNamespace(
            hessian_ev_per_a2=np.eye(3),
            primary_imaginary_mode=np.ones((1, 3)),
        ),
    )
    path = SimpleNamespace(
        problem=SimpleNamespace(
            reactant=atoms,
            product=atoms.copy(),
            metadata={},
        )
    )
    saddle = SimpleNamespace(atoms=atoms)
    config = TransitionSearchConfig()

    result = follow_and_validate_irc(
        path=path,
        saddle=saddle,
        screening=screening,
        config=type(config.irc)(enabled=True),
        cpu_calculator=None,
        cancel_event=Event(),
        progress_callback=None,
    )

    assert called is True
    assert result.performed is True
    assert result.validation_passed is True
    assert result.result is expected_result


def test_bidirectional_irc_rejects_partial_hessian() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="full Cartesian Hessian"):
        run_bidirectional_irc(
            atoms,
            np.zeros((2, 2)),
            validated_cartesian_mode=np.ones((1, 3)),
            calculator=None,
            expected_reactant=atoms,
            expected_product=atoms,
            config=BidirectionalIrcConfig(),
            cancel_event=Event(),
        )


def test_unconverged_irc_endpoint_is_relaxed_to_contact_minimum() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms = Atoms("H2", positions=reference)
    frame = IrcFrame(
        step=10,
        energy_ev=0.5,
        max_force_ev_per_a=1.0,
        positions_a=np.asarray([[1.0, 0.0, 0.0]]),
        converged=False,
    )

    result = _relax_endpoint(
        atoms,
        frame,
        endpoint="endpoint_a",
        calculator=HarmonicEndpointCalculator(),
        config=BidirectionalIrcConfig(
            endpoint_relaxation_fmax=0.01,
            endpoint_relaxation_max_steps=50,
        ),
        cancel_event=Event(),
        on_frame=None,
    )

    assert result.performed is True
    assert result.converged is True
    assert result.final_energy_ev < result.initial_energy_ev
    assert result.final_max_force_ev_per_a <= 0.01
    assert result.trajectory


def test_incomplete_irc_result_is_retained_with_structured_failure(
    monkeypatch,
) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    relaxation = SimpleNamespace(converged=True)
    partial = SimpleNamespace(
        warnings=("IRC_INNER_LOOP_NOT_CONVERGED",),
        endpoint_a_relaxation=relaxation,
        endpoint_b_relaxation=relaxation,
        path_complete=False,
        endpoint_topology=SimpleNamespace(matches_expected_endpoints=True),
    )
    monkeypatch.setattr(irc_workflow, "run_bidirectional_irc", lambda *a, **k: partial)
    screening = SimpleNamespace(
        irc_eligible=True,
        frequency_scope="full",
        frequency_analysis=SimpleNamespace(
            hessian_ev_per_a2=np.eye(3),
            primary_imaginary_mode=np.ones((1, 3)),
        ),
    )
    path = SimpleNamespace(
        problem=SimpleNamespace(reactant=atoms, product=atoms.copy(), metadata={})
    )

    result = follow_and_validate_irc(
        path=path,
        saddle=SimpleNamespace(atoms=atoms),
        screening=screening,
        config=IrcConfig(enabled=True),
        cpu_calculator=None,
        cancel_event=Event(),
        progress_callback=None,
    )

    assert result.performed is True
    assert result.result is partial
    assert result.validation_passed is False
    assert result.failure_code == "IRC_INNER_LOOP_NOT_CONVERGED"

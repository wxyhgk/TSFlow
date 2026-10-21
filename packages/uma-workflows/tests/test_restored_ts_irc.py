from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from uma_workflows.irc.workflow import (
    BidirectionalIrcConfig,
    _relax_endpoint,
    run_bidirectional_irc,
)
from uma_workflows.irc.models import IrcFrame
from uma_workflows.irc.validation import IrcValidationError
from uma_workflows.transition_states.supplied_guess import (
    SuppliedGuessConfig,
    run_supplied_guess,
)


class Harmonic(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float((self.atoms.positions**2).sum() / 2),
            "forces": -self.atoms.positions.copy(),
        }


def test_endpoint_relaxation_lowers_energy():
    atoms = Atoms("H", positions=[[0.3, 0, 0]])
    result = _relax_endpoint(
        atoms,
        IrcFrame(0, 0.045, 0.3, atoms.positions.copy(), False),
        endpoint="endpoint_a",
        calculator=Harmonic(),
        config=BidirectionalIrcConfig(endpoint_relaxation_fmax=0.01),
        cancel_event=Event(),
        on_frame=None,
    )
    assert result.converged and result.final_energy_ev < result.initial_energy_ev
    assert result.final_max_force_ev_per_a <= 0.01 and result.trajectory


def test_irc_rejects_partial_and_nonfinite_hessian():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]])
    kwargs = dict(
        validated_cartesian_mode=np.ones((2, 3)),
        calculator=Harmonic(),
        expected_reactant=atoms,
        expected_product=atoms,
        config=BidirectionalIrcConfig(),
        cancel_event=Event(),
    )
    with pytest.raises(ValueError, match="full Cartesian Hessian"):
        run_bidirectional_irc(atoms, np.eye(3), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        run_bidirectional_irc(atoms, np.full((6, 6), np.nan), **kwargs)


def test_irc_rejects_nonstationary_start():
    atoms = Atoms("H", positions=[[1, 0, 0]])
    with pytest.raises(IrcValidationError, match="TS_NOT_STATIONARY"):
        run_bidirectional_irc(
            atoms,
            np.eye(3),
            validated_cartesian_mode=np.ones((1, 3)),
            calculator=Harmonic(),
            expected_reactant=atoms,
            expected_product=atoms,
            config=BidirectionalIrcConfig(),
            cancel_event=Event(),
        )


@pytest.mark.parametrize("eligible", [False, True])
def test_failed_screen_or_incomplete_path_cannot_validate(monkeypatch, eligible):
    import uma_workflows.transition_states.supplied_guess as workflow

    atoms = Atoms("H", positions=[[0, 0, 0]])
    frequency = SimpleNamespace(
        hessian_ev_per_a2=np.eye(3),
        primary_imaginary_mode=np.ones((1, 3)),
        hessian_fallback_reason=None,
    )
    monkeypatch.setattr(
        workflow, "numerical_frequency_analysis", lambda *a, **k: frequency
    )
    monkeypatch.setattr(
        workflow,
        "refine_geometric_saddle",
        lambda *a, **k: SimpleNamespace(converged=True),
    )
    monkeypatch.setattr(
        workflow,
        "screen_transition_state",
        lambda *a, **k: SimpleNamespace(
            warnings=(),
            irc_eligible=eligible,
            failure_code="MULTIPLE_IMAGINARY_MODES",
            pes_index1_saddle_valid=eligible,
            frequency_analysis=frequency,
        ),
    )

    def follow(*args, **kwargs):
        assert eligible, "Invalid saddle must not run IRC"
        return SimpleNamespace(
            path_complete=False, warnings=("IRC_INNER_LOOP_NOT_CONVERGED",)
        )

    monkeypatch.setattr(workflow, "run_bidirectional_irc", follow)
    result = run_supplied_guess(
        atoms,
        atoms,
        atoms,
        calculator=Harmonic(),
        config=SuppliedGuessConfig(),
        cancel_event=Event(),
    )
    assert not result.validated and result.irc_performed == eligible
    assert result.irc_error == (
        "IRC_INNER_LOOP_NOT_CONVERGED" if eligible else "MULTIPLE_IMAGINARY_MODES"
    )


@pytest.mark.parametrize("complete", [False, True])
def test_stage_uses_pes_eligibility_and_preserves_incomplete_result(
    monkeypatch, complete
):
    from uma_workflows.transition_states.workflow import irc as stage
    from uma_workflows.transition_states.workflow.models import IrcConfig

    atoms = Atoms("H", positions=[[0, 0, 0]])
    evidence = SimpleNamespace(
        path_complete=complete,
        endpoint_a_relaxation=SimpleNamespace(converged=True),
        endpoint_b_relaxation=SimpleNamespace(converged=True),
        endpoint_topology=SimpleNamespace(matches_expected_endpoints=True),
        relaxed_endpoint_distance_delta_a=1.0,
        warnings=() if complete else ("IRC_INNER_LOOP_NOT_CONVERGED",),
    )
    monkeypatch.setattr(stage, "run_bidirectional_irc", lambda *a, **k: evidence)
    screen = SimpleNamespace(
        irc_eligible=True,
        passes_screening=False,
        frequency_scope="full",
        frequency_analysis=SimpleNamespace(
            hessian_ev_per_a2=np.eye(3), primary_imaginary_mode=np.ones((1, 3))
        ),
    )
    result = stage.follow_and_validate_irc(
        SimpleNamespace(
            problem=SimpleNamespace(reactant=atoms, product=atoms, metadata={})
        ),
        SimpleNamespace(atoms=atoms),
        screen,
        config=IrcConfig(enabled=True),
        cpu_calculator=None,
        cancel_event=Event(),
        progress_callback=None,
    )
    assert result.performed and result.result is evidence
    assert result.validation_passed is complete
    assert result.failure_code == (None if complete else "IRC_INNER_LOOP_NOT_CONVERGED")

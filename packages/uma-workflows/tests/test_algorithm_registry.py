from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.transition_states.algorithm_api import (
    AlgorithmMetadata,
    PathCandidate,
    PathInitializationResult,
    PathOptimizationResult,
    PathParameterization,
    ReactionPath,
    TransitionStateProblem,
)
from uma_workflows.transition_states.algorithm_registry import (
    AlgorithmCompatibility,
    AlgorithmRegistration,
    AlgorithmRegistry,
)
from uma_workflows.transition_states.builtin_algorithms import (
    build_builtin_algorithm_registry,
)
from uma_workflows.transition_states.direct_maxflux import DirectMaxFluxConfig
from uma_workflows.transition_states.fbenm import FbenmConfig
from uma_workflows.transition_states.workflow.models import PathConfig
from uma_workflows.transition_states.workflow.path import optimize_reaction_path
from uma_workflows.transition_states.workflow.path_candidate import (
    optimize_path_candidate,
)


class YEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = float(self.atoms.positions[0, 1])
        self.results["forces"] = np.zeros_like(self.atoms.positions)


class RecordingInitializer:
    metadata = AlgorithmMetadata("test.initializer", "1", "Test Initializer")

    def __init__(self, _config) -> None:
        self.called = False

    def generate(self, problem, **_kwargs) -> PathInitializationResult:
        self.called = True
        positions = np.stack(
            [
                problem.reactant.positions,
                np.asarray([[0.0, 1.0, 0.0]]),
                problem.product.positions,
            ]
        )
        return PathInitializationResult(candidates=(PathCandidate(
            path=ReactionPath(positions, np.asarray([0.0, 0.5, 1.0])),
            candidate_index=0,
            parameterization=PathParameterization.from_array(
                "test.path", positions
            ),
        ),))


class RecordingOptimizer:
    metadata = AlgorithmMetadata("test.optimizer", "2", "Test Optimizer")

    def __init__(self, _config) -> None:
        self.called = False

    def optimize(self, _problem, candidate, **_kwargs) -> PathOptimizationResult:
        self.called = True
        return PathOptimizationResult(
            path=candidate.path,
            converged=True,
            iterations=1,
            optimizer_status="ok",
        )


def _test_registry() -> AlgorithmRegistry:
    registry = AlgorithmRegistry()
    registry.register(AlgorithmRegistration(
        role="path_initializer",
        metadata=RecordingInitializer.metadata,
        config_type=FbenmConfig,
        factory=RecordingInitializer,
        compatibility=AlgorithmCompatibility(produced_parameterization="test.path"),
    ))
    registry.register(AlgorithmRegistration(
        role="path_optimizer",
        metadata=RecordingOptimizer.metadata,
        config_type=DirectMaxFluxConfig,
        factory=RecordingOptimizer,
        compatibility=AlgorithmCompatibility(
            accepted_parameterizations=frozenset({"test.path"})
        ),
    ))
    return registry


def test_builtin_registry_has_stable_algorithm_catalog() -> None:
    catalog = build_builtin_algorithm_registry().describe()
    assert [item.metadata.algorithm_id for item in catalog] == [
        "fbenm.correlated",
        "dmf.direct_maxflux",
    ]
    assert [item.role for item in catalog] == [
        "path_initializer",
        "path_optimizer",
    ]


def test_registry_rejects_duplicates_unknown_roles_and_bad_configs() -> None:
    registry = _test_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.describe()[0])
    with pytest.raises(ValueError, match="unknown path_initializer"):
        registry.create_initializer("missing", FbenmConfig(nmove=3))
    with pytest.raises(ValueError, match="has role path_optimizer"):
        registry.create_initializer(
            "test.optimizer", FbenmConfig(nmove=3)
        )
    with pytest.raises(TypeError, match="requires config FbenmConfig"):
        registry.create_initializer("test.initializer", object())


def test_registry_checks_path_parameterization_compatibility() -> None:
    registry = _test_registry()
    candidate = PathCandidate(
        path=ReactionPath(np.zeros((2, 1, 3)), np.asarray([0.0, 1.0])),
        parameterization=PathParameterization.from_array(
            "unsupported", np.zeros((2, 1, 3))
        ),
    )
    with pytest.raises(ValueError, match="does not accept"):
        registry.create_optimizer(
            "test.optimizer",
            DirectMaxFluxConfig(
                nmove=3,
                max_iterations=1,
                convergence="loose",
                path_bias=None,  # type: ignore[arg-type]
            ),
            candidate=candidate,
        )


def test_workflow_path_uses_explicitly_selected_registry_algorithms() -> None:
    reactant = Atoms("H", positions=[[-1.0, 0.0, 0.0]])
    product = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    result = optimize_reaction_path(
        reactant,
        product,
        None,  # type: ignore[arg-type]
        reaction_coordinates=(),
        config=PathConfig(
            initializer_id="test.initializer",
            optimizer_id="test.optimizer",
            ensemble_size=1,
            nmove=3,
            path_points=3,
        ),
        device="cpu",
        calculator=YEnergyCalculator(),
        cpu_calculator=YEnergyCalculator(),
        cancel_event=Event(),
        progress_callback=None,
        algorithm_registry=_test_registry(),
    )

    assert result.initializer_algorithm_id == "test.initializer"
    assert result.initializer_algorithm_version == "1"
    assert result.optimizer_algorithm_id == "test.optimizer"
    assert result.optimizer_algorithm_version == "2"
    assert result.has_interior_maximum is True


def test_single_candidate_helper_uses_registry_and_unbiased_energy_profile() -> None:
    reactant = Atoms("H", positions=[[-1.0, 0.0, 0.0]])
    product = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    positions = np.asarray(
        [[[-1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]]
    )
    candidate = PathCandidate(
        path=ReactionPath(positions, np.asarray([0.0, 0.5, 1.0])),
        candidate_index=2,
        parameterization=PathParameterization.from_array("test.path", positions),
    )

    result = optimize_path_candidate(
        TransitionStateProblem(reactant, product),
        candidate,
        config=PathConfig(
            optimizer_id="test.optimizer",
            nmove=3,
            path_points=3,
        ),
        device="cpu",
        calculator=YEnergyCalculator(),
        cpu_calculator=YEnergyCalculator(),
        cancel_event=Event(),
        algorithm_registry=_test_registry(),
    )

    assert result.candidate_index == 2
    assert result.optimizer_algorithm_id == "test.optimizer"
    assert result.cpu_energies_ev == (0.0, 1.0, 0.0)
    assert result.has_interior_maximum is True

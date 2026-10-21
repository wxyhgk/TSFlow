from dataclasses import replace
import json
from threading import Event

from ase import Atoms
from ase.calculators.emt import EMT

from uma_workflows.transition_states import (
    DirectMaxFluxConfig,
    DirectMaxFluxPathOptimizer,
    FbenmConfig,
    FbenmPathInitializer,
    PathBias,
    TransitionStateProblem,
    select_best_candidate,
)


def _zero_bias(symbols: tuple[str, ...]) -> PathBias:
    return PathBias(
        symbols=symbols,
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


def test_direct_maxflux_implements_common_path_optimizer_result() -> None:
    reactant = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.0, 0.0]],
    )
    product = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
    )
    problem = TransitionStateProblem(reactant=reactant, product=product)
    calculator = EMT()
    initialized = FbenmPathInitializer(FbenmConfig(nmove=3)).generate(
        problem,
        calculator=calculator,
        cancel_token=Event(),
    )
    _, candidate = select_best_candidate(initialized.candidates)

    result = DirectMaxFluxPathOptimizer(
        DirectMaxFluxConfig(
            nmove=3,
            max_iterations=3,
            convergence="loose",
            path_bias=_zero_bias(tuple(reactant.get_chemical_symbols())),
            result_points=5,
        )
    ).optimize(
        problem,
        candidate,
        calculator=calculator,
        cancel_token=Event(),
    )

    assert result.path.positions_a.shape == (5, 3, 3)
    assert result.iterations > 0
    assert result.path.tangents_a is not None
    assert result.status_code is not None
    assert "artifacts" not in result.metadata
    json.dumps(result.to_dict())

    generic_result = DirectMaxFluxPathOptimizer(
        DirectMaxFluxConfig(
            nmove=3,
            max_iterations=1,
            convergence="loose",
            path_bias=_zero_bias(tuple(reactant.get_chemical_symbols())),
            result_points=5,
        )
    ).optimize(
        problem,
        replace(candidate, parameterization=None),
        calculator=calculator,
        cancel_token=Event(),
    )

    assert generic_result.path.positions_a.shape == (5, 3, 3)

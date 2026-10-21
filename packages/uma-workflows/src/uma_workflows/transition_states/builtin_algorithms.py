from __future__ import annotations

from uma_workflows.transition_states.algorithm_registry import (
    AlgorithmCompatibility,
    AlgorithmRegistration,
    AlgorithmRegistry,
)


def build_builtin_algorithm_registry() -> AlgorithmRegistry:
    """Build a fresh registry so tests and callers never share mutable state."""
    from uma_workflows.transition_states.direct_maxflux import (
        DirectMaxFluxConfig,
        DirectMaxFluxPathOptimizer,
    )
    from uma_workflows.transition_states.fbenm import FbenmConfig, FbenmPathInitializer

    registry = AlgorithmRegistry()
    registry.register(
        AlgorithmRegistration(
            role="path_initializer",
            metadata=FbenmPathInitializer.metadata,
            config_type=FbenmConfig,
            factory=FbenmPathInitializer,
            compatibility=AlgorithmCompatibility(
                produced_parameterization="pydmf.bspline"
            ),
            capabilities=(
                "disconnected_endpoints",
                "directional_ensemble",
                "correlated_path",
            ),
        )
    )
    registry.register(
        AlgorithmRegistration(
            role="path_optimizer",
            metadata=DirectMaxFluxPathOptimizer.metadata,
            config_type=DirectMaxFluxConfig,
            factory=DirectMaxFluxPathOptimizer,
            compatibility=AlgorithmCompatibility(
                accepts_unparameterized=True,
                accepted_parameterizations=frozenset({"pydmf.bspline"}),
            ),
            capabilities=(
                "continuous_bspline_path",
                "adaptive_energy_sampling",
                "path_bias",
            ),
        )
    )
    return registry

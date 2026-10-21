from __future__ import annotations

from dataclasses import dataclass
from threading import Event

from ase import Atoms

from uma_workflows.transition_states.workflow.metrics import max_force
from uma_workflows.transition_states.workflow.models import (
    Device,
    EndpointConfig,
    WorkflowProgress,
    WorkflowProgressCallback,
)
from uma_workflows.endpoints import (
    EndpointPreoptimization,
    covalent_fragments,
    preoptimize_endpoint,
)


@dataclass(frozen=True)
class EndpointStageResult:
    reactant: EndpointPreoptimization
    product: EndpointPreoptimization
    warnings: tuple[str, ...]


def prepare_endpoints(
    reactant: Atoms,
    product: Atoms,
    *,
    config: EndpointConfig,
    device: Device,
    mps_calculator,
    cpu_calculator,
    cancel_event: Event,
    progress_callback: WorkflowProgressCallback | None,
) -> EndpointStageResult:
    reactant_result = _empty_result(reactant)
    product_result = _empty_result(product)
    if not config.enabled:
        return EndpointStageResult(reactant_result, product_result, ())

    def optimize(
        atoms: Atoms,
        label: str,
    ) -> EndpointPreoptimization:
        def report(step: int, max_steps: int, current: Atoms) -> None:
            if progress_callback is None:
                return
            progress_callback(
                WorkflowProgress(
                    stage="endpoint_preoptimization",
                    step=step,
                    max_steps=max_steps,
                    energy_ev=float(current.get_potential_energy()),
                    max_force_ev_per_a=max_force(current.get_forces()),
                    device=(
                        device
                        if step <= config.max_steps
                        else "cpu"
                    ),
                    positions_a=current.positions.copy(),
                    name=f"{label} endpoint preoptimization",
                    reactant_positions_a=(
                        current.positions.copy()
                        if label == "Reactant"
                        else reactant.positions.copy()
                    ),
                    product_positions_a=(
                        current.positions.copy()
                        if label == "Product"
                        else None
                    ),
                )
            )

        return preoptimize_endpoint(
            atoms,
            optimization_device=device,
            mps_calculator=mps_calculator,
            cpu_calculator=cpu_calculator,
            fmax=config.fmax,
            max_steps=config.max_steps,
            cancel_event=cancel_event,
            on_step=report,
        )

    reactant_result = optimize(reactant, "Reactant")
    product_result = optimize(product, "Product")
    warnings = _warnings(reactant_result, product_result)
    return EndpointStageResult(reactant_result, product_result, tuple(warnings))


def _empty_result(atoms: Atoms) -> EndpointPreoptimization:
    return EndpointPreoptimization(
        False,
        len(covalent_fragments(atoms)),
        0,
        0,
        True,
        [],
    )


def _warnings(
    reactant: EndpointPreoptimization,
    product: EndpointPreoptimization,
) -> list[str]:
    warnings = []
    for label, result in (("reactant", reactant), ("product", product)):
        if result.performed and not result.converged:
            warnings.append(
                f"The constrained {label} endpoint preoptimization did not converge."
            )
    if reactant.fragment_count > 1 or product.fragment_count > 1:
        warnings.append(
            "Disconnected endpoint fragments were internally relaxed while "
            "one deterministic anchor atom per fragment was fixed."
        )
    return warnings

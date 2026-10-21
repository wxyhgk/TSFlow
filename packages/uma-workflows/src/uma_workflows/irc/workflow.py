from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.irc.models import IrcFrame
from uma_workflows.irc.topology import (
    IrcEndpointTopologyMatch,
    match_irc_endpoint_topologies,
)
from uma_workflows.irc.validation import (
    IrcValidationError,
    endpoint_internal_distance_delta,
    require_mode_gradient_bracket,
    require_opposite_first_steps,
)
from uma_workflows.irc.workflow_internals import (
    follow_direction,
    mode_and_gradient_bracket,
    normalize_frames,
    with_positions,
)


@dataclass(frozen=True)
class BidirectionalIrcConfig:
    engine: Literal["geometric", "sella"] = "geometric"
    steps: int = 30
    dx: float = 0.12
    inner_steps: int = 10
    inner_fmax: float = 0.01
    fmax: float = 0.05
    ts_fmax: float = 0.05
    endpoint_distance_threshold_a: float = 0.02
    require_endpoint_match: bool = True
    relax_endpoints: bool = True
    endpoint_relaxation_fmax: float = 0.05
    endpoint_relaxation_max_steps: int = 200


@dataclass(frozen=True)
class IrcPathFrame:
    path_coordinate: float
    energy_ev: float
    relative_energy_ev: float
    max_force_ev_per_a: float
    positions_a: np.ndarray
    converged: bool


@dataclass(frozen=True)
class IrcEndpointRelaxation:
    endpoint: str
    performed: bool
    converged: bool
    initial_energy_ev: float
    final_energy_ev: float
    final_max_force_ev_per_a: float
    initial_positions_a: np.ndarray
    final_positions_a: np.ndarray
    trajectory: tuple[IrcFrame, ...]
    error: str | None = None


def _relax_endpoint(
    atoms, frame, *, endpoint, calculator, config, cancel_event, on_frame
):
    from ase.optimize import BFGS, LBFGS

    image = with_positions(atoms, frame.positions_a)
    image.calc = calculator
    frames = []
    initial = float(image.get_potential_energy())

    def report():
        raise_if_cancelled(cancel_event)
        force = float(np.linalg.norm(image.get_forces(), axis=1).max(initial=0.0))
        current = IrcFrame(
            len(frames),
            float(image.get_potential_energy()),
            force,
            image.positions.copy(),
            force <= config.endpoint_relaxation_fmax,
        )
        frames.append(current)
        if on_frame is not None:
            on_frame(endpoint + "_relaxation", current)

    report()
    if config.relax_endpoints:
        if endpoint == "endpoint_a":
            # The reactant branch is already close to its contact minimum and
            # converges faster with the standard BFGS update.
            optimizer = BFGS(image, logfile=None, maxstep=0.1)
        else:
            # The product branch can contain a soft Li coordination coordinate.
            # A damped, small-step update avoids jumping across that basin.
            optimizer = LBFGS(
                image,
                logfile=None,
                maxstep=0.03,
                memory=20,
                damping=0.5,
            )
        optimizer.attach(report, interval=1)
        optimizer.run(
            fmax=config.endpoint_relaxation_fmax,
            steps=config.endpoint_relaxation_max_steps,
        )
    last = frames[-1]
    return IrcEndpointRelaxation(
        endpoint,
        config.relax_endpoints,
        last.converged,
        initial,
        last.energy_ev,
        last.max_force_ev_per_a,
        frame.positions_a.copy(),
        last.positions_a.copy(),
        tuple(frames),
    )


@dataclass(frozen=True)
class BidirectionalIrcResult:
    frames: tuple[IrcPathFrame, ...]
    forward_frame_count: int
    reverse_frame_count: int
    forward_inner_loop_failed: bool
    reverse_inner_loop_failed: bool
    forward_first_projection: float
    reverse_first_projection: float
    endpoint_distance_delta_a: float
    endpoint_topology: IrcEndpointTopologyMatch
    warnings: tuple[str, ...]
    path_complete: bool
    raw_endpoint_topology: IrcEndpointTopologyMatch
    endpoint_a_relaxation: IrcEndpointRelaxation
    endpoint_b_relaxation: IrcEndpointRelaxation
    relaxed_endpoint_distance_delta_a: float


def run_bidirectional_irc(
    transition_state: Atoms,
    hessian_ev_per_a2: np.ndarray,
    *,
    validated_cartesian_mode: np.ndarray,
    calculator,
    expected_reactant: Atoms,
    expected_product: Atoms,
    config: BidirectionalIrcConfig,
    cancel_event: CancellationToken,
    on_frame: Callable[[str, IrcFrame], None] | None = None,
) -> BidirectionalIrcResult:
    atoms = transition_state.copy()
    atoms.set_constraint()
    atoms.calc = calculator
    hessian = np.asarray(hessian_ev_per_a2, dtype=float)
    expected_shape = (3 * len(atoms), 3 * len(atoms))
    if hessian.shape != expected_shape:
        raise ValueError(
            f"IRC requires a full Cartesian Hessian with shape {expected_shape}; "
            f"received {hessian.shape}"
        )
    if not np.isfinite(hessian).all():
        raise ValueError("IRC Hessian must be finite")
    mode = np.asarray(validated_cartesian_mode, dtype=float)
    if mode.shape != (len(atoms), 3) or not np.isfinite(mode).all():
        raise ValueError(
            "IRC validated mode must contain finite full-system coordinates"
        )
    raise_if_cancelled(cancel_event)
    ts_energy = float(atoms.get_potential_energy())
    ts_forces = np.asarray(atoms.get_forces(), dtype=float)
    ts_fmax = float(np.linalg.norm(ts_forces, axis=1).max(initial=0.0))
    if ts_fmax > config.ts_fmax:
        raise IrcValidationError(
            "TS_NOT_STATIONARY_ON_SELECTED_PES: "
            f"Fmax={ts_fmax:.6f} eV/A exceeds {config.ts_fmax:.6f} eV/A"
        )

    _mode, gradients = mode_and_gradient_bracket(
        atoms,
        validated_cartesian_mode,
        calculator,
        dx=config.dx,
        cancel_event=cancel_event,
    )
    require_mode_gradient_bracket(*gradients)
    forward_atoms, forward = follow_direction(
        atoms,
        hessian,
        "forward",
        validated_cartesian_mode,
        calculator,
        config,
        cancel_event,
        on_frame,
    )
    reverse_atoms, reverse = follow_direction(
        atoms,
        hessian,
        "reverse",
        validated_cartesian_mode,
        calculator,
        config,
        cancel_event,
        on_frame,
    )
    path_complete = not (forward.inner_loop_failed or reverse.inner_loop_failed)
    if len(forward.frames) < 2 or len(reverse.frames) < 2:
        raise IrcValidationError(
            "IRC_INCOMPLETE: each direction must produce at least one path step"
        )
    forward_projection, reverse_projection = require_opposite_first_steps(
        forward.frames[1].positions_a,
        reverse.frames[1].positions_a,
        atoms.positions,
        atoms.get_masses(),
        forward_atoms.info["irc_initial_mode"],
        reverse_atoms.info["irc_initial_mode"],
    )
    raw_frames = (
        list(reversed(reverse.frames[1:]))
        + [IrcFrame(0, ts_energy, ts_fmax, atoms.positions.copy(), False)]
        + forward.frames[1:]
    )
    endpoint_delta = endpoint_internal_distance_delta(
        raw_frames[0].positions_a,
        raw_frames[-1].positions_a,
    )
    if endpoint_delta < config.endpoint_distance_threshold_a:
        raise IrcValidationError(
            "IRC_SAME_BASIN: endpoint internal-distance change "
            f"{endpoint_delta:.6f} A is below "
            f"{config.endpoint_distance_threshold_a:.6f} A"
        )
    topology = match_irc_endpoint_topologies(
        with_positions(atoms, raw_frames[0].positions_a),
        with_positions(atoms, raw_frames[-1].positions_a),
        expected_reactant,
        expected_product,
    )
    raw_topology = topology
    endpoint_a = _relax_endpoint(
        atoms,
        raw_frames[0],
        endpoint="endpoint_a",
        calculator=calculator,
        config=config,
        cancel_event=cancel_event,
        on_frame=on_frame,
    )
    endpoint_b = _relax_endpoint(
        atoms,
        raw_frames[-1],
        endpoint="endpoint_b",
        calculator=calculator,
        config=config,
        cancel_event=cancel_event,
        on_frame=on_frame,
    )
    topology = match_irc_endpoint_topologies(
        with_positions(atoms, endpoint_a.final_positions_a),
        with_positions(atoms, endpoint_b.final_positions_a),
        expected_reactant,
        expected_product,
    )
    relaxed_delta = endpoint_internal_distance_delta(
        endpoint_a.final_positions_a, endpoint_b.final_positions_a
    )
    frames = normalize_frames(raw_frames, atoms.get_masses(), IrcPathFrame)
    warnings = []
    if not path_complete:
        warnings.append("IRC_INNER_LOOP_NOT_CONVERGED")
    if not (endpoint_a.converged and endpoint_b.converged):
        warnings.append("IRC_ENDPOINT_RELAXATION_NOT_CONVERGED")
    if not topology.matches_expected_endpoints:
        warnings.append("IRC_ENDPOINT_TOPOLOGY_MISMATCH")
    if relaxed_delta < config.endpoint_distance_threshold_a:
        warnings.append("IRC_SAME_BASIN_AFTER_RELAXATION")
    for direction, direction_result in (
        ("forward", forward),
        ("reverse", reverse),
    ):
        energy_drop = direction_result.frames[-1].energy_ev - ts_energy
        if energy_drop > -0.005:
            warnings.append(
                f"The {direction} IRC branch lowered the energy by only "
                f"{abs(energy_drop):.6f} eV; it may be a shallow adjacent basin "
                "or may require a longer/restarted IRC trajectory."
            )
    return BidirectionalIrcResult(
        frames=frames,
        forward_frame_count=len(forward.frames),
        reverse_frame_count=len(reverse.frames),
        forward_inner_loop_failed=forward.inner_loop_failed,
        reverse_inner_loop_failed=reverse.inner_loop_failed,
        forward_first_projection=forward_projection,
        reverse_first_projection=reverse_projection,
        endpoint_distance_delta_a=endpoint_delta,
        endpoint_topology=topology,
        warnings=tuple(warnings),
        path_complete=path_complete,
        raw_endpoint_topology=raw_topology,
        endpoint_a_relaxation=endpoint_a,
        endpoint_b_relaxation=endpoint_b,
        relaxed_endpoint_distance_delta_a=relaxed_delta,
    )

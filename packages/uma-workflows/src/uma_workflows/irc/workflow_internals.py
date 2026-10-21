from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.irc.geometric_runner import follow_geometric_irc_direction
from uma_workflows.irc.models import IrcDirectionResult, IrcFrame
from uma_workflows.irc.validation import mass_weighted_arc_coordinates


def mode_and_gradient_bracket(
    atoms: Atoms,
    cartesian_mode: np.ndarray,
    calculator,
    *,
    dx: float,
    cancel_event: CancellationToken,
) -> tuple[np.ndarray, tuple[float, float]]:
    sqrt_masses = np.repeat(np.sqrt(atoms.get_masses()), 3)
    cartesian_mode = np.asarray(cartesian_mode, dtype=float).reshape(-1)
    mass_mode = cartesian_mode * sqrt_masses
    mass_norm = float(np.linalg.norm(mass_mode))
    if mass_norm <= 1.0e-14:
        raise ValueError("validated IRC mode has zero mass-weighted norm")
    mass_mode /= mass_norm
    cartesian_mode = mass_mode / sqrt_masses
    gradients = []
    for sign in (-1.0, 1.0):
        raise_if_cancelled(cancel_event)
        displaced = atoms.copy()
        displaced.positions = (
            atoms.positions + sign * dx * cartesian_mode.reshape(-1, 3)
        )
        displaced.calc = calculator
        gradient = -np.asarray(displaced.get_forces(), dtype=float).reshape(-1)
        gradients.append(float(np.dot(gradient / sqrt_masses, mass_mode)))
    return cartesian_mode, (gradients[0], gradients[1])


def follow_direction(
    atoms: Atoms,
    hessian: np.ndarray,
    direction: str,
    initial_cartesian_mode: np.ndarray,
    calculator,
    config,
    cancel_event: CancellationToken,
    on_frame: Callable[[str, IrcFrame], None] | None,
) -> tuple[Atoms, IrcDirectionResult]:
    if config.engine == "geometric":
        geometric_direction = "backward" if direction == "reverse" else direction
        result = follow_geometric_irc_direction(
            atoms,
            hessian,
            geometric_direction,
            calculator,
            trust_a=config.dx,
            fmax=config.fmax,
            steps=config.steps,
            cancel_event=cancel_event,
            regularize_secondary_negative_modes=True,
            on_frame=(
                (lambda frame: on_frame(direction, frame))
                if on_frame is not None
                else None
            ),
        )
        if not result.frames:
            raise RuntimeError("geomeTRIC IRC returned an empty trajectory")
        image = atoms.copy()
        image.set_positions(result.frames[-1].positions_a)
        image.calc = calculator
        image.info["irc_initial_mode"] = np.asarray(initial_cartesian_mode).copy()
        return image, result
    from uma_workflows.irc.runner import (
        follow_sella_irc_direction,
        prepare_sella_irc,
    )

    image = atoms.copy()
    image.calc = calculator
    optimizer = prepare_sella_irc(
        image,
        hessian,
        initial_cartesian_mode=initial_cartesian_mode,
        dx=config.dx,
        inner_steps=config.inner_steps,
    )
    image.info["irc_initial_mode"] = optimizer.v0ts.copy()
    result = follow_sella_irc_direction(
        optimizer,
        image,
        direction,
        fmax=config.fmax,
        inner_fmax=config.inner_fmax,
        steps=config.steps,
        on_frame=(
            (lambda frame: on_frame(direction, frame))
            if on_frame is not None
            else None
        ),
        cancel_event=cancel_event,
    )
    return image, result


def normalize_frames(
    frames: list[IrcFrame],
    masses: np.ndarray,
    frame_type,
) -> tuple:
    coordinates = mass_weighted_arc_coordinates(
        [frame.positions_a for frame in frames],
        masses,
    )
    minimum = min(frame.energy_ev for frame in frames)
    return tuple(
        frame_type(
            path_coordinate=coordinate,
            energy_ev=frame.energy_ev,
            relative_energy_ev=frame.energy_ev - minimum,
            max_force_ev_per_a=frame.max_force_ev_per_a,
            positions_a=frame.positions_a,
            converged=frame.converged,
        )
        for coordinate, frame in zip(coordinates, frames, strict=True)
    )


def with_positions(template: Atoms, positions: np.ndarray) -> Atoms:
    atoms = template.copy()
    values = np.asarray(positions, dtype=float)
    if values.shape != (len(atoms), 3):
        # A malformed endpoint frame may contain only the moved subset. Keep
        # the remaining atoms at the TS/template coordinates so diagnostics
        # can still be retained and validated by the caller.
        if values.ndim != 2 or values.shape[1:] != (3,) or values.shape[0] > len(atoms):
            raise ValueError("IRC endpoint frame has incompatible coordinates")
        merged = atoms.positions.copy()
        merged[: values.shape[0]] = values
        values = merged
    atoms.set_positions(values)
    atoms.calc = None
    return atoms

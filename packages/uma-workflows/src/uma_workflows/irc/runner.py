from __future__ import annotations

from typing import Callable

import numpy as np
from ase import Atoms
from sella import IRC
from sella.optimize.irc import IRCInnerLoopConvergenceFailure

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.irc.models import IrcDirectionResult, IrcFrame


def prepare_sella_irc(
    atoms: Atoms,
    hessian_ev_per_a2: np.ndarray,
    *,
    initial_cartesian_mode: np.ndarray,
    dx: float,
    inner_steps: int,
) -> IRC:
    """Initialize Sella IRC from an already computed Cartesian Hessian."""
    optimizer = IRC(
        atoms,
        logfile=None,
        trajectory=None,
        ninner_iter=inner_steps,
        irctol=1.0e-2,
        dx=dx,
        keep_going=False,
    )
    optimizer.pes.kick(0, True)
    optimizer.H0 = np.asarray(hessian_ev_per_a2, dtype=float).copy()
    mode = np.asarray(initial_cartesian_mode, dtype=float).reshape(-1)
    if mode.shape != optimizer.sqrtm.shape:
        raise ValueError("IRC initial mode shape must match all Cartesian coordinates")
    mass_norm = float(np.linalg.norm(mode * optimizer.sqrtm))
    if mass_norm <= 1.0e-14:
        raise ValueError("IRC initial mode must have nonzero mass-weighted norm")
    optimizer.v0ts = dx * mode / mass_norm
    first_nonzero = np.flatnonzero(np.abs(optimizer.v0ts) > 1.0e-14)
    if first_nonzero.size and optimizer.v0ts[first_nonzero[0]] < 0:
        optimizer.v0ts *= -1
    optimizer.pescurr = optimizer.pes.curr.copy()
    optimizer.peslast = optimizer.pes.last.copy()
    optimizer.pes.set_H(optimizer.H0.copy(), initialized=True)
    # Sella 2.5 implements this stopping rule in converged(), while modern ASE
    # invokes gradient_converged() directly.
    optimizer.gradient_converged = lambda _gradient: optimizer.converged()
    return optimizer


def follow_sella_irc_direction(
    optimizer: IRC,
    atoms: Atoms,
    direction: str,
    *,
    fmax: float,
    inner_fmax: float,
    steps: int,
    on_frame: Callable[[IrcFrame], None] | None = None,
    cancel_event: CancellationToken | None = None,
) -> IrcDirectionResult:
    """Follow and retain one IRC direction, including partial failed paths."""
    frames: list[IrcFrame] = []
    last_step = -1
    iterator = optimizer.irun(
        fmax=fmax,
        fmax_inner=inner_fmax,
        steps=steps,
        direction=direction,
    )
    inner_loop_failed = False
    try:
        if cancel_event is not None:
            raise_if_cancelled(cancel_event)
        for converged in iterator:
            if cancel_event is not None:
                raise_if_cancelled(cancel_event)
            step = int(optimizer.nsteps)
            if step == last_step:
                continue
            forces = np.asarray(atoms.get_forces(), dtype=float)
            frame = IrcFrame(
                step=step,
                energy_ev=float(atoms.get_potential_energy()),
                max_force_ev_per_a=float(
                    np.linalg.norm(forces, axis=1).max(initial=0.0)
                ),
                positions_a=atoms.positions.copy(),
                converged=bool(converged),
            )
            frames.append(frame)
            if on_frame is not None:
                on_frame(frame)
            last_step = step
    except IRCInnerLoopConvergenceFailure:
        inner_loop_failed = True
    return IrcDirectionResult(
        frames=frames,
        inner_loop_failed=inner_loop_failed,
    )

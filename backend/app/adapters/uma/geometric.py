from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree
from geometric.engine import Engine
from geometric.molecule import Molecule
from geometric.optimize import run_optimizer


@dataclass(frozen=True)
class GeometricFrame:
    step: int
    energy_ev: float
    forces_ev_per_a: np.ndarray
    positions_a: np.ndarray


@dataclass(frozen=True)
class GeometricResult:
    frames: list[GeometricFrame]
    evaluations: int

    @property
    def final(self) -> GeometricFrame:
        return self.frames[-1]


EvaluationCallback = Callable[[GeometricFrame], None]

_QUIET_LOG_CONFIG = """\
[loggers]
keys=root
[handlers]
keys=null
[formatters]
keys=plain
[logger_root]
level=CRITICAL
handlers=null
[handler_null]
class=NullHandler
level=NOTSET
formatter=plain
args=()
[formatter_plain]
format=%(message)s
"""


class UmaGeometricEngine(Engine):
    """Expose an ASE/UMA calculator to geomeTRIC in atomic units."""

    def __init__(
        self,
        molecule: Molecule,
        atoms: Atoms,
        calculator: Calculator,
        check_cancelled: Callable[[], None],
        on_evaluation: EvaluationCallback | None,
    ) -> None:
        super().__init__(molecule)
        self.atoms = atoms
        self.atoms.calc = calculator
        self.check_cancelled = check_cancelled
        self.on_evaluation = on_evaluation
        self.evaluations = 0

    def calc_new(self, coords, dirname):  # type: ignore[no-untyped-def]
        del dirname
        self.check_cancelled()
        positions_a = np.asarray(coords, dtype=float).reshape(-1, 3) * Bohr
        self.atoms.set_positions(positions_a)
        energy_ev = float(self.atoms.get_potential_energy())
        forces_ev_per_a = np.asarray(self.atoms.get_forces(), dtype=float)
        frame = GeometricFrame(
            step=self.evaluations,
            energy_ev=energy_ev,
            forces_ev_per_a=forces_ev_per_a.copy(),
            positions_a=positions_a.copy(),
        )
        self.evaluations += 1
        if self.on_evaluation is not None:
            self.on_evaluation(frame)
        # geomeTRIC expects Hartree and dE/dR in Hartree/Bohr.
        gradient = -forces_ev_per_a * Bohr / Hartree
        return {"energy": energy_ev / Hartree, "gradient": gradient.ravel()}


def optimize_with_geometric(
    atoms: Atoms,
    calculator: Calculator,
    *,
    fmax_ev_per_a: float,
    max_steps: int,
    check_cancelled: Callable[[], None],
    on_evaluation: EvaluationCallback | None = None,
) -> GeometricResult:
    """Minimize a molecule on the UMA PES using geomeTRIC TRIC coordinates."""
    if len(atoms) < 2:
        atoms.calc = calculator
        check_cancelled()
        forces = np.asarray(atoms.get_forces(), dtype=float)
        return GeometricResult(
            frames=[
                GeometricFrame(
                    step=0,
                    energy_ev=float(atoms.get_potential_energy()),
                    forces_ev_per_a=forces,
                    positions_a=atoms.positions.copy(),
                )
            ],
            evaluations=1,
        )

    molecule = Molecule()
    molecule.elem = atoms.get_chemical_symbols()
    molecule.xyzs = [atoms.positions.copy()]
    engine = UmaGeometricEngine(
        molecule,
        atoms,
        calculator,
        check_cancelled,
        on_evaluation,
    )
    gmax_hartree_per_bohr = fmax_ev_per_a * Bohr / Hartree

    with TemporaryDirectory(prefix="uma-geometric-") as temporary_dir:
        workdir = Path(temporary_dir)
        input_path = workdir / "uma.in"
        input_path.touch()
        log_config = workdir / "quiet-log.ini"
        log_config.write_text(_QUIET_LOG_CONFIG, encoding="utf-8")
        progress = run_optimizer(
            customengine=engine,
            input=str(input_path),
            prefix=str(workdir / "optimization"),
            logIni=str(log_config),
            coordsys="tric",
            maxiter=max_steps,
            # The public API exposes an ASE-style maximum atomic force. Make
            # the other geomeTRIC convergence tests non-limiting, and compute
            # the final convergence flag again from the unprojected UMA force.
            convergence_energy=1.0e6,
            convergence_grms=1.0e6,
            convergence_gmax=gmax_hartree_per_bohr,
            convergence_drms=1.0e6,
            convergence_dmax=1.0e6,
            converge=["maxiter"],
            hessian="never",
            subfrctor=2,
            verbose=False,
        )

    frames: list[GeometricFrame] = []
    for step, (positions, energy_h, gradient_h_per_bohr) in enumerate(
        zip(progress.xyzs, progress.qm_energies, progress.qm_grads, strict=True)
    ):
        gradient = np.asarray(gradient_h_per_bohr, dtype=float).reshape(-1, 3)
        frames.append(
            GeometricFrame(
                step=step,
                energy_ev=float(energy_h) * Hartree,
                forces_ev_per_a=-gradient * Hartree / Bohr,
                positions_a=np.asarray(positions, dtype=float).copy(),
            )
        )
    if not frames:
        raise RuntimeError("geomeTRIC returned an empty optimization trajectory")
    atoms.set_positions(frames[-1].positions_a)
    return GeometricResult(frames=frames, evaluations=engine.evaluations)

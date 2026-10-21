from __future__ import annotations

import re
from pathlib import Path

from app.contracts.jobs import AtomicStructure, CalculationResult


DEFAULT_GAUSSIAN_ROUTE = "#p B3LYP/6-31G(d) Opt Freq"
DEFAULT_TRANSITION_STATE_ROUTE = "#p B3LYP/6-31G(d) Opt=(TS,CalcFC) Freq"


def gaussian_stem(input_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem).strip("._-")
    return stem or "uma-optimized"


def render_gaussian_input(
    structure: AtomicStructure,
    result: CalculationResult,
    *,
    stem: str,
    route: str = DEFAULT_GAUSSIAN_ROUTE,
    nproc: int = 8,
    memory: str = "8GB",
) -> str:
    normalized_route = route.strip()
    if not normalized_route.startswith("#"):
        normalized_route = f"#p {normalized_route}"
    lines = [
        f"%chk={stem}.chk",
        f"%nprocshared={nproc}",
        f"%mem={memory}",
        normalized_route,
        "",
        (
            f"UMA optimized geometry; model energy={result.energy_ev:.10f} eV; "
            f"Fmax={result.max_force_ev_per_a:.8f} eV/A"
        ),
        "",
        f"{structure.charge} {structure.spin}",
    ]
    lines.extend(
        f"{atom.symbol:<2}  "
        + "  ".join(f"{coordinate:16.10f}" for coordinate in atom.position)
        for atom in structure.atoms
    )
    return "\n".join([*lines, "", ""])


def render_transition_state_input(
    structure: AtomicStructure,
    *,
    energy_ev: float,
    max_force_ev_per_a: float,
    stem: str = "uma-dmf-ts",
    route: str = DEFAULT_TRANSITION_STATE_ROUTE,
    nproc: int = 8,
    memory: str = "8GB",
) -> str:
    normalized_route = route.strip()
    if not normalized_route.startswith("#"):
        normalized_route = f"#p {normalized_route}"
    lines = [
        f"%chk={stem}.chk",
        f"%nprocshared={nproc}",
        f"%mem={memory}",
        normalized_route,
        "",
        (
            f"UMA/DMF transition-state candidate; energy={energy_ev:.10f} eV; "
            f"Fmax={max_force_ev_per_a:.8f} eV/A"
        ),
        "",
        f"{structure.charge} {structure.spin}",
    ]
    lines.extend(
        f"{atom.symbol:<2}  "
        + "  ".join(f"{coordinate:16.10f}" for coordinate in atom.position)
        for atom in structure.atoms
    )
    return "\n".join([*lines, "", ""])

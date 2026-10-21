from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.adapters.uma.calculation import run_energy_or_optimization
from app.adapters.uma.frequency import run_frequency_analysis


class XtbExecutor:
    """ASE-compatible GFN2-xTB executor.

    The optional ``xtb`` Python package supplies the calculator and its native
    library. Importing it lazily keeps UMA-only installations usable.
    """

    engine = "xtb"
    method = "GFN2-xTB"
    _optimization_device: Literal["cpu"] = "cpu"

    def _calculator(self, _checkpoint: Path | None, _device: str, _task_name: str):
        try:
            from xtb.ase.calculator import XTB
        except ImportError as exc:
            raise RuntimeError(
                "xTB is unavailable; install the optional Python package with "
                "`pip install xtb` (or `uv add xtb`)."
            ) from exc
        return XTB(method="GFN2-xTB")

    calculator = _calculator

    def hessian_calculator(self, *_args, **_kwargs):
        raise RuntimeError("xTB analytic Hessians are not exposed by this adapter")

    def _resolve_optimization_device(self, preferred=None):
        if preferred not in (None, "auto", "cpu"):
            raise ValueError("xTB supports CPU execution only")
        return "cpu"

    def resolve_optimization_device(self):
        return "cpu"

    def execute(self, request, checkpoint, cancel_event, progress_callback=None):
        if request.calculation == "frequency_analysis":
            return run_frequency_analysis(
                self, request, checkpoint, cancel_event, progress_callback
            )
        if request.calculation in {"single_point", "geometry_optimization"}:
            return run_energy_or_optimization(
                self, request, checkpoint, cancel_event, progress_callback
            )
        raise NotImplementedError(
            "xTB transition-state execution requires the supplied-guess workflow"
        )

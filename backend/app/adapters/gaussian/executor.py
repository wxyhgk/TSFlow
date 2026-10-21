from __future__ import annotations

import os
from pathlib import Path

from app.adapters.uma.calculation import run_energy_or_optimization
from app.adapters.uma.frequency import run_frequency_analysis


class GaussianExecutor:
    """ASE Gaussian bridge using the executable selected by ``GAUSSIAN_EXE``."""

    engine = "gaussian"
    method = "B3LYP/def2SVP"
    _optimization_device = "cpu"

    def _calculator(self, _checkpoint: Path | None, _device: str, _task_name: str):
        try:
            from ase.calculators.gaussian import Gaussian
        except ImportError as exc:  # pragma: no cover - ASE is a backend dependency
            raise RuntimeError("ASE Gaussian support is unavailable") from exc
        command = os.environ.get("GAUSSIAN_EXE")
        if not command:
            raise RuntimeError("set GAUSSIAN_EXE to the Gaussian executable path")
        return Gaussian(command=command, method="B3LYP", basis="def2SVP")

    calculator = _calculator

    def hessian_calculator(self, *_args, **_kwargs):
        raise RuntimeError("Gaussian analytic Hessian is not exposed by this adapter")

    def _resolve_optimization_device(self, preferred=None):
        if preferred not in (None, "auto", "cpu"):
            raise ValueError("Gaussian supports CPU execution only")
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
            "Gaussian transition-state execution requires an external route setup"
        )

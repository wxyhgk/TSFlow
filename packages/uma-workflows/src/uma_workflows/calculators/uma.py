from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Literal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from omegaconf import OmegaConf

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
from uma_workflows.calculators.mps import enable_experimental_mps


OptimizationDevice = Literal["auto", "cpu", "mps"]
ANALYTIC_HESSIAN_VMAP_MAX_ATOMS = 30


class UmaCalculatorFactory:
    """Create and cache ASE calculators backed by one UMA reference set."""

    def __init__(
        self,
        reference_dir: Path,
        optimization_device: OptimizationDevice = "auto",
    ) -> None:
        self._optimization_device = optimization_device
        self._atom_refs = OmegaConf.to_container(
            OmegaConf.load(reference_dir / "iso_atom_elem_refs.yaml"), resolve=True
        )
        form_config = OmegaConf.to_container(
            OmegaConf.load(reference_dir / "form_elem_refs.yaml"), resolve=True
        )
        self._form_refs = form_config.get("refs") if isinstance(form_config, dict) else None
        self._predictors: dict[
            tuple[Path, str, str | None, bool], MLIPPredictUnit
        ] = {}

    def _calculator(
        self,
        checkpoint: Path,
        device: Literal["cpu", "mps"],
        task_name: str,
    ):
        predictor = self._predictor(checkpoint, device)
        return FAIRChemCalculator(predictor, task_name=task_name)

    def _predictor(
        self,
        checkpoint: Path,
        device: Literal["cpu", "mps"],
        hessian_task: str | None = None,
        hessian_vmap: bool = True,
    ) -> MLIPPredictUnit:
        key = (checkpoint, device, hessian_task, hessian_vmap)
        predictor = self._predictors.get(key)
        if predictor is None:
            if device == "mps":
                enable_experimental_mps()
            settings = replace(
                inference_settings_default(),
                execution_mode="general",
                compile=False,
                tf32=False,
                external_graph_gen=device == "mps",
                predict_untrained_hessian=(
                    {hessian_task} if hessian_task is not None else set()
                ),
                hessian_vmap=hessian_vmap,
            )
            predictor = load_predict_unit(
                checkpoint,
                inference_settings=settings,
                device=device,
                atom_refs=self._atom_refs,
                form_elem_refs=self._form_refs,
            )
            if hessian_task is not None:
                _apply_hessian_backend_workaround(predictor, hessian_vmap)
            self._predictors[key] = predictor
        return predictor

    def calculator(
        self,
        checkpoint: Path,
        device: Literal["cpu", "mps"],
        task_name: str,
    ):
        return self._calculator(checkpoint, device, task_name)

    def hessian_calculator(
        self,
        checkpoint: Path,
        task_name: str,
        *,
        use_vmap: bool = True,
    ):
        """Return a separate CPU predictor whose output includes d2E/dR2."""
        predictor = self._predictor(
            checkpoint,
            "cpu",
            task_name,
            hessian_vmap=use_vmap,
        )
        return FAIRChemCalculator(predictor, task_name=task_name)

    def _resolve_optimization_device(self, preferred: OptimizationDevice | None = None) -> Literal["cpu", "mps"]:
        requested = preferred or self._optimization_device
        if requested == "cpu":
            return "cpu"
        mps_ready = (
            torch.backends.mps.is_available()
            and importlib.util.find_spec("pymatgen") is not None
        )
        if requested == "mps" and not mps_ready:
            raise RuntimeError(
                "MPS optimization requested, but MPS or pymatgen is unavailable"
            )
        return "mps" if mps_ready else "cpu"

    def resolve_optimization_device(self) -> Literal["cpu", "mps"]:
        return self._resolve_optimization_device()


def _apply_hessian_backend_workaround(
    predictor: MLIPPredictUnit,
    use_vmap: bool,
) -> None:
    """Correct FAIR-Chem 2.21's Hessian inference configuration."""
    backbone = predictor.model.module.backbone
    regress_config = getattr(backbone, "regress_config", None)
    if regress_config is None or not hasattr(regress_config, "hessian_vmap"):
        raise RuntimeError("FAIR-Chem predictor does not expose a Hessian backend")
    regress_config.hessian_vmap = use_vmap
    _disable_inference_third_derivative_graph()


def _disable_inference_third_derivative_graph() -> None:
    """Avoid retaining a third-derivative graph during Hessian inference.

    FAIR-Chem 2.21 passes its ``create_graph`` flag, which must be true while
    differentiating energy into forces, into the subsequent force-to-Hessian
    derivative as well.  The latter only needs a graph when training against a
    Hessian.  UMA workflows are inference-only, so retaining that graph wastes
    a large amount of memory without changing the returned Hessian.
    """
    from fairchem.core.models.uma import escn_md

    original = escn_md.compute_hessian
    if getattr(original, "_uma_workflows_inference_patch", False):
        return

    @wraps(original)
    def inference_hessian(forces, pos, vmap=True, training=False):
        return original(forces, pos, vmap=vmap, training=False)

    inference_hessian._uma_workflows_inference_patch = True
    escn_md.compute_hessian = inference_hessian

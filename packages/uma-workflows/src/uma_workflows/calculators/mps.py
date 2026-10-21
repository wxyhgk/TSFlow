from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma import outputs as uma_outputs
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit


_MPS_PATCHED = False


def enable_experimental_mps() -> None:
    """Enable UMA MPS inference while retaining unsupported CPU fallbacks."""
    global _MPS_PATCHED
    if _MPS_PATCHED:
        return

    original_setup_device = MLIPPredictUnit._setup_device
    original_move_to_device = MLIPPredictUnit.move_to_device
    original_reduce_node_to_system = uma_outputs.reduce_node_to_system
    original_set_mole_coefficients = eSCNMDMoeBackbone.set_MOLE_coefficients

    def setup_device(self: MLIPPredictUnit, device: str) -> None:
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("PyTorch MPS is not available")
            self.device = "mps"
            return
        original_setup_device(self, device)

    def move_to_device(self: MLIPPredictUnit) -> None:
        if self.device != "mps":
            original_move_to_device(self)
            return
        dtype = self.inference_settings.base_precision_dtype
        self.model.to(device=self.device, dtype=dtype)
        for task in self.model.module.tasks.values():
            task.normalizer.to(device=self.device, dtype=dtype)
            if task.element_references is not None:
                task.element_references.to(device=self.device, dtype=dtype)

    def reduce_node_to_system(
        node_values: torch.Tensor,
        batch: torch.Tensor,
        num_systems: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_values.device.type != "mps":
            return original_reduce_node_to_system(node_values, batch, num_systems)
        output_shape = (num_systems,) + node_values.shape[1:]
        system_values = torch.zeros(
            output_shape,
            device=node_values.device,
            dtype=node_values.dtype,
        )
        flat_node = node_values.reshape(node_values.shape[0], -1)
        flat_system = system_values.reshape(num_systems, -1)
        flat_system.index_add_(0, batch, flat_node)
        system_values = flat_system.reshape(output_shape)
        reduced = (
            gp_utils.reduce_from_model_parallel_region(system_values)
            if gp_utils.initialized()
            else system_values
        )
        return reduced, system_values

    def set_mole_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        """MPS replacement for FAIR-Chem's unsupported index_reduce(mean)."""
        if atomic_numbers_full.device.type != "mps" or self.num_experts == 0:
            return original_set_mole_coefficients(
                self, atomic_numbers_full, batch_full, csd_mixed_emb
            )
        with torch.autocast(device_type="mps", enabled=False):
            effective_atomic_numbers = atomic_numbers_full
            effective_batch = batch_full
            if self.training and self.composition_dropout > 0.0:
                mask = torch.rand_like(atomic_numbers_full, dtype=torch.float) > self.composition_dropout
                effective_atomic_numbers = atomic_numbers_full[mask]
                effective_batch = batch_full[mask]
            composition_by_atom = self.composition_embedding(effective_atomic_numbers)
            composition = composition_by_atom.new_zeros(csd_mixed_emb.shape[0], self.sphere_channels)
            composition.index_add_(0, effective_batch, composition_by_atom)
            counts = composition_by_atom.new_zeros(csd_mixed_emb.shape[0], 1)
            counts.index_add_(0, effective_batch, composition_by_atom.new_ones((len(effective_batch), 1)))
            if np.isclose(self.model_version, 1.0).item():
                counts += 1.0
            composition = composition / counts.clamp_min(1.0)
            expert_input = torch.vstack([composition.unsqueeze(0), csd_mixed_emb[None]])
            coefficients = self.routing_mlp(expert_input.transpose(0, 1).reshape(csd_mixed_emb.shape[0], -1))
            self.global_mole_tensors.expert_mixing_coefficients = self.mole_expert_coefficient_norm(
                self.mole_dropout(coefficients)
            )

    MLIPPredictUnit._setup_device = setup_device
    MLIPPredictUnit.move_to_device = move_to_device
    uma_outputs.reduce_node_to_system = reduce_node_to_system
    eSCNMDMoeBackbone.set_MOLE_coefficients = set_mole_coefficients
    _MPS_PATCHED = True

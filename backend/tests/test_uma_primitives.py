import io
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from app.adapters.uma.primitives import UmaPrimitiveExecutor
from app.contracts.actions import (
    ComputeFullHessianAndModesAction,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
    RunCollectiveCoordinateScanAction,
)
from app.contracts.structures import Atom, AtomicStructure
from app.infrastructure.artifacts import FileArtifactStore


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        self.results["energy"] = float(0.5 * np.sum(positions**2))
        self.results["forces"] = -positions


class FakeUma:
    def _calculator(self, *_args, **_kwargs):
        return QuadraticCalculator()

    def hessian_calculator(self, *_args, **_kwargs):
        raise AssertionError("analytic calculator should not be requested")


def _structure() -> AtomicStructure:
    return AtomicStructure(
        atoms=[Atom(symbol="H", position=(0.0, 0.0, 0.0))],
        charge=-1,
        spin=1,
        name="hydride",
    )


def test_compute_full_hessian_persists_reusable_npy_and_mode_artifacts(
    tmp_path: Path,
) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    executor = UmaPrimitiveExecutor(FakeUma(), store)
    action = ComputeFullHessianAndModesAction(
        action_type="compute_full_hessian_and_modes",
        structure=_structure(),
        parameters={"hessian_method": "finite_difference"},
    )

    result = executor.compute_full_hessian_and_modes(
        uuid4(), action, tmp_path / "checkpoint.pt", Event(), None
    )

    hessian = np.load(
        io.BytesIO(store.read_verified(result.hessian_artifact)),
        allow_pickle=False,
    )
    assert hessian.shape == (3, 3)
    assert result.hessian_scope == "full"
    assert result.hessian_artifact.kind == "full_hessian"
    assert result.mode_artifact.kind == "normal_modes"
    mode_payload = json.loads(store.read_verified(result.mode_artifact))
    assert mode_payload["structure_sha256"] == structure_fingerprint(_structure())
    assert mode_payload["model_name"] == "uma-s-1p2p1"
    assert mode_payload["task_name"] == "omol"


def test_refine_saddle_returns_unvalidated_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    structure = _structure()
    atoms = __import__("ase").Atoms("H", positions=[[0.0, 0.0, 0.0]])
    frame = SimpleNamespace(
        step=1,
        energy_ev=0.0,
        forces_ev_per_a=np.zeros((1, 3)),
        positions_a=np.zeros((1, 3)),
    )
    monkeypatch.setattr(
        "app.adapters.uma.primitives.refine_saddle_guess",
        lambda *args, **kwargs: SimpleNamespace(
            atoms=atoms,
            refinement=SimpleNamespace(converged=True),
            trajectory=(frame,),
            warnings=(),
        ),
    )
    executor = UmaPrimitiveExecutor(FakeUma(), FileArtifactStore(tmp_path / "artifacts"))
    action = RefineSaddleAction(
        action_type="refine_saddle",
        structure=structure,
        parameters={"hessian_method": "finite_difference"},
    )

    result = executor.refine_saddle(
        uuid4(), action, tmp_path / "checkpoint.pt", Event(), None
    )

    assert result.converged is True
    assert result.stationary is True
    assert result.scientific_status == "unvalidated_candidate"
    assert len(result.trajectory) == 1


def test_irc_consumes_verified_artifacts_without_expected_endpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    source_job = uuid4()
    buffer = io.BytesIO()
    np.save(buffer, np.eye(3), allow_pickle=False)
    hessian = store.write_bytes(
        source_job,
        "full-hessian-npy",
        "full_hessian",
        "hessian.npy",
        buffer.getvalue(),
        media_type="application/x-npy",
    )
    mode = store.write_json(
        source_job,
        "normal-modes-json",
        "normal_modes",
        {
            "atom_symbols": ["H"],
            "charge": -1,
            "spin": 1,
            "primary_imaginary_mode": [[1.0, 0.0, 0.0]],
            "hessian_sha256": hessian.sha256,
            "pes_index1_saddle_valid": True,
        },
    )
    relaxation = SimpleNamespace(
        endpoint="endpoint_a",
        performed=False,
        converged=True,
        initial_energy_ev=-1.0,
        final_energy_ev=-1.0,
        final_max_force_ev_per_a=0.0,
        initial_positions_a=np.asarray([[-0.2, 0.0, 0.0]]),
        final_positions_a=np.asarray([[-0.2, 0.0, 0.0]]),
        trajectory=(),
        error=None,
    )
    relaxation_b = SimpleNamespace(
        **(relaxation.__dict__ | {
            "endpoint": "endpoint_b",
            "initial_positions_a": np.asarray([[0.2, 0.0, 0.0]]),
            "final_positions_a": np.asarray([[0.2, 0.0, 0.0]]),
        })
    )
    frames = (
        SimpleNamespace(
            path_coordinate=0.0,
            energy_ev=-1.0,
            relative_energy_ev=0.0,
            max_force_ev_per_a=0.0,
            positions_a=np.asarray([[-0.2, 0.0, 0.0]]),
            converged=True,
        ),
        SimpleNamespace(
            path_coordinate=1.0,
            energy_ev=-0.9,
            relative_energy_ev=0.1,
            max_force_ev_per_a=0.0,
            positions_a=np.asarray([[0.2, 0.0, 0.0]]),
            converged=True,
        ),
    )
    monkeypatch.setattr(
        "app.adapters.uma.primitives.run_bidirectional_irc",
        lambda *args, **kwargs: SimpleNamespace(
            frames=frames,
            forward_frame_count=2,
            reverse_frame_count=2,
            path_complete=True,
            endpoint_a_relaxation=relaxation,
            endpoint_b_relaxation=relaxation_b,
            endpoint_topology=None,
            warnings=(),
        ),
    )
    action = RunBidirectionalIrcAction(
        action_type="run_bidirectional_irc",
        structure=_structure(),
        hessian_artifact=hessian,
        mode_artifact=mode,
    )
    executor = UmaPrimitiveExecutor(FakeUma(), store)

    result = executor.run_bidirectional_irc(
        uuid4(), action, tmp_path / "checkpoint.pt", Event(), None
    )

    assert result.endpoint_match_evaluated is False
    assert result.endpoint_matches_expected is None
    assert len(result.frames) == 2


def test_irc_rejects_mode_artifact_not_bound_to_hessian(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    source_job = uuid4()
    buffer = io.BytesIO()
    np.save(buffer, np.eye(3), allow_pickle=False)
    hessian = store.write_bytes(
        source_job,
        "full-hessian-npy",
        "full_hessian",
        "hessian.npy",
        buffer.getvalue(),
        media_type="application/x-npy",
    )
    mode = store.write_json(
        source_job,
        "normal-modes-json",
        "normal_modes",
        {
            "atom_symbols": ["H"],
            "charge": -1,
            "spin": 1,
            "primary_imaginary_mode": [[1.0, 0.0, 0.0]],
            "hessian_sha256": "0" * 64,
            "pes_index1_saddle_valid": True,
        },
    )
    action = RunBidirectionalIrcAction(
        action_type="run_bidirectional_irc",
        structure=_structure(),
        hessian_artifact=hessian,
        mode_artifact=mode,
    )

    with pytest.raises(ValueError, match="not bound"):
        UmaPrimitiveExecutor(FakeUma(), store).run_bidirectional_irc(
            uuid4(), action, tmp_path / "checkpoint.pt", Event(), None
        )


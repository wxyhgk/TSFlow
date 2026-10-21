from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from app.contracts.artifacts import ArtifactReference
from app.contracts.action_results import (
    ActionResult,
    BidirectionalIrcActionResult,
    HessianModesActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.jobs import (
    AtomicStructure,
    CalculationResult,
    FrequencyAnalysisResult,
    TransitionStateResult,
)
from app.infrastructure.artifacts.store import FileArtifactStore


Result = CalculationResult | TransitionStateResult | FrequencyAnalysisResult | ActionResult


def materialize_result_artifacts(
    store: FileArtifactStore,
    job_id: UUID,
    result: Result,
) -> list[ArtifactReference]:
    if isinstance(result, TransitionStateResult):
        return _transition_state_artifacts(store, job_id, result)
    if isinstance(result, CalculationResult):
        return _calculation_artifacts(store, job_id, result)
    if isinstance(result, HessianModesActionResult):
        for reference in (result.hessian_artifact, result.mode_artifact):
            store.read_verified(reference)
        return [
            result.hessian_artifact,
            result.mode_artifact,
            store.write_json(
                job_id,
                "hessian-modes-summary",
                "hessian_modes_summary",
                result.model_dump(
                    mode="json",
                    exclude={"hessian_artifact", "mode_artifact"},
                ),
            ),
        ]
    if isinstance(result, SaddleRefinementActionResult):
        references = [
            store.write_json(
                job_id,
                "saddle-action-summary",
                "saddle_action_summary",
                result.model_dump(mode="json", exclude={"trajectory"}),
            )
        ]
        if result.trajectory:
            references.extend(
                _trajectory_pair(
                    store,
                    job_id,
                    "saddle-action-trajectory",
                    "saddle_optimization_trajectory",
                    [frame.model_dump(mode="json") for frame in result.trajectory],
                    [frame.structure for frame in result.trajectory],
                    [
                        f"step={frame.step} energy_ev={frame.energy_ev:.12f}"
                        for frame in result.trajectory
                    ],
                )
            )
        return references
    if isinstance(result, BidirectionalIrcActionResult):
        references = [
            store.write_json(
                job_id,
                "irc-action-summary",
                "irc_action_summary",
                result.model_dump(
                    mode="json",
                    exclude={"frames", "endpoint_a_relaxation", "endpoint_b_relaxation"},
                ),
            )
        ]
        if result.frames:
            references.extend(
                _trajectory_pair(
                    store,
                    job_id,
                    "irc-action-trajectory",
                    "irc_trajectory",
                    [frame.model_dump(mode="json") for frame in result.frames],
                    [frame.structure for frame in result.frames],
                    [
                        f"irc_coordinate={frame.path_coordinate:.10f} "
                        f"energy_ev={frame.energy_ev:.12f}"
                        for frame in result.frames
                    ],
                )
            )
        return references
    return [
        store.write_json(
            job_id,
            "frequencies",
            "frequency_analysis",
            result.model_dump(mode="json"),
        )
    ]


def _calculation_artifacts(
    store: FileArtifactStore,
    job_id: UUID,
    result: CalculationResult,
) -> list[ArtifactReference]:
    references = [
        store.write_json(
            job_id,
            "calculation-summary",
            "calculation_summary",
            result.model_dump(mode="json", exclude={"trajectory"}),
        )
    ]
    if result.trajectory:
        references.extend(
            _trajectory_pair(
                store,
                job_id,
                "optimization-trajectory",
                "optimization_trajectory",
                [frame.model_dump(mode="json") for frame in result.trajectory],
                [frame.structure for frame in result.trajectory],
                [
                    f"step={frame.step} energy_ev={frame.energy_ev:.12f} "
                    f"fmax_ev_per_a={frame.max_force_ev_per_a:.8f}"
                    for frame in result.trajectory
                ],
            )
        )
    return references


def _transition_state_artifacts(
    store: FileArtifactStore,
    job_id: UUID,
    result: TransitionStateResult,
) -> list[ArtifactReference]:
    large_fields = {
        "path",
        "energy_profile",
        "saddle_trajectory",
        "irc_frames",
        "irc_endpoint_a_relaxation",
        "irc_endpoint_b_relaxation",
    }
    references = [
        store.write_json(
            job_id,
            "summary",
            "transition_state_summary",
            result.model_dump(mode="json", exclude=large_fields),
        ),
        store.write_json(
            job_id,
            "energy-profile",
            "reaction_energy_profile",
            [point.model_dump(mode="json") for point in result.energy_profile],
        ),
    ]
    if result.path:
        references.extend(
            _trajectory_pair(
                store,
                job_id,
                "reaction-path",
                "reaction_path",
                [frame.model_dump(mode="json") for frame in result.path],
                [frame.structure for frame in result.path],
                [f"path_coordinate={frame.path_coordinate:.10f}" for frame in result.path],
            )
        )
    if result.saddle_trajectory:
        references.extend(
            _trajectory_pair(
                store,
                job_id,
                "saddle-trajectory",
                "saddle_optimization_trajectory",
                [frame.model_dump(mode="json") for frame in result.saddle_trajectory],
                [frame.structure for frame in result.saddle_trajectory],
                [
                    f"stage={frame.stage} step={frame.step} "
                    f"energy_ev={frame.energy_ev:.12f}"
                    for frame in result.saddle_trajectory
                ],
            )
        )
    if result.irc_frames:
        references.extend(
            _trajectory_pair(
                store,
                job_id,
                "irc-trajectory",
                "irc_trajectory",
                [frame.model_dump(mode="json") for frame in result.irc_frames],
                [frame.structure for frame in result.irc_frames],
                [
                    f"irc_coordinate={frame.path_coordinate:.10f} "
                    f"energy_ev={frame.energy_ev:.12f}"
                    for frame in result.irc_frames
                ],
            )
        )
    for endpoint in (
        result.irc_endpoint_a_relaxation,
        result.irc_endpoint_b_relaxation,
    ):
        if endpoint is None:
            continue
        artifact_id = endpoint.endpoint.replace("_", "-") + "-relaxation"
        structures = [frame.structure for frame in endpoint.trajectory]
        comments = [
            f"step={frame.step} energy_ev={frame.energy_ev:.12f}"
            for frame in endpoint.trajectory
        ]
        if not structures:
            structures = [endpoint.relaxed_endpoint_structure]
            comments = [endpoint.final_structure_label]
        references.extend(
            _trajectory_pair(
                store,
                job_id,
                artifact_id,
                "irc_endpoint_relaxation",
                endpoint.model_dump(mode="json"),
                structures,
                comments,
            )
        )
    return references


def _trajectory_pair(
    store: FileArtifactStore,
    job_id: UUID,
    artifact_id: str,
    kind: str,
    json_payload: Any,
    structures: list[AtomicStructure],
    comments: list[str],
) -> list[ArtifactReference]:
    frame_count = len(structures)
    return [
        store.write_json(
            job_id,
            artifact_id + "-json",
            kind,
            json_payload,
            frame_count=frame_count,
        ),
        store.write_text(
            job_id,
            artifact_id + "-xyz",
            kind,
            artifact_id + ".xyz",
            _multi_xyz(structures, comments),
            media_type="chemical/x-xyz",
            frame_count=frame_count,
        ),
    ]


def _multi_xyz(
    structures: Iterable[AtomicStructure],
    comments: Iterable[str],
) -> str:
    blocks = []
    for structure, comment in zip(structures, comments, strict=True):
        lines = [str(len(structure.atoms)), comment]
        lines.extend(
            f"{atom.symbol:<2} {atom.position[0]: .12f} "
            f"{atom.position[1]: .12f} {atom.position[2]: .12f}"
            for atom in structure.atoms
        )
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + ("\n" if blocks else "")

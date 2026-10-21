"""Gaussian and content-verified artifact download routes."""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response

from app.contracts.jobs import CalculationResult, TransitionStateResult
from app.formats.gaussian import gaussian_stem, render_gaussian_input


router = APIRouter()


@router.get("/{job_id}/gaussian")
async def download_gaussian_input(job_id: UUID, request: Request) -> Response:
    job = await request.app.state.job_repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if isinstance(job.result, TransitionStateResult):
        content = job.result.gaussian_input
        filename = f"dmf-ts-{job.id}.gjf"
    elif isinstance(job.result, CalculationResult):
        parameters = job.request.parameters
        structure = job.result.final_structure
        stem = gaussian_stem(Path(structure.name or f"uma-{job.id}"))
        content = render_gaussian_input(
            structure,
            job.result,
            stem=stem,
            route=parameters.gaussian_route,
            nproc=parameters.gaussian_nproc,
            memory=parameters.gaussian_memory,
        )
        filename = f"{stem}.gjf"
    else:
        raise HTTPException(
            status_code=409,
            detail="Gaussian input is not available for this job",
        )
    return Response(
        content=content,
        media_type="chemical/x-gaussian-input",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{job_id}/artifacts/{artifact_id}")
async def download_job_artifact(
    job_id: UUID,
    artifact_id: str,
    request: Request,
) -> Response:
    job = await request.app.state.job_repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    reference = next(
        (
            artifact
            for artifact in job.artifacts
            if artifact.artifact_id == artifact_id
        ),
        None,
    )
    if reference is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    try:
        content = request.app.state.artifact_store.read_verified(reference)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact file is missing") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    filename = Path(reference.relative_path).name
    return Response(
        content=content,
        media_type=reference.media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

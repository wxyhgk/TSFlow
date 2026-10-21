"""Conversion between SQLite rows and public job contracts."""

import json
import sqlite3
from typing import Any

from app.contracts.jobs import Job, JobSummary, job_summary_from_job
from app.infrastructure.artifacts.hydrate import compact_result_for_storage
from app.infrastructure.artifacts.store import FileArtifactStore


FULL_JOB_SELECT = (
    "SELECT id, status, request_json, created_at, updated_at, service_managed, "
    "recovery_count, idempotency_key, canonical_request_hash, execution_json, error, "
    "result_json, progress_json, artifacts_json, "
    "artifact_error, result_summary_json, result_external_fields_json, "
    "result_storage_version FROM jobs"
)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def job_to_row(
    job: Job,
    artifact_store: FileArtifactStore | None = None,
) -> dict[str, Any]:
    summary = job_summary_from_job(job).result_summary
    result_payload = None
    external_fields: list[str] = []
    if job.result is not None:
        result_payload, external_fields = compact_result_for_storage(
            job.result,
            job.artifacts,
            artifact_store,
        )
    return {
        "id": str(job.id),
        "status": job.status.value,
        "request_json": _compact_json(job.request.model_dump(mode="json")),
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "service_managed": int(job.service_managed),
        "recovery_count": job.recovery_count,
        "idempotency_key": job.idempotency_key,
        "canonical_request_hash": job.canonical_request_hash,
        "execution_json": (
            _compact_json(job.execution_envelope.model_dump(mode="json"))
            if job.execution_envelope is not None
            else None
        ),
        "error": job.error,
        "result_json": (
            _compact_json(result_payload) if result_payload is not None else None
        ),
        "progress_json": (
            _compact_json(job.progress.model_dump(mode="json"))
            if job.progress is not None
            else None
        ),
        "artifacts_json": _compact_json(
            [artifact.model_dump(mode="json") for artifact in job.artifacts]
        ),
        "artifact_error": job.artifact_error,
        "result_summary_json": (
            _compact_json(summary.model_dump(mode="json"))
            if summary is not None
            else None
        ),
        "result_external_fields_json": _compact_json(external_fields),
        "result_storage_version": 2,
    }


def job_from_row(row: sqlite3.Row) -> Job:
    return Job.model_validate(
        {
            "id": row["id"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "service_managed": bool(row["service_managed"]),
            "recovery_count": row["recovery_count"],
            "idempotency_key": row["idempotency_key"],
            "canonical_request_hash": row["canonical_request_hash"],
            "execution_envelope": (
                json.loads(row["execution_json"])
                if row["execution_json"]
                else None
            ),
            "error": row["error"],
            "result": (
                json.loads(row["result_json"]) if row["result_json"] else None
            ),
            "progress": (
                json.loads(row["progress_json"]) if row["progress_json"] else None
            ),
            "artifacts": json.loads(row["artifacts_json"]),
            "artifact_error": row["artifact_error"],
        }
    )


def summary_from_row(row: sqlite3.Row) -> JobSummary:
    return JobSummary.model_validate(
        {
            "id": row["id"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
            "progress": (
                json.loads(row["progress_json"]) if row["progress_json"] else None
            ),
            "result_summary": (
                json.loads(row["result_summary_json"])
                if row["result_summary_json"]
                else None
            ),
            "artifacts": json.loads(row["artifacts_json"]),
            "artifact_error": row["artifact_error"],
        }
    )

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from app.contracts.artifacts import ArtifactReference


class FileArtifactStore:
    """Content-verified, atomic filesystem artifact storage."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        job_id: UUID,
        artifact_id: str,
        kind: str,
        payload: Any,
        *,
        frame_count: int | None = None,
    ) -> ArtifactReference:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.write_bytes(
            job_id,
            artifact_id,
            kind,
            f"{artifact_id}.json",
            data,
            media_type="application/json",
            frame_count=frame_count,
        )

    def write_text(
        self,
        job_id: UUID,
        artifact_id: str,
        kind: str,
        filename: str,
        text: str,
        *,
        media_type: str,
        frame_count: int | None = None,
    ) -> ArtifactReference:
        return self.write_bytes(
            job_id,
            artifact_id,
            kind,
            filename,
            text.encode("utf-8"),
            media_type=media_type,
            frame_count=frame_count,
        )

    def write_bytes(
        self,
        job_id: UUID,
        artifact_id: str,
        kind: str,
        filename: str,
        data: bytes,
        *,
        media_type: str,
        frame_count: int | None = None,
    ) -> ArtifactReference:
        safe_filename = _safe_filename(filename)
        job_directory = self.root / str(job_id)
        job_directory.mkdir(parents=True, exist_ok=True)
        target = job_directory / safe_filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        digest = hashlib.sha256(data).hexdigest()
        return ArtifactReference(
            artifact_id=artifact_id,
            kind=kind,
            media_type=media_type,
            relative_path=f"{job_id}/{safe_filename}",
            sha256=digest,
            size_bytes=len(data),
            created_at=datetime.now(UTC),
            frame_count=frame_count,
        )

    def read_verified(self, reference: ArtifactReference) -> bytes:
        target = self._resolve(reference.relative_path)
        data = target.read_bytes()
        if len(data) != reference.size_bytes:
            raise ValueError("artifact size does not match its index")
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError("artifact checksum does not match its index")
        return data

    def _resolve(self, relative_path: str) -> Path:
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("artifact path must remain inside the artifact store")
        target = (self.root / Path(*pure.parts)).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("artifact path escapes the artifact store")
        return target


def _safe_filename(filename: str) -> str:
    pure = PurePosixPath(filename)
    if len(pure.parts) != 1 or pure.name in {"", ".", ".."}:
        raise ValueError("artifact filename must be a single safe path component")
    return pure.name

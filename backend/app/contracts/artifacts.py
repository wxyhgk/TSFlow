from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ArtifactReference(BaseModel):
    artifact_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: str
    media_type: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    created_at: datetime
    schema_version: int = Field(default=1, ge=1)
    frame_count: int | None = Field(default=None, ge=0)


class ArtifactSelector(BaseModel):
    """Opaque client selector resolved against a persisted source action."""

    source_action_id: UUID
    artifact_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

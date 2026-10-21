# Backend Application Layout

The FastAPI backend follows a layered structure so HTTP, orchestration,
scientific integrations, and storage can evolve independently.

## Primary packages

- `contracts/`: stable Pydantic request, result, job, and artifact models.
- `api/`: HTTP routes and transport-level validation.
- `application/`: use-case orchestration and job lifecycle coordination.
- `adapters/`: UMA and third-party scientific library integrations.
- `formats/`: XYZ, SDF, and Gaussian parsing/serialization.
- `infrastructure/`: SQLite, artifact files, and other external resources.
- `core/`: process-wide configuration and bootstrap primitives.

## Current ownership

| Concern | Current owner | Long-term owner |
| --- | --- | --- |
| HTTP routes and request status | `api/` | `api/` |
| Job use cases and Worker lifecycle | `application/` | `application/` |
| TS request-to-workflow mapping | `application/transition_states/` | `application/transition_states/` |
| Job-level calculation dispatch | `services/platform_executor.py` | `application/calculations/` |
| UMA/xTB/Gaussian calls | `adapters/`, `calculators/` | `adapters/`, `calculators/` |
| UMA geomeTRIC optimization bridge | `adapters/uma/geometric.py` | `adapters/uma/geometric.py` |
| XYZ/SDF/Gaussian serialization | `formats/xyz.py`, `formats/sdf.py`, `formats/gaussian.py` | `formats/` |
| SQLite and artifact storage | `infrastructure/` | `infrastructure/` |
| Model availability and checkpoint discovery | `calculators/models.py` | `calculators/models.py` |

The application layer now owns TS request-to-workflow mapping. The old
`services/transition_state_search/` modules remain only as compatibility
exports; existing imports remain stable until a dedicated compatibility removal
is reviewed.

## Migration-only packages

- `services/`: old flat modules being moved into the layers above.
- `schemas/`: compatibility exports for `contracts/`.
- `repositories/` and `artifacts/`: compatibility exports for
  `infrastructure/`.
- `compat/`: explicit adapters for APIs that cannot migrate atomically.

Case-specific construction, scan, TS refinement, IRC, and animation scripts
are outside this application package and live in the ignored
`calculations/scripts/` directory.

Each primary package contains its own README with its dependency boundary. New
code should use the primary package paths, while compatibility paths remain only
long enough for callers to migrate safely.

# Artifacts

This package owns durable files produced by calculations: JSON summaries,
multi-frame XYZ trajectories, endpoint structures, and integrity metadata.

## Responsibilities

- `store.py`: atomic writes, safe paths, size checks, and SHA-256 validation.
- `materialize.py`: converts result contracts into downloadable artifacts.
- `hydrate.py`: restores large result fields that were externalized from SQLite.

## Dependency rule

This package may depend on `app.contracts`, but must not import FastAPI routes or
calculation executors. Callers should use artifact references instead of knowing
the on-disk layout.

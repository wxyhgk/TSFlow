# Persistence

This package contains implementations that store and retrieve job state.

## Responsibilities

- `protocol.py`: the repository interface used by application workers.
- `memory.py`: active-job cache and optional legacy JSON snapshots.
- `sqlite/`: durable SQLite/WAL storage, summaries, migration, and hydration.

## Dependency rule

Application code should type against `JobRepository`. Persistence may use
contracts and artifact storage, but it must not depend on API routes or chemical
calculation executors.

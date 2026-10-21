"""SQLite schema creation and forward-compatible column migrations."""

import sqlite3


def ensure_schema(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                service_managed INTEGER NOT NULL DEFAULT 0,
                recovery_count INTEGER NOT NULL DEFAULT 0,
                idempotency_key TEXT,
                canonical_request_hash TEXT,
                execution_json TEXT,
                error TEXT,
                result_json TEXT,
                progress_json TEXT,
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                artifact_error TEXT,
                result_summary_json TEXT,
                result_external_fields_json TEXT NOT NULL DEFAULT '[]',
                result_storage_version INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
        }
        if "result_summary_json" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN result_summary_json TEXT")
        if "result_external_fields_json" not in columns:
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN result_external_fields_json "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        if "result_storage_version" not in columns:
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN result_storage_version "
                "INTEGER NOT NULL DEFAULT 1"
            )
        if "idempotency_key" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN idempotency_key TEXT")
        if "canonical_request_hash" not in columns:
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN canonical_request_hash TEXT"
            )
        if "execution_json" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN execution_json TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency_key "
            "ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )

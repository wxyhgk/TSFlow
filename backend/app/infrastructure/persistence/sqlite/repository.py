"""Durable local SQLite repository using the shared job lifecycle."""
import sqlite3
from pathlib import Path
from app.infrastructure.persistence.memory import InMemoryJobRepository
from app.infrastructure.persistence.sqlite.schema import ensure_schema
from app.infrastructure.persistence.sqlite.codec import FULL_JOB_SELECT, job_from_row, job_to_row

class SQLiteJobRepository(InMemoryJobRepository):
    def __init__(self, database_path, *, artifact_store=None, legacy_archive_dir=None):
        super().__init__(artifact_store=artifact_store)
        path=Path(database_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self._connection=sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory=sqlite3.Row
        self._connection.execute('PRAGMA journal_mode=WAL')
        ensure_schema(self._connection)
        for row in self._connection.execute(FULL_JOB_SELECT):
            job=job_from_row(row)
            self._jobs[job.id]=job
        if legacy_archive_dir:
            from app.contracts.jobs import Job
            for file in Path(legacy_archive_dir).glob('*.json'):
                job=Job.model_validate_json(file.read_text())
                if job.id not in self._jobs: self._save(job)

    def _save(self, job):
        # Keep complete result JSON until large-result hydration is restored.
        row=job_to_row(job, None)
        columns=list(row)
        updates=', '.join(f'{c}=excluded.{c}' for c in columns if c!='id')
        sql=f"INSERT INTO jobs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) ON CONFLICT(id) DO UPDATE SET {updates}"
        with self._connection:
            self._connection.execute(sql,[row[c] for c in columns])
        super()._save(job)

    async def close(self): self._connection.close()

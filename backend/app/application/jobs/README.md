# Job Application Services

This package owns job scheduling and execution lifecycle orchestration.

- `local_worker.py`: single-thread workstation worker.
- `process_worker.py`: recyclable child-process worker and queue lifecycle.
- `child_process.py`: isolated UMA process entry point.
- `messages.py`: structured progress/result message decoding.

Workers depend on repository protocols and scientific executor adapters. They do
not implement SQL, HTTP routes, or numerical chemistry algorithms.

Before invoking an executor, each worker calls `repository.start()` and requires
the returned job to have status `running`. A queued job cancelled before the
worker consumes it therefore exits without model lookup or process/thread
creation.

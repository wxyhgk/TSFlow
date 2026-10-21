# UMA Adapter

This package isolates FAIR Chemistry/ASE integration from application logic.

- `executor.py`: stable `UmaExecutor` facade.
- `calculation.py`: single-point and geometry-optimization workflows.
- `frequency.py`: analytic or finite-difference Hessian workflow.
- `structures.py`: conversion between public contracts and ASE atoms.

The adapter produces backend contracts and must not know about FastAPI,
repositories, SSE, or UI state.

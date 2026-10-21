# TSFlow

> **TSFlow: Verifiable Transition-State Search for Large Molecules**

[中文 README](README.md) · [MIT License](LICENSE)

![TSFlow workflow](assets/tsflow-workflow.svg)

*Workflow schematic: endpoint preparation → saddle-point screening → bidirectional IRC validation. Molecular structures and the energy profile are illustrative.*

TSFlow is a backend and workflow library for auditable transition-state search in large-molecule reactions. It turns endpoint preparation, path generation, saddle refinement, Hessian screening, and bidirectional IRC into a traceable workflow with replaceable calculation providers.

UMA is the current primary engine for high-frequency energy, force, optimization, path, and Hessian work. The provider boundary also supports xTB and Gaussian integration, so fast screening and higher-level review can be staged without rewriting the workflow.

## Highlights

- **Designed for large molecules:** endpoint optimization, path search, saddle refinement, frequency screening, and IRC are separate stages with persisted intermediate artifacts.
- **Replaceable algorithms:** FB-ENM and Direct MaxFlux are exposed through workflow contracts; future path methods can be added without changing the HTTP task model.
- **Verifiable results:** the workflow records convergence, Hessian quality, imaginary-mode checks, IRC directionality, endpoint relaxation, and heavy-atom topology matching.
- **API-first integration:** the algorithm package is independent of FastAPI and databases; the backend provides routes, workers, persistence, cancellation, and provider selection.

## Workflow

```text
reactants / products / reaction sites
                |
                v
        endpoint pre-optimization
                |
                v
      FB-ENM / Direct MaxFlux path
                |
                v
          saddle refinement
                |
                v
       Hessian and frequency screen
                |
                v
          bidirectional IRC
                |
                v
     endpoint and reaction-coordinate validation
```

When a transition-state guess already exists, path generation can be skipped:

```text
R / TS guess / P -> UMA saddle refinement -> Hessian screen -> bidirectional IRC
```

## Scientific scope

Task status is separate from scientific validation. A `succeeded` task means execution completed; it does not by itself establish a transition state. A validated candidate requires a converged stationary saddle, a full-system Hessian with one significant imaginary mode, a compatible reaction coordinate, and bidirectional IRC connection to the expected endpoints. Endpoint relaxation and topology checks should also pass.

Electronic barriers must be reported separately from experimental activation free energies, which include temperature, solvent, and entropy effects. The project does not claim a universal success rate or cross-method benchmark; those require controlled datasets, hardware, convergence settings, and reference methods.

## Engines

| Engine | Current role |
| --- | --- |
| UMA | Primary large-system search, optimization, Hessian, and IRC engine |
| xTB | Fast endpoint pre-optimization and low-cost screening provider |
| Gaussian | Replaceable higher-level review adapter for selected calculations |
| ORCA | External research review; not yet a general backend provider |

## Repository layout

```text
packages/uma-workflows/     reusable path, saddle, frequency, and IRC algorithms
backend/                    FastAPI routes, workers, persistence, and adapters
docs/                       public architecture, installation, and workflow docs
examples/                   generic examples without private research structures
scripts/                    environment, startup, checks, and source export
calculations/               local research calculations; excluded from releases
frontend/                   local workbench; excluded from the first source release
```

The Python distribution is named `ts-flow`. Its import namespace remains `uma_workflows`, and its source directory remains `packages/uma-workflows/` for compatibility. The backend package remains `uma-platform-api`; the public project name is TSFlow.

## Quick start

Python 3.11+ and `uv` are required:

```bash
uv venv --python 3.13 .venv-uma
scripts/install-backend-env.sh
cp backend/.env.example backend/.env
scripts/run-backend.sh
```

The API listens on `http://127.0.0.1:8010` by default, with OpenAPI docs at `/docs`. Model weights are obtained separately under their upstream terms and are not distributed with this repository.

Install the standalone workflow library from GitHub:

```bash
pip install "ts-flow @ git+https://github.com/wxyhgk/TSFlow.git#subdirectory=packages/uma-workflows"
pip install "ts-flow[all] @ git+https://github.com/wxyhgk/TSFlow.git#subdirectory=packages/uma-workflows"
```

After a PyPI release, the short commands will be `pip install ts-flow` and `pip install "ts-flow[all]"`. This project has not yet been released to PyPI. The name `tsflow` on PyPI belongs to a different project.

Run the focused smoke tests with:

```bash
scripts/test-backend-smoke.sh
```

See [AI.md](AI.md) for repository boundaries, validation rules, and instructions for code agents.

## Current limitations

- A general HTTP endpoint that plans and completes an automatic search from only a molecule and reaction sites is still being productized.
- ORCA is not yet a general backend provider.
- The frontend remains a local workbench and is not part of the first public source export.
- Cross-method speed, success-rate, and barrier-error claims require a separate controlled benchmark.

## License

TSFlow source code is released under the [MIT License](LICENSE). UMA model weights and upstream assets are governed by their own license and usage policy.

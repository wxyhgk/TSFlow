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

### Using TSFlow with an AI planner

TSFlow can be used as the calculation backend for an external LLM or agent. The AI turns a natural-language reaction description into an explicit reaction definition; TSFlow executes the calculation and returns auditable validation fields:

```text
user describes the reaction
    -> AI identifies structures, atom mapping, and reaction sites
    -> a human reviews the generated plan
    -> POST /api/v1/jobs
    -> UMA / xTB / Gaussian executes the workflow
    -> Hessian, imaginary-mode, and bidirectional IRC validation
```

The planner should provide at least:

- reactant and product 3D structures with identical atom ordering;
- total charge and spin multiplicity;
- breaking, forming, and coordination pairs using zero-based atom indices;
- a calculation engine and model such as `uma-s-1p2p1`;
- whether a TS guess is supplied and whether bidirectional IRC should run;
- fallback paths or parameters when a candidate fails, rather than labeling an unvalidated geometry as a TS.

A planner can first produce an **intermediate reaction plan** like this. The file paths below are not backend API fields; the caller must read them and expand them into complete `AtomicStructure` objects before submission:

```json
{
  "calculation": "transition_state_search",
  "model_name": "uma-s-1p2p1",
  "structure": "reactant.json",
  "product_structure": "product.json",
  "transition_state_guess": "ts_guess.json",
  "reaction_coordinates": [
    {"atom_i": 23, "atom_j": 47, "kind": "breaking", "label": "C-Cl"},
    {"atom_i": 23, "atom_j": 81, "kind": "forming", "label": "C-Li"},
    {"atom_i": 81, "atom_j": 14, "kind": "coordination", "label": "Li-O"}
  ],
  "parameters": {
    "irc_follow": true,
    "compute_frequencies": true,
    "path_initializer_id": "fbenm.correlated",
    "path_optimizer_id": "dmf.direct_maxflux"
  }
}
```

A useful system-prompt constraint for the planner is:

```text
You are a TSFlow reaction planner. Use only atom indices present in the input structures; never invent indices.
Return reactant, product, breaking, forming, coordination, charge, spin, and calculation parameters.
If the product, atom mapping, or reaction sites are unclear, ask for clarification instead of guessing.
Never call a job complete or a TS validated; validation requires Hessian, imaginary-mode, and IRC results.
```

After expansion into the API's `AtomicStructure` objects, submit the completed payload with:

```bash
curl -X POST http://127.0.0.1:8010/api/v1/jobs \
  -H 'Content-Type: application/json' \
  --data @api-payload.json
```

The current release does not include a general LLM API, API-key management, or an endpoint that completes a search from only a molecule and reaction sites. Any OpenAI, Gemini, Claude, or local-model planner must validate structures, atom mapping, and reaction sites before submission. A `succeeded` job still does not prove a scientifically validated transition state. `AI.md` is a repository guide for coding agents, not a runtime AI endpoint.

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

The Python distribution is named `tsflow-chem`. Its import namespace remains `uma_workflows`, and its source directory remains `packages/uma-workflows/` for compatibility. The backend package remains `uma-platform-api`; the public project name is TSFlow.

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
pip install "tsflow-chem @ git+https://github.com/wxyhgk/TSFlow.git#subdirectory=packages/uma-workflows"
pip install "tsflow-chem[all] @ git+https://github.com/wxyhgk/TSFlow.git#subdirectory=packages/uma-workflows"
```

After a PyPI release, the short commands will be `pip install tsflow-chem` and `pip install "tsflow-chem[all]"`. This project has not yet been released to PyPI. The names `tsflow` and `ts-flow` on PyPI belong to a different project.

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

# TSFlow workflow engine (`ts-flow`)

`ts-flow` is the reusable TSFlow computational chemistry engine shared by
the FastAPI service, command-line tools, benchmarks, and research studies.
The Python import namespace remains `uma_workflows`, and the source directory
remains `packages/uma-workflows/`. TSFlow is the public project name.

The package must not import FastAPI, backend Pydantic schemas, repositories, or
project-specific data paths.

Currently extracted:

- UMA ASE calculator construction, caching, and CPU/MPS selection;
- reusable Sella minimum and first-order saddle refinement;
- FAIR-Chem analytic/autodiff and finite-difference Hessian generation;
- geomeTRIC Eckart-frame vibrational analysis;
- bidirectional Sella IRC initialization and direction following;
- IRC physical validity checks;
- IRC endpoint heavy-atom topology matching;
- FB-ENM directional initial-path generation and candidate ranking;
- cancellation, device synchronization, and non-periodic graph-cell helpers.

Python API:

```python
from uma_workflows.irc import (
    follow_sella_irc_direction,
    match_irc_endpoint_topologies,
    prepare_sella_irc,
    require_mode_gradient_bracket,
)
from uma_workflows.calculators import UmaCalculatorFactory
from uma_workflows.frequencies import (
    analyze_vibrational_hessian,
    compute_analytic_hessian,
    compute_numerical_hessian,
)
from uma_workflows.optimization import refine_first_order_saddle
```

The caller supplies checkpoint, reference, input, and output paths. The package
does not assume this repository's `model-assets` or `data` directories.

`numerical_frequency_analysis()` remains available as a compatibility
composition. New workflows can cache the result of
`compute_numerical_hessian()` and call `analyze_vibrational_hessian()` multiple
times without repeating UMA force evaluations.

For UMA, `hessian_method="auto"` first requests FAIR-Chem's built-in
second-autograd-derivative Hessian on CPU and falls back to central finite
differences only if that property is unavailable. Full-system vibrational
analysis delegates Eckart-frame translation/rotation removal, mass weighting,
and normal-mode construction to geomeTRIC. Partial active-region Hessians stay
unprojected because motion relative to the frozen environment is not a global
rigid-body zero mode.

FAIR-Chem 2.21 propagates the graph-retention flag needed for energy-to-force
differentiation into force-to-Hessian differentiation, retaining an unnecessary
third-derivative graph during inference. The UMA calculator applies an
inference-only compatibility patch. Systems up to 30 atoms use the faster
vectorized Hessian; larger systems use the lower-memory row-wise implementation.
The threshold selects an implementation and is not an atom-count rejection.
Likewise, `frequency_max_atoms` limits only central finite-difference work. A
full analytic Hessian remains eligible above that limit; if it fails, the
workflow reports that numerical fallback was disallowed instead of launching an
unexpectedly large finite-difference calculation.

The workflow records the raw Hessian antisymmetry residual before
symmetrization and can reject results above a configurable tolerance. IRC is
initialized from the already screened primary imaginary mode; it does not
silently diagonalize the raw Hessian again and choose a potentially different
lowest mode.

Install from GitHub until the first PyPI release:

```bash
pip install "ts-flow @ git+https://github.com/wxyhgk/TSFlow.git#subdirectory=packages/uma-workflows"
pip install "ts-flow[all] @ git+https://github.com/wxyhgk/TSFlow.git#subdirectory=packages/uma-workflows"
```

After publication to PyPI, install only the capabilities a project needs with
`pip install ts-flow`, or select extras such as `ts-flow[frequencies]`,
`ts-flow[dmf]`, `ts-flow[sella]`, `ts-flow[uma]`, and `ts-flow[all]`.
The PyPI name `tsflow` belongs to another project.

The base package contains NumPy/ASE geometry, geomeTRIC normal-mode analysis,
and validation utilities. Public packages use lazy exports, so
`import uma_workflows` does not import Sella, PyTorch, FairChem, or OmegaConf.

Long-running APIs accept any object implementing the minimal
`CancellationToken` protocol (`is_set() -> bool`); `threading.Event` remains
compatible but is no longer required.

Optimization pipelines are not restricted to MPS followed by CPU:

```python
from uma_workflows.optimization import (
    OptimizationStage,
    refine_saddle_stages,
)

result = refine_saddle_stages(
    atoms,
    path_tangent=tangent,
    stages=[
        OptimizationStage(
            name="uma_mps",
            device="mps",
            calculator=uma_mps,
            fmax=0.05,
            max_steps=100,
        ),
        OptimizationStage(
            name="uma_cpu",
            device="cpu",
            calculator=uma_cpu,
            fmax=0.01,
            max_steps=50,
        ),
    ],
    cancel_event=cancel_token,
)
```

The same stage model can represent CUDA, another ML potential, or a future DFT
calculator. Legacy fixed `mps_calculator`/`cpu_calculator` functions remain as
compatibility adapters.

Transition-state algorithms use plugin-oriented contracts:

```python
from uma_workflows.transition_states import (
    DirectMaxFluxPathOptimizer,
    PathInitializer,
    PathOptimizer,
    TransitionStateProblem,
    TransitionStateValidator,
)
```

Path biasing, collision checks, unbiased energy analysis, active-region
selection, and bond-change assessment are public schema-independent algorithms.
FastAPI/Pydantic conversion remains in the backend adapter.

FB-ENM is an optional path initializer:

```python
from uma_workflows.transition_states import FbenmConfig, FbenmPathInitializer

initializer = FbenmPathInitializer(
    FbenmConfig(nmove=6, ensemble_size=4),
)
result = initializer.generate(
    problem,
    calculator=calculator,
    cancel_token=cancel_token,
)
```

The initializer returns scored `PathCandidate` objects; selection is a separate
operation via `select_best_candidate()`. This separation lets other
initializers, such as IDPP or imported paths, share the same ranking policy.

Direct MaxFlux implements the matching `PathOptimizer` contract. Its result
contains a uniformly sampled `ReactionPath` with sampled tangents. The result
is JSON-serializable through `to_dict()` and does not retain a live PyDMF,
calculator, or ASE runtime object. A candidate without PyDMF coefficients is
accepted by fitting the optimizer from its public sampled path.

Post-path processing is also composable:

- `refine_path_saddle()` prepares an optional constrained scan, active region,
  reaction-corridor seed, and mandatory unbiased CPU Sella refinement whenever
  that bias was used;
- `screen_transition_state()` performs an unbiased CPU force check, optional
  partial Hessian, imaginary-mode count, Hessian-quality, and bond-progress
  checks.

A partial Hessian is explicitly provisional and cannot set the definitive
`passes_screening` flag; final first-order-saddle confirmation requires the
full-system Hessian.

When forming/breaking coordinates are supplied, the primary imaginary mode is
projected onto their bond-length gradients. Because an eigenvector's global
sign is arbitrary, both orientations are considered. Final screening requires
stationarity, a full Hessian, exactly one significant imaginary mode, and an
acceptable Hessian-quality residual. Mode overlap, endpoint reaction-definition
consistency, and endpoint-relative geometry progress are reported separately as
mechanism findings. They rank or explain candidates but do not prevent a valid
index-1 saddle from entering bidirectional IRC discovery.

Both functions consume `TransitionStateProblem` and plain ASE objects. Progress
is reported through callbacks rather than backend job schemas.

The complete workflow is also a public package API:

```python
from uma_workflows.transition_states import (
    TransitionSearchConfig,
    run_transition_state_search,
)

result = run_transition_state_search(
    reactant,
    product,
    reaction_coordinates=coordinates,
    config=TransitionSearchConfig(),
    device="cpu",
    calculator=calculator,
    cpu_calculator=calculator,
    cancel_event=cancel_token,
    progress_callback=on_progress,
)
```

It returns typed endpoint, path, saddle, screening, and warning results without
importing FastAPI or backend Pydantic schemas.

The DMF portion is also available as the API-independent
`optimize_path_candidate(...)` helper. It accepts one typed `PathCandidate`, a
`TransitionStateProblem`, a `PathConfig`, calculators, cancellation and an
optional `AlgorithmRegistry`. The macro workflow calls this same helper for
each generated candidate, so standalone agent actions and the macro share
identical path bias, optimizer construction, unbiased CPU energy sampling,
interior-maximum detection and candidate scoring.

The API-independent `probe_path_curvature(...)` helper performs a bounded,
cheap seed screen around the unbiased path maximum. It evaluates the center and
two tangent-displaced force fields for each selected frame and reports
directional Cartesian curvature, path-gradient projection and maximum force.
Negative directional curvature is deliberately not reported as an imaginary
normal mode or an index-one saddle.

`relaxed_collective_coordinate_scan(...)` generalizes the former single-bond
scan to `q = Σ cₖ r(iₖ,jₖ)`. A harmonic restraint is applied to the complete
linear coordinate while all orthogonal degrees of freedom relax. Reported
energies, coordinate derivatives and orthogonal forces are recomputed with the
unrestrained calculator, and only converged three-point interior maxima are
marked as saddle seeds.

When `transition_state_guess` is supplied, the workflow does not run endpoint
preoptimization, FB-ENM, or DMF. It builds a three-frame reactant/guess/product
reference energy profile, refines the supplied structure directly on the UMA
surface, then performs fresh Hessian screening and optional IRC. The initial
RS-P-RFO Hessian follows `ScreeningConfig.hessian_method`: `auto` tries the
analytic UMA Hessian first and falls back to finite differences only within the
configured finite-difference atom limit.

Set `IrcConfig(enabled=True)` on the workflow configuration to request the
bidirectional CPU geomeTRIC IRC stage. geomeTRIC follows the
Gonzalez-Schlegel mass-weighted internal-coordinate algorithm and receives the
validated UMA Cartesian Hessian directly. Set `engine="sella"` only to
reproduce the legacy implementation. IRC only starts after the candidate passes
the full-system Hessian, one-imaginary-mode, reaction-coordinate overlap, and
stationarity gates. The workflow then checks:

- projected gradients on opposite imaginary-mode displacements;
- opposite forward/reverse first-step projections;
- distinct IRC endpoint geometries;
- heavy-atom topology matching against the expected reactant and product.

An IRC validation failure is returned as a typed stage result and warning; it
does not discard the transition-state candidate or its Hessian.

The default saddle pipeline keeps Sella as the path-tangent seed and then runs
an unbiased geomeTRIC RS-P-RFO final refinement from a full analytic UMA
Hessian. Set `SaddleConfig(final_refinement_engine="sella")` to reproduce the
legacy behavior. The final candidate is screened with a newly computed Hessian;
the initialization Hessian is never reused as proof of a first-order saddle.

# AI action application layer

This package exposes a small, typed action vocabulary for API and agent
clients.  The first version deliberately maps actions onto the existing Job
execution system; it does not duplicate worker, persistence, or scientific
validation logic.

Rules:

- action contracts are discriminated unions, never free-form dictionaries;
- capabilities list only actions that are actually executable;
- capability discovery includes each action's JSON schema, prerequisites,
  possible successor actions, and current Worker/model availability;
- action parameters control search/execution, while scientific validation is
  derived by the server;
- `validation_profiles.py` owns immutable, versioned validation thresholds;
  clients may select a published profile but cannot override its criteria;
- the bridge remains removable when independent Hessian, saddle, and IRC
  execution envelopes are introduced.

`guidance.py` and `ts_guidance.py` convert completed evidence into advisory
`next_actions`. Every advertised next action is a real `ActionType` accepted by
`POST /actions`; internal workflow hints and human review instructions are not
misrepresented as executable operations. Guidance never submits work by itself.

Path search is being decomposed into inspectable primitives. The first one,
`generate_path_candidates`, persists endpoint-bound candidate paths and their
native parameterization without running path optimization or saddle refinement.
`optimize_path_candidate` resolves one of those artifacts from the persisted
source-action manifest, validates its endpoint and PES bindings, then calls the
same single-candidate optimizer helper used by the macro workflow. Client input
never controls an artifact path.

`probe_path_curvature` consumes only a verified, converged optimized-path
artifact with an interior energy maximum. It screens a bounded set of nearby
frames with tangent finite-difference Hessian-vector products. Its output is a
non-stationary refinement seed, not a frequency result or validated saddle.

`run_collective_coordinate_scan` is the bounded fallback when tangent curvature
does not provide a seed. It accepts an explicit linear combination of bond
distances and optional explicit targets. If omitted, the server derives a
recorded default from breaking/forming reaction coordinates. Only an unbiased
interior maximum bracketed by converged relaxed points is promoted.

The runtime action surface is narrower than this contract vocabulary: the
mounted `/actions` route currently accepts the three direct calculation actions
and the supplied-guess transition-state workflow. These path primitives and
automatic path search are staged capabilities, not submit-able HTTP endpoints.

# Transition-State Search Compatibility Services

This package adapts backend request/result contracts to the reusable
`uma_workflows.transition_states` package.

It currently remains under `services` for compatibility. New orchestration will
move to `app.application.transition_states`, while scientific algorithms remain
in the reusable `packages/uma-workflows` project.

Do not add Hessian, IRC, DMF, or saddle-point mathematics here. This layer only
maps configuration, progress, structures, and results.

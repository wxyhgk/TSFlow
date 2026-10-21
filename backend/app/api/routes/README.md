# API Routes

FastAPI route modules translate HTTP requests into application operations.

- `health.py`: liveness/readiness information.
- `models.py`: available UMA model metadata.
- `jobs.py`: legacy aggregate job routes; this will be split by endpoint group.

Routes may depend on contracts and application services. They must not contain
SQL, filesystem persistence, or numerical chemistry algorithms.

## Composition

`create_router()` is the package entry point. It consumes `RouteDefinition`
objects and returns one aggregate `APIRouter`; `DEFAULT_ROUTE_DEFINITIONS`
contains the production modules. Tests or deployments can replace one module by
passing a different definition list, while `app.api.router.api_router` remains a
backward-compatible default export.

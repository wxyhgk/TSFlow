# Job Routes

The public `/jobs` API is divided by transport responsibility:

- `creation.py`: JSON job submission only. File upload routes (XYZ/SDF) and
  automatic transition-state path search are not mounted in the restored API.
- `queries.py`: list, summary, get, and cancel operations.
- `events.py`: legacy and compact SSE feeds.
- `downloads.py`: Gaussian inputs and verified artifacts.
- `shared.py`: model validation and enqueueing shared by creation routes.

`app.api.routes.jobs` combines these routers. The OpenAPI document is the
authoritative list of currently mounted URLs; do not document upload URLs until
their parsers and end-to-end execution path are restored.

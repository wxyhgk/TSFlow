"""Submit CLI calculations to the persistent local Worker without loading Torch."""
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from app.contracts.jobs import CalculationProgress, CalculationResult


def execute_via_service(request, service_url, progress_callback=None):
    base = service_url.rstrip('/')

    def call(path, payload=None, method=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(base + path, data=data, method=method,
                      headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=30) as response:
                return json.load(response)
        except URLError as exc:
            raise RuntimeError(
                f'UMA service request failed at {base}: {exc}. '
                'Start scripts/run-backend.sh; calculations are not retried automatically.'
            ) from exc

    health = call('/api/v1/health')
    if not health.get('worker_enabled'):
        raise RuntimeError('UMA service Worker is disabled; start scripts/run-backend.sh')
    job = call('/api/v1/jobs', request.model_dump(mode='json'))
    path = '/api/v1/jobs/' + job['id']
    last_progress = None
    try:
        while True:
            job = call(path)
            progress = job.get('progress')
            if progress and progress != last_progress:
                if progress_callback:
                    progress_callback(CalculationProgress.model_validate(progress))
                last_progress = progress
            if job['status'] == 'succeeded':
                return CalculationResult.model_validate(job['result'])
            if job['status'] in ('failed', 'cancelled'):
                raise RuntimeError(f"UMA job {job['id']} {job['status']}: {job.get('error')}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        call(path, method='DELETE')
        raise

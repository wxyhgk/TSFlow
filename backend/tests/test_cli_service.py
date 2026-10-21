import json
from unittest.mock import patch

import pytest
from app.contracts.jobs import JobCreate
from app.services.cli_service import execute_via_service


def test_cli_uses_service_and_preserves_requested_device():
    request = JobCreate(calculation='geometry_optimization', structure={
        'atoms':[{'symbol':'H','position':[0,0,0]}], 'charge':0,'spin':2},
        parameters={'optimization_device':'mps'})
    responses = iter([
        {'worker_enabled':True}, {'id':'job-id'},
        {'id':'job-id','status':'succeeded','result':{
            'energy_ev':-1,'forces_ev_per_a':[[0,0,0]],'max_force_ev_per_a':0,
            'final_structure':request.structure.model_dump(),'elapsed_seconds':.1}}])
    calls = []
    class Response:
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self):return json.dumps(next(responses)).encode()
    def open_(req, **_):
        calls.append(req)
        return Response()
    with patch('app.services.cli_service.urlopen', open_):
        result = execute_via_service(request,'http://127.0.0.1:8010')
    assert result.energy_ev == -1
    assert json.loads(calls[1].data)['parameters']['optimization_device']=='mps'
    assert calls[1].full_url.endswith('/api/v1/jobs')


def test_cli_fails_without_worker_instead_of_loading_local_model():
    class Response:
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self):return b'{"worker_enabled":false}'
    with patch('app.services.cli_service.urlopen',return_value=Response()):
        with pytest.raises(RuntimeError, match='disabled'):
            execute_via_service(None,'http://127.0.0.1:8010')

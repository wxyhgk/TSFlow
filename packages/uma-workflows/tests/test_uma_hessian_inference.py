from fairchem.core.models.uma import escn_md

from uma_workflows.calculators.uma import (
    _disable_inference_third_derivative_graph,
)


def test_hessian_inference_does_not_retain_third_derivative_graph(monkeypatch):
    received = {}

    def fake_compute_hessian(forces, pos, vmap=True, training=False):
        received.update(vmap=vmap, training=training)
        return "hessian"

    monkeypatch.setattr(escn_md, "compute_hessian", fake_compute_hessian)
    _disable_inference_third_derivative_graph()

    result = escn_md.compute_hessian(
        "forces",
        "positions",
        vmap=False,
        training=True,
    )

    assert result == "hessian"
    assert received == {"vmap": False, "training": False}

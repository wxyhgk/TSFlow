from threading import Event

import pytest
from ase import Atoms

from uma_workflows.geometry import set_nonperiodic_graph_cell
from uma_workflows.runtime import CalculationCancelled, raise_if_cancelled


def test_raise_if_cancelled_uses_public_exception() -> None:
    cancel_event = Event()
    cancel_event.set()
    with pytest.raises(CalculationCancelled):
        raise_if_cancelled(cancel_event)


def test_cancellation_accepts_protocol_without_threading_event() -> None:
    class Token:
        def is_set(self) -> bool:
            return True

    with pytest.raises(CalculationCancelled):
        raise_if_cancelled(Token())


def test_nonperiodic_graph_cell_is_shared_across_images() -> None:
    first = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    second = Atoms("H", positions=[[2.0, 3.0, 4.0]])

    set_nonperiodic_graph_cell((first, second), padding_a=10.0)

    assert first.cell.lengths() == pytest.approx([12.0, 13.0, 14.0])
    assert second.cell.lengths() == pytest.approx(first.cell.lengths())

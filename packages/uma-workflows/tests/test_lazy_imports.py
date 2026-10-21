import json
import subprocess
import sys


def test_base_import_does_not_load_optional_compute_stacks() -> None:
    program = """
import json
import sys
import uma_workflows
optional = [
    "scipy", "sella", "torch", "fairchem", "omegaconf", "dmf", "cyipopt"
]
print(json.dumps([name for name in optional if name in sys.modules]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_workflow_config_import_does_not_load_compute_stacks() -> None:
    program = """
import json
import sys
from uma_workflows.transition_states.workflow import TransitionSearchConfig
optional = [
    "sella", "torch", "fairchem", "omegaconf", "dmf", "cyipopt"
]
print(json.dumps([name for name in optional if name in sys.modules]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []

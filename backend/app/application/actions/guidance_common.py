from __future__ import annotations

from typing import Any, Literal

from app.contracts.actions import (
    ActionFinding,
    ActionGuidance,
    ActionType,
    NextActionDescriptor,
)
from app.contracts.job_records import Job, JobStatus


ScientificState = Literal[
    "pending",
    "completed",
    "candidate",
    "requires_refinement",
    "index1_saddle",
    "irc_connected_saddle",
    "validated_transition_state",
    "mechanism_mismatch",
    "inconclusive",
    "execution_failed",
    "cancelled",
]
FindingSeverity = Literal["info", "warning", "error"]


def guidance(
    job: Job,
    state: ScientificState,
    *,
    findings: list[ActionFinding] | None = None,
    next_actions: list[NextActionDescriptor] | None = None,
) -> ActionGuidance:
    return ActionGuidance(
        action_id=job.id,
        status=job.status,
        terminal=job.status
        in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED},
        scientific_state=state,
        findings=findings or [],
        next_actions=next_actions or [],
    )


def finding(
    code: str,
    severity: FindingSeverity,
    message: str,
    **values: Any,
) -> ActionFinding:
    return ActionFinding(code=code, severity=severity, message=message, values=values)


def next_action(
    action_type: ActionType,
    reason_code: str,
    description: str,
    **bindings: str,
) -> NextActionDescriptor:
    return NextActionDescriptor(
        action_type=action_type,
        reason_code=reason_code,
        description=description,
        input_bindings=bindings,
    )

from app.contracts import jobs, actions, action_results
jobs.Job.model_rebuild(_types_namespace={"ExecutionEnvelope": actions.ExecutionEnvelope, "ActionResult": action_results.ActionResult})

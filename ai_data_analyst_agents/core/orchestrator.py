from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ai_data_analyst_agents.core.messages import Message
from ai_data_analyst_agents.core.task_planner import Task


class Orchestrator:
    def __init__(self, agents: dict[str, Any], logger) -> None:
        self.agents = agents
        self.logger = logger

    def run(self, tasks: list[Task], ctx: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        memory = ctx.get("memory")
        for t in tasks:
            self.logger.info(f"[Orchestrator] {t.name}: {t.reason}")
            if memory is not None:
                memory.log(
                    Message(
                        sender="orchestrator",
                        role="agent",
                        content=f"Dispatching task '{t.name}'",
                        data={"task": t.name, "reason": t.reason, "status": "start"},
                    )
                )
            agent = self.agents.get(t.name)
            if agent is None:
                raise KeyError(f"No agent registered for task '{t.name}'.")
            actor_scope = memory.as_actor(t.name) if memory is not None else nullcontext()
            with actor_scope:
                try:
                    out = agent.run(ctx)
                except Exception as e:
                    if memory is not None:
                        memory.log(
                            Message(
                                sender="orchestrator",
                                role="agent",
                                content=f"Task '{t.name}' failed",
                                data={"task": t.name, "status": "error", "error": str(e)},
                            )
                        )
                    raise
                results[t.name] = out
                if memory is not None:
                    memory.set(f"result.{t.name}", out)
            if memory is not None:
                memory.log(
                    Message(
                        sender=t.name,
                        role="agent",
                        content=f"Task '{t.name}' completed",
                        data={"task": t.name, "status": "done", "result_type": type(out).__name__},
                    )
                )

            # A reviewer should be a feedback agent, not merely a terminal critic.
            # Retry only report-fixable failures; missing computations require a new
            # analysis task and must remain visible as a substantive quality failure.
            if t.name == "reviewer" and isinstance(out, dict) and out.get("status") == "fail":
                violations = list(out.get("violations", []) or [])
                rules = {str(v.get("rule", "")) for v in violations}
                cfg = ctx.get("cfg")
                planning_cfg = getattr(cfg, "planning", None) if cfg is not None else None
                max_revisions = int(getattr(planning_cfg, "max_report_revisions", 1))
                report_fixable = bool(rules) and "analysis.coverage" not in rules
                if report_fixable and max_revisions > 0 and "reporting" in self.agents and memory is not None:
                    self.logger.info("[Orchestrator] Reviewer requested one evidence-safe report revision.")
                    memory.set("review.feedback", out)
                    with memory.as_actor("reporting"):
                        revised_report = self.agents["reporting"].run(ctx)
                        memory.set("result.reporting", revised_report)
                    results["reporting"] = revised_report
                    with memory.as_actor("reviewer"):
                        revised_review = self.agents["reviewer"].run(ctx)
                        revised_review["revision_attempted"] = True
                        memory.set("result.reviewer", revised_review)
                    results["reviewer"] = revised_review
        return results

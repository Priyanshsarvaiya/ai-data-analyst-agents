from __future__ import annotations
from typing import Any, Dict, List
from ai_data_analyst_agents.core.task_planner import Task
from ai_data_analyst_agents.core.messages import Message

class Orchestrator:
    def __init__(self, agents: Dict[str, Any], logger) -> None:
        self.agents = agents
        self.logger = logger

    def run(self, tasks: List[Task], ctx: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
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
            agent = self.agents[t.name]
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
            ctx["memory"].set(f"result.{t.name}", out)
            if memory is not None:
                memory.log(
                    Message(
                        sender=t.name,
                        role="agent",
                        content=f"Task '{t.name}' completed",
                        data={"task": t.name, "status": "done", "result_type": type(out).__name__},
                    )
                )
        return results

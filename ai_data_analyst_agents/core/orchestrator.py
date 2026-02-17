from __future__ import annotations
from typing import Any, Dict, List
from ai_data_analyst_agents.core.task_planner import Task

class Orchestrator:
    def __init__(self, agents: Dict[str, Any], logger) -> None:
        self.agents = agents
        self.logger = logger

    def run(self, tasks: List[Task], ctx: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for t in tasks:
            self.logger.info(f"[Orchestrator] {t.name}: {t.reason}")
            agent = self.agents[t.name]
            out = agent.run(ctx)
            results[t.name] = out
            ctx["memory"].set(f"result.{t.name}", out)
        return results
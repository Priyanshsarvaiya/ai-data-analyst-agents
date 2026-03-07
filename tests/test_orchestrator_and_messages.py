from __future__ import annotations

import pytest

from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.orchestrator import Orchestrator
from ai_data_analyst_agents.core.task_planner import Task


class _OkAgent:
    def __init__(self, value):
        self.value = value

    def run(self, ctx):  # noqa: ANN001
        return {"value": self.value, "has_memory": ctx["memory"] is not None}


class _FailAgent:
    def run(self, ctx):  # noqa: ANN001
        raise RuntimeError("boom")


def test_orchestrator_runs_tasks_and_persists_results(tmp_path) -> None:
    logger = setup_logging("INFO", tmp_path / "logs.txt")
    memory = SharedMemory()
    ctx = {"memory": memory}
    tasks = [Task("intake", "step1"), Task("profiling", "step2")]
    agents = {"intake": _OkAgent(1), "profiling": _OkAgent(2)}

    out = Orchestrator(agents=agents, logger=logger).run(tasks, ctx)

    assert set(out.keys()) == {"intake", "profiling"}
    assert memory.get("result.intake")["value"] == 1
    assert memory.get("result.profiling")["value"] == 2

    # For each task: start + done
    assert len(memory.messages) == 4
    assert memory.messages[0].data["status"] == "start"
    assert memory.messages[1].data["status"] == "done"
    assert memory.messages[2].data["status"] == "start"
    assert memory.messages[3].data["status"] == "done"


def test_orchestrator_logs_error_and_raises(tmp_path) -> None:
    logger = setup_logging("INFO", tmp_path / "logs.txt")
    memory = SharedMemory()
    ctx = {"memory": memory}
    tasks = [Task("intake", "step1"), Task("profiling", "step2")]
    agents = {"intake": _OkAgent(1), "profiling": _FailAgent()}

    with pytest.raises(RuntimeError, match="boom"):
        Orchestrator(agents=agents, logger=logger).run(tasks, ctx)

    assert any(m.data.get("status") == "error" for m in memory.messages)

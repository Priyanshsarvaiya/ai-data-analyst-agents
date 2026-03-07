from __future__ import annotations

from ai_data_analyst_agents.core.messages import Message


def test_message_schema_defaults_and_fields() -> None:
    msg = Message(sender="planner", role="agent", content="done", data={"task": "planner"})
    assert msg.sender == "planner"
    assert msg.role == "agent"
    assert msg.content == "done"
    assert msg.data["task"] == "planner"
    assert msg.ts
    assert msg.id

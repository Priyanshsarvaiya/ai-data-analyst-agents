from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Dict
from datetime import datetime
import uuid

Role = Literal["system", "agent", "tool"]

@dataclass
class Message:
    sender: str
    role: Role
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
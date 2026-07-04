from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    researcher = "researcher"
    coder = "coder"
    critic = "critic"
    summarizer = "summarizer"


class TaskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    # let callers force a starting agent for debugging/evals, otherwise the
    # router decides
    force_agent: Optional[AgentName] = None


class AgentStep(BaseModel):
    agent: AgentName
    input_snippet: str
    output: str
    latency_ms: float
    passed_critic: Optional[bool] = None


class TaskResult(BaseModel):
    session_id: str
    query: str
    steps: list[AgentStep]
    final_answer: str
    completed: bool
    total_latency_ms: float
    self_correction_loops: int


class TokenChunk(BaseModel):
    """Streamed over the websocket while an agent is 'thinking'."""
    session_id: str
    agent: AgentName
    delta: str
    done: bool = False
    meta: Optional[dict[str, Any]] = None

"""
A scripted stand-in for a real LLM.

Lets us test the graph's wiring and load-test the server without paying for
(or waiting on) real model calls. It looks at which agent's prompt it received
and returns a plausible canned answer.
"""
import asyncio
import json
import os
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DELAY = float(os.getenv("FAKE_LLM_DELAY", "0"))

RESPONSES = {
    "PLANNER": json.dumps({"subtasks": [
        "Research the definition and key properties of the concept",
        "Research common edge cases and pitfalls",
        "Write a Python function implementing it with tests",
    ]}),
    "RESEARCHER": "Notes: the concept is well defined; handle empty input and negative numbers.",
    "CODER": "```python\ndef solve(x):\n    return x\n\nassert solve(1) == 1\n```",
    "CRITIC": json.dumps({"approved": True, "feedback": "Looks correct."}),
    "SUMMARIZER": "Summary: researched the concept and implemented a tested solution.",
}


class FakeAgentLLM(BaseChatModel):

    @property
    def _llm_type(self) -> str:
        return "fake-agent"

    def _reply(self, messages):
        prompt = messages[-1].content if messages else ""
        for role, reply in RESPONSES.items():
            if f"ROLE: {role}" in prompt:
                return reply
        return "ok"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if DELAY:
            time.sleep(DELAY)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply(messages)))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if DELAY:
            # Simulates network wait for a remote LLM without blocking the event loop
            await asyncio.sleep(DELAY)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply(messages)))])

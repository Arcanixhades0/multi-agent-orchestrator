import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.orchestrator.llm_client import llm_client


class BaseAgent(ABC):
    name: str
    system_prompt: str

    @abstractmethod
    def build_prompt(self, query: str, context: dict) -> str:
        """Turn the running graph state into a prompt for this agent."""
        raise NotImplementedError

    async def run(self, query: str, context: dict) -> tuple[str, float]:
        """Non-streaming run, used by the graph nodes and benchmarks."""
        prompt = self.build_prompt(query, context)
        start = time.perf_counter()
        output = await llm_client.complete(self.system_prompt, prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        return output, latency_ms

    async def run_streaming(self, query: str, context: dict) -> AsyncIterator[str]:
        """Streaming variant used by the websocket endpoint."""
        prompt = self.build_prompt(query, context)
        async for chunk in llm_client.stream(self.system_prompt, prompt):
            yield chunk

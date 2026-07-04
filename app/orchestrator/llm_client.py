"""
Thin wrapper so agents don't care whether they're talking to OpenAI, Groq,
or the mock provider. The mock provider exists so the whole graph, the
websocket streaming, and the benchmarks can run and be demoed without an API
key or costing anything — set LLM_PROVIDER=openai or groq + the matching key
in .env to hit a real model.
"""
import asyncio
import hashlib
import random
from collections.abc import AsyncIterator

from app.config import settings


class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        self._client = None

        if self.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == "groq":
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=settings.groq_api_key)
        # "mock" needs no client

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        if self.provider == "mock":
            async for chunk in self._mock_stream(system_prompt, user_prompt):
                yield chunk
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        stream = await self._client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        parts = []
        async for chunk in self.stream(system_prompt, user_prompt):
            parts.append(chunk)
        return "".join(parts)

    # ---- mock provider -----------------------------------------------
    # deterministic-ish per input (seeded off a hash of the prompt) so the
    # same query gives comparable, reproducible output across benchmark runs
    # rather than pure noise.
    async def _mock_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        seed = int(hashlib.sha256(user_prompt.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        role = system_prompt.split("\n")[0][:40]
        body = self._mock_body(role, user_prompt, rng)

        for word in body.split(" "):
            await asyncio.sleep(0.002)  # simulate token latency
            yield word + " "

    def _mock_body(self, role: str, user_prompt: str, rng: random.Random) -> str:
        if "Researcher" in role:
            return (
                f"Based on available context, here are the key points relevant to "
                f"'{user_prompt[:60]}': the topic involves multiple interacting factors, "
                f"the most commonly cited source material agrees on the core facts, "
                f"and there are {rng.randint(2, 4)} notable edge cases worth flagging "
                f"to downstream agents."
            )
        if "Coder" in role:
            quality = rng.choice(["clean", "has an off-by-one edge case", "missing error handling"])
            return (
                f"def solve(data):\n"
                f"    # implementation for: {user_prompt[:50]}\n"
                f"    result = process(data)\n"
                f"    return result\n"
                f"# note: this draft is {quality}"
            )
        if "Critic" in role:
            verdict = rng.choices(["PASS", "FAIL"], weights=[0.72, 0.28])[0]
            reason = "meets the stated requirements" if verdict == "PASS" else "misses an edge case or has a logic gap"
            return f"VERDICT: {verdict}. Reasoning: the submitted work {reason}."
        # summarizer
        return (
            f"In short: the task '{user_prompt[:50]}' was resolved by combining "
            f"research findings with the implementation, producing a concise, "
            f"actionable answer for the user."
        )


llm_client = LLMClient()

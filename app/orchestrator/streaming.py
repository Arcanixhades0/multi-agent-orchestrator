"""
Streaming version of the orchestration flow for the websocket endpoint.

This duplicates some of graph.py's control flow instead of trying to bolt
streaming onto the compiled LangGraph app. I went back and forth on this —
LangGraph does support streaming node outputs, but it streams whole node
outputs, not token-by-token deltas from inside a node, and getting
token-level streaming plus the parallel researcher/coder fan-out plus the
critic retry loop all working through astream_events turned into more
plumbing than just writing the loop explicitly here. The non-streaming
/task endpoint uses the real compiled graph in graph.py; this is only for
the live token UX over the websocket.
"""
import asyncio
from collections.abc import AsyncIterator

from app.agents.coder import CoderAgent
from app.agents.critic import CriticAgent
from app.agents.researcher import ResearcherAgent
from app.agents.summarizer import SummarizerAgent
from app.config import settings
from app.models import AgentName, TokenChunk
from app.orchestrator.memory import shared_memory
from app.orchestrator.router import router

researcher_agent = ResearcherAgent()
coder_agent = CoderAgent()
critic_agent = CriticAgent()
summarizer_agent = SummarizerAgent()


async def _stream_agent(agent, agent_name: AgentName, query: str, context: dict, session_id: str) -> AsyncIterator[TokenChunk]:
    buf = []
    async for delta in agent.run_streaming(query, context):
        buf.append(delta)
        yield TokenChunk(session_id=session_id, agent=agent_name, delta=delta)
    full = "".join(buf)
    await shared_memory.write(session_id, f"{agent_name.value}_output", full)
    yield TokenChunk(session_id=session_id, agent=agent_name, delta="", done=True, meta={"full_length": len(full)})


async def run_task_streaming(session_id: str, query: str) -> AsyncIterator[TokenChunk]:
    context: dict = {}
    loops = 0

    decision = router.route(query)
    needs_code = decision.agent == AgentName.coder or any(
        kw in query.lower() for kw in ("code", "function", "bug", "script", "implement")
    )
    yield TokenChunk(session_id=session_id, agent=decision.agent, delta="", done=True,
                      meta={"event": "routed", "confidence": decision.confidence})

    while True:
        # researcher always runs; coder runs alongside it when needed, both
        # streaming concurrently and interleaving their chunks to the client
        tasks = {AgentName.researcher: _stream_agent(researcher_agent, AgentName.researcher, query, context, session_id)}
        if needs_code:
            tasks[AgentName.coder] = _stream_agent(coder_agent, AgentName.coder, query, context, session_id)

        async for chunk in _merge_streams(list(tasks.values())):
            if chunk.done and chunk.meta and "full_length" in chunk.meta:
                # pull the accumulated text back out of shared memory to
                # feed the critic once both branches are done
                context[f"{chunk.agent.value}_output"] = await shared_memory.read(session_id, f"{chunk.agent.value}_output")
            yield chunk

        critic_output, _ = await critic_agent.run(query, context)
        passed, reason = CriticAgent.parse_verdict(critic_output)
        yield TokenChunk(session_id=session_id, agent=AgentName.critic, delta=critic_output, done=True,
                          meta={"passed": passed})

        if passed or loops >= settings.max_self_correction_loops:
            break
        context["critic_feedback"] = reason
        loops += 1

    final_answer, _ = await summarizer_agent.run(query, context)
    yield TokenChunk(session_id=session_id, agent=AgentName.summarizer, delta=final_answer, done=True,
                      meta={"final": True, "self_correction_loops": loops})


async def _merge_streams(streams: list) -> AsyncIterator[TokenChunk]:
    """Interleave chunks from concurrently-running agent streams as they
    arrive, rather than waiting for one to fully finish before the next."""
    queue: asyncio.Queue = asyncio.Queue()
    active = len(streams)

    async def pump(stream):
        nonlocal active
        async for item in stream:
            await queue.put(item)
        active -= 1
        await queue.put(None)

    pumpers = [asyncio.create_task(pump(s)) for s in streams]
    finished = 0
    while finished < len(streams):
        item = await queue.get()
        if item is None:
            finished += 1
            continue
        yield item
    await asyncio.gather(*pumpers)

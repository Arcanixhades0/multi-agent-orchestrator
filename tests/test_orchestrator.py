import pytest

from app.orchestrator.graph import run_task
from app.orchestrator.memory import shared_memory


@pytest.fixture(autouse=True, scope="module")
def _connect_memory():
    import asyncio
    asyncio.get_event_loop().run_until_complete(shared_memory.connect())


@pytest.mark.asyncio
async def test_code_query_runs_researcher_and_coder():
    result = await run_task("test-code-1", "write a function to check if a number is prime")
    assert result["researcher_output"] is not None
    assert result["coder_output"] is not None
    assert result["final_answer"]


@pytest.mark.asyncio
async def test_research_only_query_skips_coder():
    result = await run_task("test-research-1", "what caused the fall of the Roman empire")
    assert result["researcher_output"] is not None
    assert result["coder_output"] is None
    assert result["final_answer"]


@pytest.mark.asyncio
async def test_self_correction_loop_respects_max_retries():
    result = await run_task("test-loop-1", "write a function that does something impossible correctly")
    from app.config import settings
    assert result["loops"] <= settings.max_self_correction_loops


@pytest.mark.asyncio
async def test_trace_records_every_agent_that_ran():
    result = await run_task("test-trace-1", "write a function to merge two sorted lists")
    agents_in_trace = {e["agent"] for e in result["trace"] if e["event"] == "agent_run"}
    assert {"researcher", "coder", "critic", "summarizer"}.issubset(agents_in_trace)


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_share_state():
    import asyncio
    results = await asyncio.gather(
        run_task("concurrent-a", "write a function to sort a list"),
        run_task("concurrent-b", "what is the capital of France"),
    )
    assert results[0]["session_id"] == "concurrent-a"
    assert results[1]["session_id"] == "concurrent-b"
    assert results[0]["coder_output"] is not None
    assert results[1]["coder_output"] is None

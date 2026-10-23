import asyncio
import json
import re
import time

from src.agents import prompts
from src.config import MAX_REVISIONS, MAX_SUBTASKS
from src.llm import get_llm
from src.memory import memory
from src.router import get_router
from src.sandbox import run_python

llm = get_llm()

# "implement ...", "write a function", or a signature like rrf(rankings)
CODE_REQUEST = re.compile(r"\b(implement|write|code|function|class)\b|\w+\(.*\)", re.IGNORECASE)


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def extract_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def research_notes(session_id):
    notes = memory.read_all(session_id, prefix="research:")
    return "\n\n".join(f"- {v}" for v in notes.values()) or "(none)"


def timed(name):
    """Records how long each node takes, so we can find the slow step."""
    def wrap(fn):
        async def inner(state):
            start = time.perf_counter()
            update = await fn(state)
            update.setdefault("timings", {})[name] = round((time.perf_counter() - start) * 1000)
            return update
        return inner
    return wrap


# ---------------- Orchestrator steps ----------------

@timed("plan")
async def plan(state):
    reply = await llm.ainvoke(prompts.PLANNER.format(task=state["task"], max_subtasks=MAX_SUBTASKS))
    subtasks = parse_json(reply.content).get("subtasks") or [state["task"]]
    return {"subtasks": subtasks[:MAX_SUBTASKS]}


@timed("route")
async def route(state):
    router = get_router()

    # Embedding is CPU work. Run it in a thread so it doesn't freeze the event loop
    # (and every other user's WebSocket) while it computes.
    routes = await asyncio.to_thread(
        lambda: [{"subtask": s, **router.route(s)} for s in state["subtasks"]]
    )

    # Safety net: semantic routing is fuzzy. In evaluation, the "rrf" task had every subtask
    # routed to researchers, so no code was ever written. If the task clearly asks for code
    # and nothing went to the coder, add a coding subtask.
    if CODE_REQUEST.search(state["task"]) and not any(r["agent"] == "coder" for r in routes):
        routes.append({"subtask": f"Implement in Python: {state['task']}", "agent": "coder", "score": 0.0, "ms": 0.0, "forced": True})

    return {"routes": routes}


# ---------------- Agents ----------------

async def researcher(job):
    """Runs once per research subtask. In parallel mode, several run at the same time."""
    reply = await llm.ainvoke(prompts.RESEARCHER.format(task=job["task"], subtask=job["subtask"]))
    memory.write(job["session_id"], f"research:{job['index']}", reply.content)
    return {"results": {f"research:{job['index']}": reply.content}}


@timed("research_sequential")
async def research_sequential(state):
    """Baseline for comparison: the same research subtasks, one after another."""
    results = {}
    for i, r in enumerate(state["routes"]):
        if r["agent"] == "researcher":
            update = await researcher({**state, "subtask": r["subtask"], "index": i})
            results.update(update["results"])
    return {"results": results}


@timed("coder")
async def coder(state):
    coding = [r["subtask"] for r in state["routes"] if r["agent"] == "coder"]

    feedback = ""
    if state.get("feedback"):
        feedback = prompts.CODER_FEEDBACK.format(code=state["code"], feedback=state["feedback"])

    reply = await llm.ainvoke(prompts.CODER.format(
        task=state["task"],
        subtasks="; ".join(coding),
        # Cross-agent context: the coder reads what the researchers wrote to shared memory
        research=research_notes(state["session_id"]),
        feedback=feedback,
    ))

    code = extract_code(reply.content)
    execution = await asyncio.to_thread(run_python, code)
    memory.write(state["session_id"], "code", code)

    return {"code": code, "execution": execution, "revisions": state.get("revisions", 0) + 1}


@timed("critic")
async def critic(state):
    execution = state["execution"]

    # If the code crashed or an assert failed, no need to ask the LLM: send the error back
    if not execution["passed"]:
        return {"approved": False, "feedback": f"The code failed when run:\n{execution['stderr']}"}

    reply = await llm.ainvoke(prompts.CRITIC.format(task=state["task"], code=state["code"]))
    verdict = parse_json(reply.content)

    return {
        "approved": bool(verdict.get("approved", True)),
        "feedback": verdict.get("feedback", ""),
    }


@timed("summarizer")
async def summarizer(state):
    reply = await llm.ainvoke(prompts.SUMMARIZER.format(
        task=state["task"],
        research=research_notes(state["session_id"]),
        code=state.get("code") or "(no code needed)",
    ))
    return {"final": reply.content}


def needs_revision(state):
    if state.get("approved") or state.get("revisions", 0) >= MAX_REVISIONS:
        return "summarizer"
    return "coder"

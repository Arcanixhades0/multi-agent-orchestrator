r"""
The actual orchestration graph.

Shape of it:

              route
             /     \
     researcher      coder     <- run concurrently when the query needs code
             \     /
              critic
             /     \
   (fail, retries left)    (pass, or out of retries)
           |                        |
      back to route            summarizer -> END

A few notes from getting this working:
- LangGraph only runs `critic` once both `researcher` and `coder` have
  written to state, which is what gives us the "parallel branch + join"
  behavior described as the parallel execution engine. If a query is
  research-only (no code needed), the coder node is skipped entirely and
  critic just waits on researcher.
- The self-correction loop is a conditional edge out of critic: on FAIL it
  routes back to the branch node(s) with critic_feedback populated in state,
  and increments a loop counter so we don't spin forever.
- Cross-agent context propagation happens two ways: (1) inside the graph
  state itself, which LangGraph merges automatically, and (2) mirrored into
  Redis via SharedMemory so that a separate process — e.g. the eval script,
  or a future "resume this session" endpoint — can inspect what each agent
  saw without needing access to the live graph run.
"""
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.coder import CoderAgent
from app.agents.critic import CriticAgent
from app.agents.researcher import ResearcherAgent
from app.agents.summarizer import SummarizerAgent
from app.config import settings
from app.orchestrator.memory import shared_memory
from app.orchestrator.router import router

researcher_agent = ResearcherAgent()
coder_agent = CoderAgent()
critic_agent = CriticAgent()
summarizer_agent = SummarizerAgent()


class GraphState(TypedDict):
    session_id: str
    query: str
    needs_code: bool
    researcher_output: Optional[str]
    coder_output: Optional[str]
    critic_feedback: Optional[str]
    critic_passed: Optional[bool]
    loops: int
    final_answer: Optional[str]
    # Annotated with operator.add so parallel branches appending to trace
    # don't clobber each other — LangGraph reduces concurrent writes with
    # this function instead of raising a conflict.
    trace: Annotated[list[dict], operator.add]


async def _route_node(state: GraphState) -> dict:
    decision = router.route(state["query"])
    needs_code = decision.agent.value == "coder" or any(
        kw in state["query"].lower() for kw in ("code", "function", "bug", "script", "implement")
    )
    await shared_memory.write(state["session_id"], "routing_decision", decision.agent.value)
    return {
        "needs_code": needs_code,
        "trace": [{"event": "route", "agent": decision.agent.value, "confidence": decision.confidence,
                    "latency_ms": decision.latency_ms}],
    }


async def _researcher_node(state: GraphState) -> dict:
    output, latency = await researcher_agent.run(state["query"], dict(state))
    await shared_memory.write(state["session_id"], "researcher_output", output)
    return {
        "researcher_output": output,
        "trace": [{"event": "agent_run", "agent": "researcher", "latency_ms": round(latency, 2)}],
    }


async def _coder_node(state: GraphState) -> dict:
    output, latency = await coder_agent.run(state["query"], dict(state))
    await shared_memory.write(state["session_id"], "coder_output", output)
    return {
        "coder_output": output,
        "trace": [{"event": "agent_run", "agent": "coder", "latency_ms": round(latency, 2)}],
    }


async def _critic_node(state: GraphState) -> dict:
    output, latency = await critic_agent.run(state["query"], dict(state))
    passed, reason = CriticAgent.parse_verdict(output)
    await shared_memory.write(state["session_id"], "critic_output", output)
    return {
        "critic_passed": passed,
        "critic_feedback": None if passed else reason,
        "loops": state["loops"] + (0 if passed else 1),
        "trace": [{"event": "agent_run", "agent": "critic", "latency_ms": round(latency, 2), "passed": passed}],
    }


async def _summarizer_node(state: GraphState) -> dict:
    output, latency = await summarizer_agent.run(state["query"], dict(state))
    return {
        "trace": [{"event": "agent_run", "agent": "summarizer", "latency_ms": round(latency, 2)}],
        "final_answer": output,
    }


def _after_route(state: GraphState) -> list[str]:
    return ["researcher", "coder"] if state["needs_code"] else ["researcher"]


def _after_critic(state: GraphState) -> str:
    if state["critic_passed"] or state["loops"] >= settings.max_self_correction_loops:
        return "summarizer"
    return "retry"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("route", _route_node)
    graph.add_node("researcher", _researcher_node)
    graph.add_node("coder", _coder_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("summarizer", _summarizer_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges("route", _after_route, {"researcher": "researcher", "coder": "coder"})

    # both branches join at critic; LangGraph waits for all incoming edges
    graph.add_edge("researcher", "critic")
    graph.add_edge("coder", "critic")

    graph.add_conditional_edges(
        "critic",
        _after_critic,
        {"summarizer": "summarizer", "retry": "route"},
    )
    graph.add_edge("summarizer", END)

    return graph.compile()


compiled_graph = build_graph()


async def run_task(session_id: str, query: str) -> GraphState:
    initial: GraphState = {
        "session_id": session_id,
        "query": query,
        "needs_code": False,
        "researcher_output": None,
        "coder_output": None,
        "critic_feedback": None,
        "critic_passed": None,
        "loops": 0,
        "final_answer": None,
        "trace": [],
    }
    result = await compiled_graph.ainvoke(initial)
    return result

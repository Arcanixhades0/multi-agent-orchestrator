"""
The orchestration graph.

    plan -> route -> [researcher, researcher, ...]  (parallel fan-out)
                  -> gather -> coder <-> critic     (self-correction loop)
                            -> summarizer -> END

LangGraph runs every node that is ready in the same "superstep" at the same time,
so all the researchers sent by `dispatch` run in parallel. `gather` only starts
once every one of them has finished.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents import nodes


def merge(left, right):
    # Reducer: parallel branches each return part of a dict, and this combines them
    return {**(left or {}), **(right or {})}


class State(TypedDict, total=False):
    session_id: str
    task: str
    subtasks: list[str]
    routes: list[dict]
    results: Annotated[dict, merge]
    code: str
    execution: dict
    approved: bool
    feedback: str
    revisions: int
    final: str
    timings: Annotated[dict, merge]


def after_research(state):
    has_code = any(r["agent"] == "coder" for r in state["routes"])
    return "coder" if has_code else "summarizer"


def dispatch(state):
    """Fan out: one researcher per research subtask, all at once."""
    jobs = [
        Send("researcher", {
            "session_id": state["session_id"],
            "task": state["task"],
            "subtask": r["subtask"],
            "index": i,
        })
        for i, r in enumerate(state["routes"])
        if r["agent"] == "researcher"
    ]
    return jobs or "gather"


async def gather(state):
    return {}


def build_graph(parallel=True):
    graph = StateGraph(State)

    graph.add_node("plan", nodes.plan)
    graph.add_node("route", nodes.route)
    graph.add_node("gather", gather)
    graph.add_node("coder", nodes.coder)
    graph.add_node("critic", nodes.critic)
    graph.add_node("summarizer", nodes.summarizer)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "route")

    if parallel:
        graph.add_node("researcher", nodes.researcher)
        graph.add_conditional_edges("route", dispatch, ["researcher", "gather"])
        graph.add_edge("researcher", "gather")
    else:
        graph.add_node("research_sequential", nodes.research_sequential)
        graph.add_edge("route", "research_sequential")
        graph.add_edge("research_sequential", "gather")

    graph.add_conditional_edges("gather", after_research, ["coder", "summarizer"])
    graph.add_edge("coder", "critic")
    graph.add_conditional_edges("critic", nodes.needs_revision, ["coder", "summarizer"])
    graph.add_edge("summarizer", END)

    return graph.compile()


if __name__ == "__main__":
    import asyncio
    import sys
    import uuid

    task = " ".join(sys.argv[1:]) or "Explain run-length encoding and implement rle_encode(s) and rle_decode(s)."

    async def main():
        app = build_graph()
        async for update in app.astream({"session_id": uuid.uuid4().hex, "task": task}, stream_mode="updates"):
            for node, data in update.items():
                print(f"\n--- {node} ---")
                if node == "route":
                    for r in data["routes"]:
                        print(f"  {r['agent']:>10} ({r['score']:.2f}, {r['ms']:.0f}ms): {r['subtask']}")
                elif node == "critic":
                    print(f"  approved={data['approved']} {data.get('feedback', '')[:200]}")
                elif node == "summarizer":
                    print(data["final"])

    asyncio.run(main())

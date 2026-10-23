import os

# Must be set before src modules import the config
os.environ["LLM_PROVIDER"] = "fake"
os.environ["SANDBOX"] = "subprocess"
os.environ["REDIS_URL"] = ""

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app, create_token  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.memory import SharedMemory  # noqa: E402
from src.router import get_router  # noqa: E402
from src.sandbox import run_python  # noqa: E402

client = TestClient(app)


# ---------------- Router ----------------

@pytest.mark.parametrize("subtask, agent", [
    ("Explain how Dijkstra's algorithm works", "researcher"),
    ("Find common edge cases and pitfalls", "researcher"),
    ("Implement the function in Python with asserts", "coder"),
    ("Write unit tests for the parser", "coder"),
])
def test_router_picks_right_agent(subtask, agent):
    assert get_router().route(subtask)["agent"] == agent


# ---------------- Sandbox ----------------

def test_sandbox_pass_fail_and_timeout():
    assert run_python("assert 1 + 1 == 2")["passed"]
    assert not run_python("assert 1 + 1 == 3")["passed"]
    assert "Timed out" in run_python("while True: pass", timeout=2)["stderr"]


# ---------------- Memory ----------------

def test_memory_prefix_and_clear():
    m = SharedMemory(url="")
    m.write("s1", "research:0", "notes A")
    m.write("s1", "research:1", "notes B")
    m.write("s1", "code", "print(1)")
    assert list(m.read_all("s1", prefix="research:").values()) == ["notes A", "notes B"]
    m.clear("s1")
    assert m.read_all("s1") == {}


# ---------------- Graph ----------------

@pytest.mark.parametrize("parallel", [True, False])
def test_graph_runs_end_to_end(parallel):
    state = asyncio.run(build_graph(parallel).ainvoke({"session_id": "t", "task": "Explain and implement solve(x)"}))
    assert state["approved"]
    assert state["final"].startswith("Summary")
    assert len(state["results"]) == 2  # both research subtasks ran


# ---------------- API ----------------

def test_login():
    assert client.post("/token", json={"username": "demo", "password": "demo-password"}).status_code == 200
    assert client.post("/token", json={"username": "demo", "password": "wrong"}).status_code == 401


def test_websocket_rejects_bad_token():
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=not-a-jwt") as ws:
            ws.receive_json()


def test_websocket_streams_to_done():
    with client.websocket_connect(f"/ws?token={create_token('demo')}") as ws:
        ws.send_json({"task": "Explain and implement solve(x)"})
        nodes = []
        while True:
            event = ws.receive_json()
            if event["type"] == "node":
                nodes.append(event["node"])
            if event["type"] == "done":
                break
    assert event["answer"]
    assert nodes[0] == "plan" and nodes[-1] == "summarizer"
    assert nodes.count("researcher") == 2


def test_coder_is_forced_when_task_asks_for_code(monkeypatch):
    from src.agents import nodes

    class AllResearch:
        def route(self, subtask):
            return {"agent": "researcher", "score": 0.5, "ms": 1.0}

    monkeypatch.setattr(nodes, "get_router", lambda: AllResearch())
    update = asyncio.run(nodes.route({"task": "Explain RRF and implement rrf(rankings)", "subtasks": ["Research RRF"]}))
    assert any(r["agent"] == "coder" and r.get("forced") for r in update["routes"])

    update = asyncio.run(nodes.route({"task": "Explain how RRF works", "subtasks": ["Research RRF"]}))
    assert not any(r["agent"] == "coder" for r in update["routes"])

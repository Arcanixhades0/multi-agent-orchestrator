"""
API for the multi-agent system.

  POST /token   {"username", "password"}  ->  {"access_token": JWT}
  WS   /ws?token=<JWT>
       client sends   {"task": "..."}
       server streams {"type": "node",  "node": "route", "data": {...}}   as each agent finishes
                      {"type": "token", "node": "summarizer", "text": "..."}   live LLM tokens
                      {"type": "done",  "answer": "...", "timings": {...}}

Run:  uvicorn src.api.main:app
"""
import asyncio
import hmac
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from src.config import DEMO_PASSWORD, DEMO_USER, JWT_SECRET, JWT_TTL_MINUTES, LLM_PROVIDER
from src.graph import build_graph
from src.memory import memory
from src.router import get_router

@asynccontextmanager
async def lifespan(app):
    # Load the router's embedding model before accepting traffic, not on the first request
    await asyncio.to_thread(get_router)
    yield


app = FastAPI(title="Multi-Agent Orchestrator", lifespan=lifespan)
graph = build_graph(parallel=True)

active_sessions = set()


class Login(BaseModel):
    username: str
    password: str


def create_token(username):
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_TTL_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token):
    # Raises if the signature is wrong or the token expired
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])["sub"]


@app.post("/token")
def login(body: Login):
    # compare_digest takes the same time whether the first or last character is wrong,
    # so attackers can't guess the password one character at a time by timing responses
    ok = hmac.compare_digest(body.username, DEMO_USER) and hmac.compare_digest(body.password, DEMO_PASSWORD)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(body.username), "token_type": "bearer"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "memory": memory.backend,
        "active_sessions": len(active_sessions),
    }


def summarize_update(node, data):
    """Keep WebSocket messages small: send what the UI needs, not the whole state."""
    if node == "route":
        return {"routes": [{"agent": r["agent"], "subtask": r["subtask"]} for r in data["routes"]]}
    if node == "coder":
        return {"passed": data["execution"]["passed"], "revision": data["revisions"]}
    if node == "critic":
        return {"approved": data["approved"], "feedback": data.get("feedback", "")[:300]}
    if node == "plan":
        return {"subtasks": data["subtasks"]}
    return {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = ""):
    try:
        user = verify_token(token)
    except jwt.PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    session_id = uuid.uuid4().hex
    active_sessions.add(session_id)

    try:
        while True:
            message = await websocket.receive_json()
            task = (message.get("task") or "").strip()
            if not task:
                await websocket.send_json({"type": "error", "detail": "empty task"})
                continue

            start = time.perf_counter()
            final_state = {}

            # "updates": one event per finished node; "messages": LLM tokens as they are generated
            async for mode, chunk in graph.astream(
                {"session_id": session_id, "task": task},
                stream_mode=["updates", "messages"],
            ):
                if mode == "messages":
                    token_chunk, meta = chunk
                    if meta.get("langgraph_node") == "summarizer" and token_chunk.content:
                        await websocket.send_json({"type": "token", "node": "summarizer", "text": token_chunk.content})
                else:
                    for node, data in chunk.items():
                        final_state.update(data or {})
                        await websocket.send_json({"type": "node", "node": node, "data": summarize_update(node, data or {})})

            await websocket.send_json({
                "type": "done",
                "user": user,
                "answer": final_state.get("final", ""),
                "seconds": round(time.perf_counter() - start, 2),
            })

    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.discard(session_id)
        memory.clear(session_id)

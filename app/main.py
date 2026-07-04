import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.auth import create_access_token, ws_auth
from app.config import settings
from app.models import TaskRequest, TaskResult, AgentStep
from app.orchestrator.graph import run_task
from app.orchestrator.memory import shared_memory
from app.orchestrator.streaming import run_task_streaming

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await shared_memory.connect()
    logger.info("shared memory connected (redis=%s)", not shared_memory._using_fallback)
    yield
    await shared_memory.close()


app = FastAPI(title="Multi-Agent LLM Orchestration System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before putting real infra behind it
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple in-process counter to cap concurrent websocket sessions — a real
# deployment would track this in redis across workers, but for a single
# worker this is enough to enforce the 50-session ceiling
_active_ws_sessions = 0
_ws_lock = asyncio.Lock()


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "redis_connected": not shared_memory._using_fallback}


@app.post("/token")
async def issue_token(client_id: str):
    """Dev convenience endpoint — in production this would sit behind real
    login, not take a bare client_id."""
    return {"access_token": create_access_token(client_id), "token_type": "bearer"}


@app.post("/task", response_model=TaskResult)
async def run_task_endpoint(request: TaskRequest):
    session_id = request.session_id or str(uuid.uuid4())
    result = await run_task(session_id, request.query)

    steps = []
    for event in result["trace"]:
        if event["event"] != "agent_run":
            continue
        agent = event["agent"]
        output_key = f"{agent}_output" if agent != "critic" else None
        output = result.get(output_key, "") if output_key else "critic verdict recorded"
        steps.append(AgentStep(
            agent=agent,
            input_snippet=request.query[:80],
            output=(output or "")[:500],
            latency_ms=event["latency_ms"],
            passed_critic=event.get("passed"),
        ))

    total_latency = sum(e["latency_ms"] for e in result["trace"] if "latency_ms" in e)
    return TaskResult(
        session_id=session_id,
        query=request.query,
        steps=steps,
        final_answer=result.get("final_answer") or "",
        completed=bool(result.get("final_answer")),
        total_latency_ms=round(total_latency, 2),
        self_correction_loops=result["loops"],
    )


@app.websocket("/ws/task")
async def ws_task(websocket: WebSocket, client_id: str = Depends(ws_auth)):
    global _active_ws_sessions

    async with _ws_lock:
        if _active_ws_sessions >= settings.max_ws_sessions:
            await websocket.close(code=1013, reason="server at capacity")
            return
        _active_ws_sessions += 1

    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("ws session %s opened for client %s (active=%d)", session_id, client_id, _active_ws_sessions)

    try:
        payload = await websocket.receive_json()
        query = payload.get("query", "")
        if not query:
            await websocket.send_json({"error": "query is required"})
            return

        async for chunk in run_task_streaming(session_id, query):
            await websocket.send_json(chunk.model_dump())

    except WebSocketDisconnect:
        logger.info("ws session %s disconnected by client", session_id)
    finally:
        async with _ws_lock:
            _active_ws_sessions -= 1

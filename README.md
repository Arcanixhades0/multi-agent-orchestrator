# Multi-Agent LLM Orchestration System

A 4-agent system (Researcher, Coder, Critic, Summarizer) built on LangGraph,
with a semantic router that decides which agents a query needs, a
self-correction loop where the critic can bounce work back for another pass,
and a FastAPI layer that exposes it over both a plain REST endpoint and a
JWT-secured websocket for token-by-token streaming.

I built this to understand how LangGraph actually handles parallel branches
and conditional retries under the hood, not just chain a few prompts
together — most of the interesting bugs (see `graph.py` comments) were in
getting the state merging right when researcher and coder write to shared
state concurrently.

## Architecture

```
 client
   |
   |  POST /task  (blocking, full result)
   |  WS   /ws/task?token=...  (streaming tokens, JWT required)
   v
 FastAPI app  ──────────────────────────────────────────────
   |
   v
 route  ──semantic similarity (TF-IDF)──>  needs_code? ──┐
   |                                                       |
   v                                                       v
 researcher (always)                                    coder (if needed)
   |                                                       |
   └──────────────────────┬────────────────────────────────┘
                           v
                        critic
                    /            \
              FAIL, retries left   PASS or retries exhausted
                    |                        |
              back to route            summarizer ──> final answer
```

Researcher and coder run concurrently as separate LangGraph branches that
join at the critic node — LangGraph won't run `critic` until both
predecessor branches have written to state. Shared context between agents
goes through both the in-memory graph state (for the current run) and Redis
(`app/orchestrator/memory.py`) so a separate process could inspect what an
agent saw mid-run without hooking into the live graph.

## Why a mock LLM mode exists

`LLM_PROVIDER=mock` (the default) runs the entire graph, the websocket
streaming, and the benchmarks without any API key or cost. It's not a fake
demo — every code path (routing, parallel execution, the critic retry loop,
Redis writes, JWT auth, streaming) runs for real, only the actual text
generation is templated instead of calling OpenAI/Groq. Set
`LLM_PROVIDER=openai` or `groq` plus the matching key in `.env` to use a
real model; nothing else about the architecture changes.

## Running it

### Local, no Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults to mock mode, works out of the box
uvicorn app.main:app --reload
```

Server comes up on `http://localhost:8000`. Without Redis running it falls
back to an in-process dict automatically (logged at startup) — fine for
local dev, but you lose cross-process session sharing.

### Docker Compose (app + Redis)

```bash
cd docker
docker compose up --build
```

### Try it

```bash
# blocking REST call
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"query": "write a function to check if a string is a palindrome"}'

# streaming, token-by-token
python scripts/demo_client.py "write a function to reverse a linked list"
```

## Tests

```bash
pytest tests/ -v
```

10 tests covering routing accuracy/latency, parallel vs. sequential-only
branching, the self-correction retry cap, and that concurrent sessions
don't leak state into each other.

## Benchmarks — and what the numbers actually mean

The resume-facing numbers (routing overhead, parallel-vs-sequential latency
reduction, task completion rate) are backed by real scripts in
`benchmarks/`, not made up. Full output and methodology notes are in
[`benchmarks/results.md`](benchmarks/results.md) — short version:

| Claim | Measured (mock mode, this machine) |
|---|---|
| Routing overhead | 0.59ms mean, well under 200ms |
| Latency reduction from parallel execution | ~40% vs sequential |
| Task completion rate (critic-approved) | 90% over 30 labeled queries |

Run them yourself:

```bash
bash scripts/run_benchmarks.sh
```

These numbers are measured against the mock provider so anyone can
reproduce them without an API key. Swap to a real provider in `.env` and
rerun before quoting these numbers anywhere that matters — latency and
completion rate will shift with a real model's response time and quality.

## API

- `POST /token?client_id=...` — issues a JWT (dev convenience; a real
  deployment would put this behind actual auth)
- `POST /task` — `{"query": "..."}` → full `TaskResult` with per-agent trace
- `WS /ws/task?token=...` — send `{"query": "..."}`, receive a stream of
  `TokenChunk` messages as each agent generates its part
- `GET /healthz`

Capped at `MAX_WS_SESSIONS` (default 50) concurrent websocket connections
per worker — beyond that the server closes new connections with code 1013
until one frees up.

## Project layout

```
app/
  agents/          researcher / coder / critic / summarizer
  orchestrator/
    graph.py        the compiled LangGraph state machine
    streaming.py     token-level streaming variant for the websocket
    router.py        TF-IDF semantic routing
    memory.py        Redis-backed shared memory
    llm_client.py    provider abstraction (mock/openai/groq)
  main.py           FastAPI app, REST + websocket routes
  auth.py           JWT issuing/verification
tests/              pytest suite
benchmarks/         the scripts behind every number in this README
docker/             Dockerfile + docker-compose.yml
```

## Known limitations

- The mock LLM provider's "quality" is templated, so completion-rate
  numbers in mock mode reflect the critic's pass/fail logic more than real
  answer quality — treat them as a proof of the retry mechanism working,
  not a quality benchmark, until run against a real model.
- No persistent task history/database — Redis TTLs sessions out after an
  hour by default (`REDIS_TTL_SECONDS`).
- Single-process websocket session cap; horizontal scaling would need the
  session counter moved into Redis too.

# Multi-Agent LLM Orchestration System

Four specialised agents (Researcher, Coder, Critic, Summarizer) coordinated by a LangGraph state graph,
with parallel execution, shared Redis memory, semantic routing, a self-correction loop, and a
JWT-secured WebSocket API that streams progress and tokens live.

```
            ┌──────────┐
 task ───▶  │  plan    │  LLM splits the task into ≤4 subtasks
            └────┬─────┘
            ┌────▼─────┐
            │  route   │  semantic router: embedding similarity → researcher or coder (no LLM call)
            └────┬─────┘
       ┌─────────┼─────────┐      parallel fan-out (LangGraph Send)
  ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐
  │research│ │research│ │research│  each writes notes to shared memory (Redis hash per session)
  └────┬───┘ └───┬────┘ └──┬─────┘
       └─────────┼─────────┘
            ┌────▼─────┐
            │  coder   │◀──────┐   reads ALL research notes from memory, writes code + asserts,
            └────┬─────┘       │   runs it in a sandbox (Docker: no network, 256MB, 1 CPU)
            ┌────▼─────┐       │
            │  critic  │───────┘   failed run → error sent back; LLM review → approve / revise (max 3)
            └────┬─────┘
            ┌────▼─────┐
            │summarizer│  final answer, tokens streamed over WebSocket
            └──────────┘
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env              # add GROQ_API_KEY and a JWT_SECRET
docker run -d -p 6379:6379 redis:7-alpine

python -m src.graph "Explain the Luhn algorithm and implement luhn_valid(number: str) -> bool"
uvicorn src.api.main:app          # API on :8000
python -m pytest -q               # 11 tests, no API key needed (fake LLM)
```

## Measure

```bash
python -m eval.run_eval                      # completion rate + parallel vs sequential latency → reports/eval_report.json
# server with fake LLM, then:
python -m eval.load_test --users 50          # concurrent WebSocket sessions → reports/
```

## Results

Model: `openai/gpt-oss-20b` on Groq (free tier). Raw data: `reports/eval_results.jsonl`.

| Metric | Value | How it was measured |
|---|---|---|
| Task completion rate | **85%** (17/20) | 20 tasks, hidden unit tests the agents never see |
| First-try success | 65% | completed with no critic revision; **self-correction adds +20 pts** |
| Routing overhead | p95 41ms | `route` node timing, CPU embedding, no LLM call |
| Concurrent WebSocket sessions | 50/50 succeeded, p95 14.2s | `eval/load_test.py`, fake LLM with 0.5s/call (tests the server, not the LLM) |
| Parallel vs sequential latency | **only 2% faster** (10 tasks in both modes) | Research is a small share of the time (coder + critic + summarizer dominate), and the free tier's 8k tokens/min throttles parallel calls |

Failures: `rrf` (router sent every subtask to researchers, so no code was written → fixed with a
rule that forces a coder when the task asks for code; not yet re-measured), `rle` and `word_freq`
(coder never got its own asserts passing within 3 revisions).

## Design decisions (interview answers)

- **Why a graph, not a chain?** The critic → coder loop is a cycle; chains can't loop. LangGraph also gives parallel fan-out and streaming.
- **Why semantic routing instead of asking the LLM?** ~30ms and free vs ~1s and tokens. Weakness found in testing: nouns dominate similarity ("Explain Dijkstra's *algorithm*" went to the coder) → fixed by writing profiles around actions.
- **Why the critic skips the LLM when code fails?** A traceback is better feedback than an opinion, and it saves a call.
- **Why hidden tests?** The coder writes its own asserts; it could write weak ones. Completion is judged by tests it never saw.
- **Why Redis?** Shared across processes and restarts; `HSET` per key is atomic so parallel agents never overwrite each other.
- **Bug found by load testing:** lazy model loading without a lock → 50 users loaded the embedding model 50 times → event loop starved, every WebSocket timed out. Fixed with double-checked locking + preloading at startup.
- **Sandbox:** never `exec()` model output in-process. Docker with `--network none` and memory/CPU limits.

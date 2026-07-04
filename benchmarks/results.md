# Benchmark results

Numbers below are from an actual run against the **mock LLM provider**
(`LLM_PROVIDER=mock`), on a single CPU core, no GPU. They're here so the
numbers in the README aren't just made up — run any of these three scripts
yourself and you'll get output in the same ballpark (the mock critic has a
weighted random pass/fail baked in, so exact numbers move a little run to
run).

Swap `LLM_PROVIDER=openai` or `groq` in `.env` and re-run these against a
real model before quoting numbers anywhere that matters — mock mode is for
running the benchmarks without an API key, not a substitute for measuring
against the model you'd actually deploy with.

## Routing (`python -m benchmarks.routing_benchmark`)

```
labeled queries:     18
routing accuracy:    15/18 (83.3%)
latency mean (ms):   0.593
latency p95 (ms):    0.697
latency max (ms):    0.768
```

Routing overhead is comfortably under the 200ms target — it's TF-IDF +
cosine similarity against a small reference set, no network call and no
model load, so most of the "latency" is just python function call overhead.

## Parallel vs. sequential execution (`python -m benchmarks.parallel_vs_sequential_benchmark`)

```
sequential mean (ms):   178.9
parallel mean (ms):     107.2
latency reduction:      40.1%
```

Running researcher + coder concurrently instead of one after another cuts
per-task latency by ~40% here. The real-world number depends heavily on
how close the two agents' individual latencies are to each other — if one
agent is much slower than the other, the reduction from parallelizing goes
up, since the wall clock is bounded by the slower branch instead of the sum
of both. This will look different against a real model where researcher
and coder calls take noticeably longer and vary more.

## Task completion rate (`python -m benchmarks.task_completion_benchmark`)

```
queries run:            30
critic-approved:        27 (90.0%)
exhausted retry budget: 3
self-correction loops:  mean=0.33, max=2
per-task latency (ms):  mean=274.7, p95=496.0
wall clock for batch:   6.89s
```

"Critic-approved" means the critic agent's verdict was PASS, either on the
first pass or after a self-correction retry — not just "the summarizer
produced some text," which is trivially always true since the summarizer
runs regardless of critic verdict once the retry budget is exhausted.

## Reproducing against a real model

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER=openai and OPENAI_API_KEY=sk-...
python -m benchmarks.routing_benchmark
python -m benchmarks.parallel_vs_sequential_benchmark
python -m benchmarks.task_completion_benchmark
```

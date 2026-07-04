"""
Runs a batch of multi-step queries through the graph and reports the
completion rate (critic passed, or graph produced a final answer at all
within the retry budget). This is what the "task completion rate" number in
the README is measured from — run this yourself with `python -m
benchmarks.task_completion_benchmark` to reproduce it, it prints a fresh
number every run since the mock critic has some randomness built in
(weighted 72/28 pass/fail per review, same as a real critic wouldn't be
100% consistent either).

Swap LLM_PROVIDER to openai/groq in .env before running this against a real
model — the mock numbers are a stand-in so the benchmark script itself is
provable without spending API credits, not a substitute for the real run.
"""
import asyncio
import statistics
import time

from app.orchestrator.graph import run_task
from app.orchestrator.memory import shared_memory

QUERIES = [
    "write a function that reverses a linked list",
    "what caused the 2008 financial crisis",
    "implement binary search in python",
    "summarize the plot of the Odyssey",
    "fix a bug where my recursive fibonacci function is too slow",
    "compare TCP and UDP",
    "write unit tests for a stack implementation",
    "what is the difference between supervised and unsupervised learning",
    "refactor a function that has three nested for loops",
    "explain how DNS resolution works",
    "write a function to detect a cycle in a linked list",
    "research the causes of the French Revolution",
    "implement a rate limiter in python",
    "what are the tradeoffs between SQL and NoSQL databases",
    "write a function to merge two sorted arrays",
    "explain gradient descent",
    "debug a memory leak in a long running python service",
    "summarize the key ideas in the CAP theorem",
    "write a decorator that retries a function on failure",
    "compare REST and GraphQL",
    "implement a LRU cache",
    "what is the halting problem",
    "write a function to validate a binary search tree",
    "explain how consistent hashing works",
    "write a script to deduplicate rows in a csv file",
    "what is the difference between processes and threads",
    "implement quicksort and explain its worst case",
    "summarize how OAuth 2.0 authorization code flow works",
    "write a function to flatten a nested list",
    "explain eventual consistency",
]


async def main():
    await shared_memory.connect()

    critic_approved = 0
    exhausted_retries = 0
    total_latency = []
    loop_counts = []

    start_all = time.perf_counter()
    for q in QUERIES:
        result = await run_task(f"bench-{hash(q) % 10000}", q)
        # "completed" = critic actually approved the work, not just "a
        # final answer exists" — the summarizer always produces *something*
        # even when the critic never passed it within the retry budget, and
        # counting those as completions would make this number meaningless
        if result.get("critic_passed"):
            critic_approved += 1
        else:
            exhausted_retries += 1
        loop_counts.append(result["loops"])
        latency = sum(e["latency_ms"] for e in result["trace"] if "latency_ms" in e)
        total_latency.append(latency)
    wall_clock_s = time.perf_counter() - start_all

    n = len(QUERIES)
    print(f"queries run:            {n}")
    print(f"critic-approved:        {critic_approved} ({critic_approved / n:.1%})")
    print(f"exhausted retry budget: {exhausted_retries}")
    print(f"self-correction loops:   mean={statistics.mean(loop_counts):.2f}, max={max(loop_counts)}")
    print(f"per-task latency (ms):   mean={statistics.mean(total_latency):.1f}, "
          f"p95={sorted(total_latency)[int(n * 0.95) - 1]:.1f}")
    print(f"wall clock for batch:    {wall_clock_s:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

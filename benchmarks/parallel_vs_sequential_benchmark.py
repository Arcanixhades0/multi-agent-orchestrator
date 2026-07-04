"""
Compares wall-clock latency of running researcher+coder as the graph does
(concurrently, joined at critic) vs. running them sequentially one after
another. Backs the "reduced pipeline latency by X% vs sequential baseline"
claim — run this yourself, the exact percentage will vary with hardware and
whichever LLM_PROVIDER is configured; the number below is what mock mode on
this machine produced.

Run: python -m benchmarks.parallel_vs_sequential_benchmark
"""
import asyncio
import statistics
import time

from app.agents.coder import CoderAgent
from app.agents.researcher import ResearcherAgent

researcher = ResearcherAgent()
coder = CoderAgent()

QUERIES = [
    "write a function to check if a string is a palindrome",
    "implement a binary search tree insert method",
    "write a function to find the longest common subsequence",
    "implement a rate limiter",
    "write a function to detect a cycle in a graph",
    "implement merge sort",
    "write a function to flatten a nested dictionary",
    "implement a trie for autocomplete",
    "write a function to compute edit distance",
    "implement a producer-consumer queue with threading",
]


async def run_sequential(query: str) -> float:
    start = time.perf_counter()
    await researcher.run(query, {})
    await coder.run(query, {})
    return (time.perf_counter() - start) * 1000


async def run_parallel(query: str) -> float:
    start = time.perf_counter()
    await asyncio.gather(researcher.run(query, {}), coder.run(query, {}))
    return (time.perf_counter() - start) * 1000


async def main():
    seq_latencies = [await run_sequential(q) for q in QUERIES]
    par_latencies = [await run_parallel(q) for q in QUERIES]

    seq_mean = statistics.mean(seq_latencies)
    par_mean = statistics.mean(par_latencies)
    reduction = (seq_mean - par_mean) / seq_mean

    print(f"sequential mean (ms):   {seq_mean:.1f}")
    print(f"parallel mean (ms):     {par_mean:.1f}")
    print(f"latency reduction:      {reduction:.1%}")


if __name__ == "__main__":
    asyncio.run(main())

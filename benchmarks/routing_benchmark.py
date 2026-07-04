"""
Measures routing latency and accuracy in isolation (no LLM calls involved,
this is purely the TF-IDF + cosine similarity routing layer). Backs the
"<200ms routing overhead" and routing-accuracy claims.

Run: python -m benchmarks.routing_benchmark
"""
import statistics
import time

from app.orchestrator.router import SemanticRouter

# (query, expected_agent) — hand-labeled
LABELED_QUERIES = [
    ("write a function that reverses a linked list", "coder"),
    ("fix this bug in my code", "coder"),
    ("implement quicksort in python", "coder"),
    ("refactor this messy function", "coder"),
    ("debug why my server keeps crashing", "coder"),
    ("write unit tests for my stack class", "coder"),
    ("what caused World War 1", "researcher"),
    ("look up the history of the internet", "researcher"),
    ("compare python and rust", "researcher"),
    ("what is the capital of Australia", "researcher"),
    ("research the effects of inflation", "researcher"),
    ("explain how photosynthesis works", "researcher"),
    ("is this proof correct", "critic"),
    ("check this essay for logical errors", "critic"),
    ("review my argument for flaws", "critic"),
    ("summarize this article in three sentences", "summarizer"),
    ("give me a tldr of this document", "summarizer"),
    ("condense this report into bullet points", "summarizer"),
]


def main():
    router = SemanticRouter()

    # warm-up run isn't counted — first call after fit() sometimes pays a
    # one-off numpy/sklearn JIT-ish cost
    router.route("warm up query")

    latencies = []
    correct = 0
    for query, expected in LABELED_QUERIES:
        start = time.perf_counter()
        decision = router.route(query)
        latencies.append((time.perf_counter() - start) * 1000)
        if decision.agent.value == expected:
            correct += 1

    n = len(LABELED_QUERIES)
    print(f"labeled queries:     {n}")
    print(f"routing accuracy:    {correct}/{n} ({correct / n:.1%})")
    print(f"latency mean (ms):   {statistics.mean(latencies):.3f}")
    print(f"latency p95 (ms):    {sorted(latencies)[int(n * 0.95) - 1]:.3f}")
    print(f"latency max (ms):    {max(latencies):.3f}")


if __name__ == "__main__":
    main()

"""
Measures the multi-agent system on eval/tasks.py.

  completion_rate      : % of tasks whose final code passes the hidden tests
  first_try_rate       : % solved without the critic sending code back
                         (the gap between the two = what self-correction adds)
  parallel vs sequential latency on the same tasks

Each finished task is appended to reports/eval_results.jsonl immediately, and
finished tasks are skipped on the next run. The Groq free tier allows ~200k
tokens/day per model, so a full run can take two days: just run it again.

Run:  python -m eval.run_eval                       (both modes, all tasks)
      python -m eval.run_eval --modes parallel --limit 5
"""
import argparse
import asyncio
import json
import time
import uuid

import numpy as np

from eval.tasks import TASKS
from src.config import GROQ_MODEL, LLM_PROVIDER, ROOT
from src.graph import build_graph
from src.sandbox import run_python

RESULTS = ROOT / "reports" / "eval_results.jsonl"
MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else LLM_PROVIDER


async def run_task(app, task):
    start = time.perf_counter()
    state = await app.ainvoke({"session_id": uuid.uuid4().hex, "task": task["task"]})
    seconds = time.perf_counter() - start

    code = state.get("code", "")
    hidden = await asyncio.to_thread(run_python, code + "\n\n" + task["hidden_tests"])

    return {
        "id": task["id"],
        "completed": hidden["passed"],
        "revisions": state.get("revisions", 0),
        "seconds": seconds,
        "routing_ms": [r["ms"] for r in state.get("routes", [])],
        "timings": state.get("timings", {}),
        "error": "" if hidden["passed"] else hidden["stderr"][-300:],
    }


def load_results():
    if not RESULTS.exists():
        return []
    with open(RESULTS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(results):
    routing = [ms for r in results for ms in r["routing_ms"]]
    return {
        "tasks": len(results),
        "completion_rate": float(np.mean([r["completed"] for r in results])),
        "first_try_rate": float(np.mean([r["completed"] and r["revisions"] == 1 for r in results])),
        "avg_revisions": float(np.mean([r["revisions"] for r in results])),
        "mean_seconds": float(np.mean([r["seconds"] for r in results])),
        "p95_seconds": float(np.percentile([r["seconds"] for r in results], 95)),
        "routing_p95_ms": float(np.percentile(routing, 95)) if routing else 0.0,
    }


def write_report(results):
    report = {"model": MODEL}

    for mode in ["parallel", "sequential"]:
        rows = [r for r in results if r["mode"] == mode and r["model"] == MODEL]
        if rows:
            report[mode] = summarize(rows)

    # Latency comparison only on tasks finished in BOTH modes, or it isn't apples to apples
    by_mode = {m: {r["id"]: r for r in results if r["mode"] == m and r["model"] == MODEL} for m in ["parallel", "sequential"]}
    common = sorted(set(by_mode["parallel"]) & set(by_mode["sequential"]))
    if common:
        p = np.mean([by_mode["parallel"][i]["seconds"] for i in common])
        s = np.mean([by_mode["sequential"][i]["seconds"] for i in common])
        report["latency_comparison"] = {
            "tasks_in_both_modes": len(common),
            "parallel_mean_s": float(p),
            "sequential_mean_s": float(s),
            "latency_reduction": float(1 - p / s),
        }

    (ROOT / "reports" / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["parallel", "sequential"], choices=["parallel", "sequential"])
    args = parser.parse_args()

    tasks = TASKS[:args.limit] if args.limit else TASKS
    RESULTS.parent.mkdir(exist_ok=True)

    done = {(r["model"], r["mode"], r["id"]) for r in load_results()}

    for mode in args.modes:
        print(f"\n=== {mode} ({MODEL}) ===")
        app = build_graph(parallel=(mode == "parallel"))

        for task in tasks:
            if (MODEL, mode, task["id"]) in done:
                continue
            try:
                r = await run_task(app, task)
            except Exception as e:
                # Usually the daily token limit: stop cleanly, everything so far is saved
                print(f"  STOPPED at {task['id']}: {type(e).__name__}: {str(e)[:160]}")
                write_report(load_results())
                return

            row = {"model": MODEL, "mode": mode, **r}
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {'PASS' if r['completed'] else 'FAIL'} {r['id']:<16} {r['seconds']:.1f}s  revisions={r['revisions']}")

    write_report(load_results())


if __name__ == "__main__":
    asyncio.run(main())

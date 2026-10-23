"""
Concurrency test: N users connect over WebSocket at the same time and each runs a task.

Uses the fake LLM with a delay, so this measures the SERVER (async handling, memory,
streaming), not the LLM provider's speed or rate limits.

  terminal 1:  set LLM_PROVIDER=fake  FAKE_LLM_DELAY=0.5  SANDBOX=subprocess
               uvicorn src.api.main:app
  terminal 2:  python -m eval.load_test --users 50
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import websockets

BASE = "127.0.0.1:8000"


async def one_session(token, i):
    start = time.perf_counter()
    first_token = None

    async with websockets.connect(f"ws://{BASE}/ws?token={token}", open_timeout=30) as ws:
        await ws.send(json.dumps({"task": f"Explain and implement solve(x) #{i}"}))
        while True:
            event = json.loads(await ws.recv())
            if event["type"] in ("token", "node") and first_token is None:
                first_token = time.perf_counter() - start
            if event["type"] == "done":
                return {"ok": bool(event["answer"]), "seconds": time.perf_counter() - start, "first_event": first_token}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=50)
    args = parser.parse_args()

    token = httpx.post(f"http://{BASE}/token", json={"username": "demo", "password": "demo-password"}).json()["access_token"]

    start = time.perf_counter()
    results = await asyncio.gather(*[one_session(token, i) for i in range(args.users)], return_exceptions=True)
    wall = time.perf_counter() - start

    ok = [r for r in results if isinstance(r, dict) and r["ok"]]
    errors = [r for r in results if not isinstance(r, dict)]
    times = [r["seconds"] for r in ok]

    report = {
        "concurrent_users": args.users,
        "succeeded": len(ok),
        "failed": len(errors),
        "wall_seconds": round(wall, 2),
        "p50_seconds": round(float(np.percentile(times, 50)), 2) if times else None,
        "p95_seconds": round(float(np.percentile(times, 95)), 2) if times else None,
    }
    print(json.dumps(report, indent=2))

    out = Path(__file__).resolve().parents[1] / "reports" / f"load_test_{args.users}_users.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for e in errors[:3]:
        print("error:", repr(e))


if __name__ == "__main__":
    asyncio.run(main())

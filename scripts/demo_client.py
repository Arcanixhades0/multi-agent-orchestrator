"""
Quick manual test client — connects to the running server, gets a token,
sends a query over the websocket, and prints tokens as they stream in.

Usage:
    python scripts/demo_client.py "write a function to reverse a string"
"""
import asyncio
import sys

import httpx
import websockets
import json

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"


async def main(query: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}/token", params={"client_id": "demo-client"})
        token = resp.json()["access_token"]

    async with websockets.connect(f"{WS_URL}/ws/task?token={token}") as ws:
        await ws.send(json.dumps({"query": query}))
        current_agent = None
        async for raw in ws:
            msg = json.loads(raw)
            if msg["agent"] != current_agent:
                current_agent = msg["agent"]
                print(f"\n\n--- {current_agent} ---")
            if msg["delta"]:
                print(msg["delta"], end="", flush=True)
            if msg.get("meta", {}).get("final"):
                print("\n\n(done)")
                break


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "write a function to reverse a string"
    asyncio.run(main(q))

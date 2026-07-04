"""
Shared memory the agents read/write to during a run.

Why Redis instead of just passing a bigger dict through LangGraph state:
when we parallelize the researcher + coder (see graph.py), both branches
need to see partial output from each other as soon as it lands, not just at
the join. A plain in-memory dict works fine for a single process, but this
is meant to survive across worker restarts and let multiple FastAPI workers
share session state, so it goes through Redis with short TTLs instead.
"""
import json
import time
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import settings


class SharedMemory:
    def __init__(self, redis_url: str = settings.redis_url):
        self._redis: Optional[aioredis.Redis] = None
        self._redis_url = redis_url
        # fallback used only if redis is unreachable (e.g. local dev without
        # docker running) so the graph still works, just without cross-process
        # sharing
        self._local_fallback: dict[str, dict[str, Any]] = {}
        self._using_fallback = False

    async def connect(self):
        try:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
        except Exception:
            self._redis = None
            self._using_fallback = True

    def _key(self, session_id: str) -> str:
        return f"agent-mem:{session_id}"

    async def write(self, session_id: str, field: str, value: Any):
        payload = json.dumps(value) if not isinstance(value, str) else value
        if self._using_fallback or self._redis is None:
            self._local_fallback.setdefault(session_id, {})[field] = payload
            return
        await self._redis.hset(self._key(session_id), field, payload)
        await self._redis.expire(self._key(session_id), settings.redis_ttl_seconds)

    async def read(self, session_id: str, field: str) -> Optional[str]:
        if self._using_fallback or self._redis is None:
            return self._local_fallback.get(session_id, {}).get(field)
        return await self._redis.hget(self._key(session_id), field)

    async def read_all(self, session_id: str) -> dict:
        if self._using_fallback or self._redis is None:
            return dict(self._local_fallback.get(session_id, {}))
        return await self._redis.hgetall(self._key(session_id))

    async def append_trace(self, session_id: str, entry: dict):
        entry = {**entry, "ts": time.time()}
        key = f"{self._key(session_id)}:trace"
        if self._using_fallback or self._redis is None:
            self._local_fallback.setdefault(key, {}).setdefault("items", [])
            self._local_fallback[key]["items"].append(entry)
            return
        await self._redis.rpush(key, json.dumps(entry))
        await self._redis.expire(key, settings.redis_ttl_seconds)

    async def get_trace(self, session_id: str) -> list[dict]:
        key = f"{self._key(session_id)}:trace"
        if self._using_fallback or self._redis is None:
            return self._local_fallback.get(key, {}).get("items", [])
        raw = await self._redis.lrange(key, 0, -1)
        return [json.loads(r) for r in raw]

    async def close(self):
        if self._redis is not None:
            await self._redis.close()


# module-level singleton, initialized on app startup
shared_memory = SharedMemory()

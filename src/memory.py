"""
Shared memory that every agent reads from and writes to.

With REDIS_URL set, memory lives in Redis: it survives restarts and is shared
by every API worker process. Without it, a plain dict is used (one process only).

Layout: one Redis hash per session, e.g.
    session:abc123  ->  { "research:0": "...", "research:1": "...", "code": "..." }
"""
import json
import warnings

from src.config import REDIS_URL

TTL_SECONDS = 24 * 3600


class SharedMemory:

    def __init__(self, url=REDIS_URL):
        self._local = {}
        self._redis = None

        if url:
            import redis
            try:
                client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
                client.ping()
                self._redis = client
            except (redis.ConnectionError, redis.TimeoutError):
                warnings.warn(f"Redis not reachable at {url}: using in-process memory instead.")

    @property
    def backend(self):
        return "redis" if self._redis else "in-process"

    def write(self, session_id, key, value):
        data = json.dumps(value)
        if self._redis:
            name = f"session:{session_id}"
            # HSET is atomic per field, so parallel agents writing different keys never clash
            self._redis.hset(name, key, data)
            self._redis.expire(name, TTL_SECONDS)
        else:
            self._local.setdefault(session_id, {})[key] = data

    def read_all(self, session_id, prefix=""):
        if self._redis:
            raw = self._redis.hgetall(f"session:{session_id}")
        else:
            raw = self._local.get(session_id, {})

        return {k: json.loads(v) for k, v in sorted(raw.items()) if k.startswith(prefix)}

    def clear(self, session_id):
        if self._redis:
            self._redis.delete(f"session:{session_id}")
        else:
            self._local.pop(session_id, None)


memory = SharedMemory()

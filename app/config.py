"""
Central config. Everything is pulled from env vars so the same image can run
in dev (mock LLM, local redis) and prod (real provider, managed redis) without
code changes.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # LLM provider: "mock" | "openai" | "groq"
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_ttl_seconds: int = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-change-me")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

    max_self_correction_loops: int = int(os.getenv("MAX_SELF_CORRECTION_LOOPS", "2"))
    routing_confidence_floor: float = float(os.getenv("ROUTING_CONFIDENCE_FLOOR", "0.35"))

    max_ws_sessions: int = int(os.getenv("MAX_WS_SESSIONS", "50"))


settings = Settings()

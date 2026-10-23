import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# "groq" (fast, needs GROQ_API_KEY), "ollama" (local) or "fake" (tests / load tests, no model)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Empty = use in-process memory instead of Redis
REDIS_URL = os.getenv("REDIS_URL", "")

# How generated code is executed: "docker" (isolated) or "subprocess" (fallback, NOT isolated)
SANDBOX = os.getenv("SANDBOX", "docker")
SANDBOX_TIMEOUT = 15

MAX_REVISIONS = 3
MAX_SUBTASKS = 4

# HS256 needs a secret of at least 32 bytes. Set a random one in .env for anything real.
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret-change-me-0123456789abcdef")
JWT_TTL_MINUTES = 60
DEMO_USER = os.getenv("DEMO_USER", "demo")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo-password")

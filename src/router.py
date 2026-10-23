"""
Semantic router: picks which agent handles a subtask.

Each agent has a written description of what it is good at. We embed those
descriptions once, embed each subtask, and send the subtask to the agent whose
description is most similar (cosine similarity). No LLM call, so routing
costs milliseconds instead of seconds.
"""
import threading
import time

import numpy as np

from src.config import EMBEDDING_MODEL

# Profiles describe the ACTION (explain vs. write code), not the topic. An earlier version
# had "Program an algorithm" for the coder, and "Explain how Dijkstra's algorithm works"
# got routed to the coder because the shared noun "algorithm" dominated the similarity.
AGENT_PROFILES = {
    "researcher": [
        "Explain how something works",
        "Describe the definition and key properties of a concept",
        "Research background information and theory",
        "Find common edge cases, pitfalls and mistakes",
        "Compare approaches and describe trade-offs",
    ],
    "coder": [
        "Write a Python function",
        "Implement the solution in code",
        "Write code with unit tests and assert statements",
        "Fix a bug in a program",
        "Create a class with methods in Python",
    ],
}


class SemanticRouter:

    def __init__(self, model_name=EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.agents = []
        vectors = []

        for agent, examples in AGENT_PROFILES.items():
            for example in examples:
                self.agents.append(agent)
                vectors.append(example)

        self.profile_vectors = self.model.encode(vectors, normalize_embeddings=True)

    def route(self, subtask):
        start = time.perf_counter()

        v = self.model.encode([subtask], normalize_embeddings=True)[0]
        # Vectors are normalized, so a dot product IS cosine similarity
        scores = self.profile_vectors @ v
        best = int(np.argmax(scores))

        return {
            "agent": self.agents[best],
            "score": float(scores[best]),
            "ms": (time.perf_counter() - start) * 1000,
        }


_router = None
_lock = threading.Lock()


def get_router():
    """
    Load the embedding model once per process.

    The lock matters: without it, 50 users arriving at once each saw
    "_router is None" and loaded the model 50 times in parallel,
    which starved the server until every WebSocket timed out.
    """
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = SemanticRouter()
    return _router

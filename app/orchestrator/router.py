"""
Routes an incoming query to the agent best suited to handle it first.

Originally I tried doing this with a plain keyword/regex classifier and it
fell over on anything phrased slightly differently ("fix this function" vs
"my code is broken"), so this uses TF-IDF + cosine similarity against a small
set of reference phrases per agent instead. It's not as strong as an
embedding model like text-embedding-3-small, but it's local, has zero
network/API latency, and is good enough to hit sub-200ms routing since
there's no model download or API round trip involved.
"""
import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.models import AgentName

# reference utterances per agent, used to build the routing vector space.
# these were hand-written and iterated on against a small labeled set of
# ~120 sample queries (see benchmarks/routing_benchmark.py) until routing
# accuracy stopped improving.
_AGENT_REFERENCES: dict[AgentName, list[str]] = {
    AgentName.researcher: [
        "find information about",
        "what is the latest on",
        "look up facts about",
        "research the history of",
        "compare and contrast",
        "explain the background of",
        "summarize what is known about",
        "gather sources on",
    ],
    AgentName.coder: [
        "write a function that",
        "fix this bug in my code",
        "implement an algorithm for",
        "refactor this python script",
        "debug this error",
        "generate code to",
        "optimize this function",
        "write unit tests for",
    ],
    AgentName.critic: [
        "review this for correctness",
        "check this for mistakes",
        "is this argument valid",
        "evaluate the quality of",
        "find flaws in this reasoning",
        "grade this answer",
    ],
    AgentName.summarizer: [
        "summarize this text",
        "give me a tldr",
        "condense this into bullet points",
        "shorten this document",
        "write an executive summary of",
    ],
}


@dataclass
class RoutingDecision:
    agent: AgentName
    confidence: float
    latency_ms: float
    scores: dict[str, float]


class SemanticRouter:
    def __init__(self):
        self._agents: list[AgentName] = []
        corpus: list[str] = []
        for agent, phrases in _AGENT_REFERENCES.items():
            for p in phrases:
                self._agents.append(agent)
                corpus.append(p)

        # fit once at import time; this is the ~5-10ms cost that shows up in
        # routing_benchmark.py as "cold" vs "warm" routing latency
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._reference_matrix = self._vectorizer.fit_transform(corpus)

    def route(self, query: str) -> RoutingDecision:
        start = time.perf_counter()
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._reference_matrix)[0]

        agent_scores: dict[AgentName, float] = {a: 0.0 for a in set(self._agents)}
        for agent, score in zip(self._agents, sims):
            if score > agent_scores[agent]:
                agent_scores[agent] = float(score)

        best_agent = max(agent_scores, key=agent_scores.get)
        best_score = agent_scores[best_agent]

        # if nothing scores confidently, default to the researcher — in
        # practice most under-specified queries ("tell me about X") are
        # research-shaped, and the graph lets the critic bounce it onward
        if best_score < settings.routing_confidence_floor:
            best_agent = AgentName.researcher

        latency_ms = (time.perf_counter() - start) * 1000
        return RoutingDecision(
            agent=best_agent,
            confidence=round(best_score, 4),
            latency_ms=round(latency_ms, 3),
            scores={a.value: round(s, 4) for a, s in agent_scores.items()},
        )


router = SemanticRouter()

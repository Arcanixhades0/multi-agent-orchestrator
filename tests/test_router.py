from app.models import AgentName
from app.orchestrator.router import SemanticRouter


def test_router_routes_code_query_to_coder():
    r = SemanticRouter()
    decision = r.route("write a function that reverses a linked list")
    assert decision.agent == AgentName.coder


def test_router_routes_factual_query_to_researcher():
    r = SemanticRouter()
    decision = r.route("what is the history of the Roman empire")
    assert decision.agent == AgentName.researcher


def test_router_latency_is_fast():
    r = SemanticRouter()
    decision = r.route("fix this bug in my recursive function")
    # this is the number the routing overhead claim in the README is based
    # on — see benchmarks/routing_benchmark.py for the full distribution
    assert decision.latency_ms < 200


def test_router_confidence_bounds():
    r = SemanticRouter()
    decision = r.route("implement quicksort in python")
    assert 0.0 <= decision.confidence <= 1.0
    assert all(0.0 <= s <= 1.0 for s in decision.scores.values())


def test_router_low_confidence_defaults_to_researcher():
    r = SemanticRouter()
    decision = r.route("asdkjhasd qweqwe zzz")
    assert decision.agent == AgentName.researcher

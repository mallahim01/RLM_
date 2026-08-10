"""The three guards: depth, iterations, and total spend."""

from __future__ import annotations

from app.rlm.engine import RLMEngine
from app.trace import Tracer
from tests.conftest import DeepestFirstClient, make_settings


def _route_rounds_at_depth(result, depth: int) -> int:
    prefix = f"depth {depth} | iteration"
    return sum(1 for e in result.trace if e.kind == "route" and e.message.startswith(prefix))


def test_max_depth_is_enforced(erp_doc):
    client = DeepestFirstClient()
    engine = RLMEngine(client, make_settings(max_depth=1), Tracer(enabled=False))

    result = engine.answer(erp_doc, "What does SAP charge for indirect access?")

    assert result.stats.max_depth_reached == 1
    assert not any(e.kind == "route" and e.message.startswith("depth 2") for e in result.trace)


def test_at_max_depth_an_oversized_chunk_is_truncated_not_dropped(erp_doc):
    client = DeepestFirstClient()
    engine = RLMEngine(client, make_settings(max_depth=1), Tracer(enabled=False))

    result = engine.answer(erp_doc, "What does SAP charge for indirect access?")

    assert any(f.truncated for f in result.findings), "should read what fits rather than give up"
    truncations = [e for e in result.trace if e.kind == "limit" and "truncate" in e.message]
    assert truncations


def test_no_inspect_call_ever_exceeds_the_context_budget(erp_doc):
    settings = make_settings(max_depth=1)
    client = DeepestFirstClient()
    engine = RLMEngine(client, settings, Tracer(enabled=False))

    result = engine.answer(erp_doc, "What does SAP charge for indirect access?")

    for event in result.trace:
        if event.kind == "inspect":
            tokens = int(event.message.split("(")[1].split(" tok")[0])
            assert tokens <= settings.inspect_budget


def test_llm_call_budget_is_enforced_and_an_answer_still_comes_back(erp_doc):
    client = DeepestFirstClient()
    engine = RLMEngine(client, make_settings(max_llm_calls=4), Tracer(enabled=False))

    result = engine.answer(erp_doc, "What does SAP charge for indirect access?")

    assert client.call_count <= 4
    assert result.stats.llm_calls <= 4
    assert result.stopped_reason == "budget_exhausted"
    assert result.answer, "a truncated run must still produce something"


def test_max_iterations_caps_expansion_when_readers_keep_asking_for_more(flat_doc):
    # needs_more=True on every reading would loop forever without the cap.
    client = DeepestFirstClient(needs_more=True)
    engine = RLMEngine(
        client, make_settings(max_iterations=2, max_llm_calls=50), Tracer(enabled=False)
    )

    result = engine.answer(flat_doc, "What is in these sections?")

    assert _route_rounds_at_depth(result, 0) == 2
    assert result.stopped_reason == "max_iterations"


def test_a_satisfied_level_stops_early_instead_of_using_every_iteration(flat_doc):
    client = DeepestFirstClient(needs_more=False)
    engine = RLMEngine(client, make_settings(max_iterations=3), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is in these sections?")

    assert _route_rounds_at_depth(result, 0) == 1
    assert result.stopped_reason == "completed"


def test_a_level_stops_when_every_chunk_has_been_inspected(flat_doc):
    client = DeepestFirstClient(max_selections=6, needs_more=True)
    engine = RLMEngine(
        client, make_settings(max_iterations=5, max_selections_per_round=6, max_llm_calls=40),
        Tracer(enabled=False),
    )

    result = engine.answer(flat_doc, "What is in these sections?")

    assert result.stats.chunks_inspected <= result.stats.chunks_total
    assert _route_rounds_at_depth(result, 0) <= 2, "nothing left to route over after one sweep"

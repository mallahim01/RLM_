"""Bad model output is normal, not exceptional. The run must survive it."""

from __future__ import annotations

from app.llm.mock import PatternLLMClient
from app.rlm.engine import RLMEngine
from app.trace import Tracer
from tests.conftest import make_settings

_GOOD_INSPECT = {
    "found": True,
    "answer": "Section 1 talks about word1.",
    "evidence": ["word1 word1"],
    "confidence": 0.9,
    "needs_more": False,
    "suggested_chunk_ids": [],
}
_GOOD_SYNTHESIS = {"answer": "A synthesised answer.", "citations": ["Section 1"], "confidence": 0.8}


def _client(route_replies, **extra):
    """Build a pattern client with a scripted sequence of routing replies."""
    queue = list(route_replies)

    def route(_prompt):
        return queue.pop(0) if queue else {"selections": []}

    rules = [
        (r"^you are the router", route),
        (r"^you are a careful reader", extra.get("inspect", _GOOD_INSPECT)),
        (r"^you are a summariser", {"found": True, "answer": "merged", "confidence": 0.7}),
        (r"^you are the final reasoner", extra.get("synthesis", _GOOD_SYNTHESIS)),
    ]
    # PatternLLMClient matches against system+prompt, and callables are only
    # supported as the default, so route through the default for dynamic replies.
    return _DynamicPattern(rules)


class _DynamicPattern(PatternLLMClient):
    def _respond(self, reply, prompt, system):
        if callable(reply):
            reply = reply(prompt)
        return super()._respond(reply, prompt, system)


def test_a_malformed_routing_reply_is_repaired_and_the_run_continues(flat_doc):
    good_route = {"selections": [{"chunk_id": "c1", "sub_question": "what?"}]}
    client = _client(["this is not json", good_route])
    engine = RLMEngine(client, make_settings(max_iterations=1), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is here?")

    assert result.answer == "A synthesised answer."
    assert result.stats.chunks_inspected == 1


def test_two_bad_routing_replies_end_the_level_cleanly(flat_doc):
    client = _client(["not json", "still not json"])
    engine = RLMEngine(client, make_settings(max_iterations=1), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is here?")

    assert result.stopped_reason == "router_empty"
    assert result.answer  # a message, not a traceback
    assert any(e.kind == "error" for e in result.trace)


def test_hallucinated_chunk_ids_are_dropped_and_traced(flat_doc):
    route = {
        "selections": [
            {"chunk_id": "c99", "sub_question": "invented"},
            {"chunk_id": "c1", "sub_question": "real"},
        ]
    }
    client = _client([route])
    engine = RLMEngine(client, make_settings(max_iterations=1), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is here?")

    assert result.stats.chunks_inspected == 1
    assert [f.chunk_id for f in result.findings] == ["c1"]
    assert any("dropped unknown section id 'c99'" in e.message for e in result.trace)


def test_an_all_invalid_routing_reply_stops_gracefully(flat_doc):
    client = _client([{"selections": [{"chunk_id": "nope", "sub_question": "x"}]}])
    engine = RLMEngine(client, make_settings(max_iterations=1), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is here?")

    assert result.stopped_reason == "router_empty"
    assert result.stats.chunks_inspected == 0


def test_a_failed_synthesis_falls_back_to_assembling_the_findings(flat_doc):
    route = {"selections": [{"chunk_id": "c1", "sub_question": "what?"}]}
    client = _client([route], synthesis="not json either")
    engine = RLMEngine(client, make_settings(max_iterations=1), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is here?")

    assert "Section 1 talks about word1." in result.answer
    assert result.citations


def test_a_reader_returning_nonsense_types_is_coerced_not_fatal(flat_doc):
    route = {"selections": [{"chunk_id": "c1", "sub_question": "what?"}]}
    weird = {
        "found": "yes",
        "answer": 12345,
        "evidence": "a bare string, not a list",
        "confidence": "very high",
        "needs_more": None,
        "suggested_chunk_ids": ["c1", "c404"],
    }
    client = _client([route], inspect=weird)
    engine = RLMEngine(client, make_settings(max_iterations=1), Tracer(enabled=False))

    result = engine.answer(flat_doc, "What is here?")

    finding = result.findings[0]
    assert finding.found is True
    assert finding.answer == "12345"
    assert finding.evidence == ["a bare string, not a list"]
    assert finding.confidence == 0.5  # unparseable -> default, not a crash
    assert finding.suggested_chunk_ids == ["c1"]  # c404 filtered out

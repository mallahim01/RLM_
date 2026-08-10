"""The base case: when the document fits, there is nothing to recurse over."""

from __future__ import annotations

from app.llm.mock import ScriptedLLMClient
from app.rlm.engine import RLMEngine
from app.trace import Tracer
from tests.conftest import make_settings


def _kinds(result) -> list[str]:
    return [event.kind for event in result.trace]


def test_small_document_is_answered_in_exactly_one_call(tiny_doc):
    client = ScriptedLLMClient([{"answer": "Alpha, beta and gamma.", "citations": ["Alpha"]}])
    engine = RLMEngine(client, make_settings(max_context_tokens=5000), Tracer(enabled=False))

    result = engine.answer(tiny_doc, "What is in this document?")

    assert client.call_count == 1
    assert result.stats.llm_calls == 1
    assert result.answer == "Alpha, beta and gamma."
    assert result.stats.route_calls == 0
    assert "basecase" in _kinds(result)
    assert "route" not in _kinds(result)


def test_base_case_reports_reading_the_whole_document(tiny_doc):
    client = ScriptedLLMClient([{"answer": "ok"}])
    engine = RLMEngine(client, make_settings(max_context_tokens=5000), Tracer(enabled=False))

    result = engine.answer(tiny_doc, "q")

    assert result.stats.context_efficiency == 1.0
    assert result.stats.document_tokens_read == result.stats.document_tokens


def test_the_window_size_alone_decides_which_path_is_taken(flat_doc, deepest_first):  # noqa: D103
    """Same document, same code: the context budget picks the strategy."""
    narrow = RLMEngine(deepest_first, make_settings(), Tracer(enabled=False))
    narrow_result = narrow.answer(flat_doc, "What is in this document?")

    assert "basecase" not in _kinds(narrow_result)
    assert "route" in _kinds(narrow_result)
    assert narrow_result.stats.context_efficiency < 1.0

    wide = RLMEngine(
        ScriptedLLMClient([{"answer": "everything"}]),
        make_settings(max_context_tokens=60_000),
        Tracer(enabled=False),
    )
    wide_result = wide.answer(flat_doc, "What is in this document?")

    assert "basecase" in _kinds(wide_result)
    assert wide_result.stats.llm_calls == 1


def test_unusable_reply_in_the_base_case_does_not_crash(tiny_doc):
    client = ScriptedLLMClient(["not json", "still not json"])
    engine = RLMEngine(client, make_settings(max_context_tokens=5000), Tracer(enabled=False))

    result = engine.answer(tiny_doc, "q")

    assert result.stopped_reason == "error"
    assert result.answer  # a message, not an exception

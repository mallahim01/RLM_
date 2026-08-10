"""The recursive path: descend, read leaves, fold results back up."""

from __future__ import annotations

from app.llm.mock import DemoLLMClient
from app.rlm.engine import RLMEngine
from app.rlm.tokens import count_tokens
from app.trace import Tracer
from tests.conftest import DeepestFirstClient, make_settings


def test_engine_descends_through_the_tree_and_synthesises(erp_doc, deepest_first):
    engine = RLMEngine(deepest_first, make_settings(), Tracer(enabled=False))

    result = engine.answer(erp_doc, "What does SAP charge for indirect access?")

    assert result.stats.max_depth_reached >= 2, "should reach the paragraph level"
    assert result.answer
    assert result.citations
    assert result.findings
    kinds = {event.kind for event in result.trace}
    assert {"chunk", "index", "route", "recurse", "inspect", "synthesize"} <= kinds


def test_only_a_fraction_of_the_document_is_ever_read(erp_doc):
    engine = RLMEngine(DemoLLMClient(), make_settings(), Tracer(enabled=False))

    result = engine.answer(erp_doc, "Which platform is most open to third-party agents?")

    assert 0 < result.stats.document_tokens_read < result.stats.document_tokens
    assert result.stats.context_efficiency < 0.5, "the whole point: read a little, answer well"
    assert result.stats.chunks_inspected < result.stats.chunks_total


def test_the_router_sees_only_an_index_never_a_section_body(erp_doc, deepest_first):
    """The router gets headings, sizes and a 140-character preview -- no more."""
    settings = make_settings()
    engine = RLMEngine(deepest_first, settings, Tracer(enabled=False))
    engine.answer(erp_doc, "What does SAP charge for indirect access?")

    # A sentence from deep inside a section, well past any preview window.
    body = next(s.text for s in erp_doc.sections if "Joule" in s.text)
    buried = body[600:700]
    assert buried.strip(), "fixture assumption: the SAP section is long"

    router_calls = [
        call for call in deepest_first.calls if "router" in call["system"].splitlines()[0].lower()
    ]
    assert router_calls
    for call in router_calls:
        assert buried not in call["prompt"], "routing must never receive section bodies"
        assert count_tokens(call["prompt"]) < settings.max_index_tokens + 400


def test_sub_findings_are_compressed_before_travelling_upward(erp_doc):
    client = DeepestFirstClient(max_selections=2)
    engine = RLMEngine(client, make_settings(), Tracer(enabled=False))

    result = engine.answer(erp_doc, "Compare SAP and Odoo on external agent access.")

    compressions = [e for e in result.trace if e.kind == "compress"]
    assert compressions, "a descended branch should fold its findings into one"
    # Compressed findings are attributed to the parent, not the leaf part.
    assert all("part " not in f.heading_path for f in result.findings)


def test_index_stays_small_at_every_level(erp_doc, deepest_first):
    engine = RLMEngine(deepest_first, make_settings(), Tracer(enabled=False))
    result = engine.answer(erp_doc, "What does SAP charge for indirect access?")

    for event in result.trace:
        if event.kind == "index":
            assert event.data["index_tokens"] <= 800

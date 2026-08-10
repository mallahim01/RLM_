"""The one test that costs money.

Skipped unless a key is configured, and excluded from the default run by the
``integration`` marker in pyproject.toml. Run it deliberately:

    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from app.config import load_settings
from app.rlm.engine import RLMEngine
from app.trace import Tracer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="set OPENAI_API_KEY in .env to run"
    ),
]


@pytest.fixture
def engine(erp_doc):
    settings = load_settings(tokenizer="auto")
    from app.llm import build_openai_client

    client = build_openai_client(settings.openai_api_key, settings.model, settings.request_timeout)
    return RLMEngine(client, settings, Tracer(enabled=True))


def test_real_model_answers_from_a_deeply_nested_section(engine, erp_doc):
    """The answer lives inside `Details by Platform > Oracle`, two levels down."""
    result = engine.answer(
        erp_doc, "How many free AI Units per month does Oracle include, and with which product?"
    )

    assert result.answer.strip()
    assert result.citations
    assert result.stats.llm_calls <= engine.settings.max_llm_calls
    assert result.stats.max_depth_reached >= 1, "this fact requires descending"
    assert "20,000" in result.answer or "20000" in result.answer


def test_real_model_never_reads_the_whole_document(engine, erp_doc):
    result = engine.answer(erp_doc, "Which platform is hardest for third-party agent tool-calling?")

    assert result.stats.document_tokens_read < result.stats.document_tokens
    assert result.stats.context_efficiency < 0.8
    assert result.answer.strip()

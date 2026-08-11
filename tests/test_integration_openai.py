"""The one test that costs money.

Skipped unless a key is configured, and excluded from the default run by the
``integration`` marker in pyproject.toml. Run it deliberately:

    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from app.config import PROVIDERS, load_settings
from app.rlm.engine import RLMEngine
from app.trace import Tracer

_ANY_KEY = any(os.getenv(p.env_key, "").strip() for p in PROVIDERS.values())
_KEY_NAMES = " or ".join(sorted(p.env_key for p in PROVIDERS.values()))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ANY_KEY, reason=f"set {_KEY_NAMES} in .env to run"),
]


@pytest.fixture
def engine(erp_doc):
    settings = load_settings(tokenizer="auto")
    from app.llm import build_client

    return RLMEngine(build_client(settings), settings, Tracer(enabled=True))


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

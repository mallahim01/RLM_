"""Shared fixtures. The default suite is entirely offline and free.

Token counts are forced onto the chars/4 heuristic so assertions are exact and
nothing ever reaches the network for a BPE table.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["RLM_TOKENIZER"] = "heuristic"

from app.config import Settings, load_settings  # noqa: E402
from app.llm.mock import DemoLLMClient  # noqa: E402
from app.loaders import load_document  # noqa: E402
from app.models import Document, Section  # noqa: E402
from app.rlm import tokens as tokens_module  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERP_DOC = PROJECT_ROOT / "test_files" / "erp-ai-capabilities.md"


@pytest.fixture(autouse=True)
def deterministic_tokenizer(monkeypatch: pytest.MonkeyPatch):
    """Pin the heuristic tokenizer for every test."""
    monkeypatch.setenv("RLM_TOKENIZER", "heuristic")
    tokens_module.reset_tokenizer_cache()
    yield
    tokens_module.reset_tokenizer_cache()


@pytest.fixture(autouse=True)
def isolate_rlm_logger():
    """Drop the `rlm` logger's handlers after each test.

    Any test that runs the CLI installs a handler pointing at pytest's captured
    stderr. That stream is closed when the test ends, so a later test that logs
    would write to a dead file object.
    """
    import logging

    yield
    logging.getLogger("rlm").handlers.clear()


@pytest.fixture
def settings() -> Settings:
    """Defaults, with a placeholder key so nothing needs a real one."""
    return load_settings(tokenizer="heuristic", api_key="test-key-not-real")


def make_settings(**overrides) -> Settings:
    overrides.setdefault("tokenizer", "heuristic")
    overrides.setdefault("api_key", "test-key-not-real")
    return load_settings(**overrides)


def _paragraph(word: str, count: int) -> str:
    return " ".join(f"{word}{i}" for i in range(count)) + "."


@pytest.fixture
def tiny_doc() -> Document:
    """Small enough to fit any sane working budget -> exercises the base case."""
    sections = [
        Section(("Alpha",), _paragraph("alpha", 20), 1),
        Section(("Beta",), _paragraph("beta", 20), 1),
        Section(("Gamma",), _paragraph("gamma", 20), 1),
    ]
    return Document(path="tiny.md", title="Tiny", sections=sections, total_tokens=200)


@pytest.fixture
def flat_doc() -> Document:
    """Six sibling sections, none big enough to split -> recursion stays at depth 0."""
    sections = [
        Section((f"Section {i}",), "\n\n".join(_paragraph(f"word{i}", 120) for _ in range(2)), 1)
        for i in range(1, 7)
    ]
    return Document(path="flat.md", title="Flat", sections=sections, total_tokens=6000)


@pytest.fixture
def nested_doc() -> Document:
    """An H1/H2/H3 shape with one oversized branch, for chunker tests."""
    sections = [
        Section(("Root",), "Intro paragraph that is long enough to matter. " * 6, 1),
        Section(("Root", "Small"), _paragraph("small", 30), 2),
        Section(("Root", "Big"), "", 2),
        Section(("Root", "Big", "Huge"), "\n\n".join(_paragraph("huge", 200) for _ in range(6)), 3),
        Section(("Root", "Tail"), _paragraph("tail", 25), 2),
    ]
    total = sum(len(s.text) // 4 for s in sections)
    return Document(path="nested.md", title="Root", sections=sections, total_tokens=total)


@pytest.fixture(scope="session")
def erp_doc() -> Document:
    """The real sample document."""
    if not ERP_DOC.exists():  # pragma: no cover - the file ships with the repo
        pytest.skip(f"missing sample document: {ERP_DOC}")
    return load_document(ERP_DOC)


class DeepestFirstClient(DemoLLMClient):
    """Demo client whose router always picks the largest section available.

    Deterministic and level-aware, which makes it the right driver for tests
    about recursion depth: it walks straight down the biggest branch.
    """

    def __init__(self, max_selections: int = 1, needs_more: bool = False) -> None:
        super().__init__(max_selections=max_selections)
        self.needs_more = needs_more

    def _route(self, prompt: str) -> dict:
        from app.llm.mock import _INDEX_ENTRY

        entries = [(int(m.group("tokens")), m.group("id")) for m in _INDEX_ENTRY.finditer(prompt)]
        entries.sort(reverse=True)
        question = self._question(prompt)
        return {
            "reasoning": "largest section first",
            "selections": [
                {"chunk_id": chunk_id, "sub_question": question, "why": f"{size} tok"}
                for size, chunk_id in entries[: self.max_selections]
            ],
        }

    def _inspect(self, prompt: str) -> dict:
        payload = super()._inspect(prompt)
        payload["needs_more"] = self.needs_more
        return payload


@pytest.fixture
def deepest_first() -> DeepestFirstClient:
    return DeepestFirstClient()

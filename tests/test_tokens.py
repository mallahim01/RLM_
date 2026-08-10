from __future__ import annotations

import pytest

from app.rlm import tokens as tokens_module
from app.rlm.tokens import count_tokens, estimate_tokens, truncate_to_tokens


def test_empty_text_costs_nothing():
    assert count_tokens("") == 0
    assert estimate_tokens("") == 0


def test_heuristic_is_chars_over_four():
    text = "a" * 400
    assert estimate_tokens(text) == 100
    assert count_tokens(text) == 100  # RLM_TOKENIZER=heuristic is pinned in conftest


def test_short_text_never_rounds_to_zero():
    assert estimate_tokens("hi") == 1


def test_broken_tokenizer_falls_back_instead_of_raising(monkeypatch: pytest.MonkeyPatch):
    class Exploding:
        def encode(self, text):
            raise RuntimeError("BPE table unavailable")

    monkeypatch.setattr(tokens_module, "_get_encoder", lambda: Exploding())
    assert count_tokens("a" * 400) == 100


def test_truncate_respects_budget_and_word_boundaries():
    text = " ".join(f"word{i}" for i in range(300))
    cut = truncate_to_tokens(text, 50)
    assert count_tokens(cut) <= 50
    assert not cut.endswith(" ")
    # The final token must be a whole word from the original text.
    assert cut.split()[-1] in text.split()


def test_truncate_is_a_no_op_when_it_already_fits():
    assert truncate_to_tokens("short text", 100) == "short text"
    assert truncate_to_tokens("anything", 0) == ""

"""Token counting.

This project is *about* enforcing a token budget, so it should measure honestly:
``tiktoken`` when it is available, a chars/4 estimate when it is not.

tiktoken fetches its BPE table from the network on first use, which would turn a
fresh clone with no connectivity into a hang. So the first failure latches the
heuristic permanently and logs once, and ``RLM_TOKENIZER=heuristic`` forces the
estimate outright (which is what the test suite does, for determinism).
"""

from __future__ import annotations

import logging
import os
import threading

_LOG = logging.getLogger("rlm")

_CHARS_PER_TOKEN = 4

_lock = threading.Lock()
_encoder = None  # lazily resolved tiktoken encoding
_resolved = False  # True once we have decided between tiktoken and the heuristic


def estimate_tokens(text: str) -> int:
    """Pure chars/4 estimate. Always available, never touches the network."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _heuristic_forced() -> bool:
    return os.getenv("RLM_TOKENIZER", "auto").strip().lower() == "heuristic"


def _get_encoder():
    """Resolve the tokenizer once. Returns None to mean "use the heuristic"."""
    global _encoder, _resolved
    if _resolved:
        return _encoder
    with _lock:
        if _resolved:
            return _encoder
        if _heuristic_forced():
            _encoder = None
        else:
            try:
                import tiktoken

                _encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:  # network, missing package, anything
                _LOG.info(
                    "[RLM] tokenizer: tiktoken unavailable, using chars/4 estimate"
                )
                _encoder = None
        _resolved = True
    return _encoder


def count_tokens(text: str) -> int:
    """Best available token count. Never raises."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is None:
        return estimate_tokens(text)
    try:
        return len(encoder.encode(text))
    except Exception:
        return estimate_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` down to ``max_tokens``, snapping back to a whitespace boundary."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    # Binary search on characters: cheaper and simpler than decoding token ids,
    # and works identically under both the real tokenizer and the heuristic.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    cut = text[:low]
    space = cut.rfind(" ")
    if space > len(cut) * 0.8:  # only snap back if we are not throwing much away
        cut = cut[:space]
    return cut.rstrip()


def reset_tokenizer_cache() -> None:
    """Test hook: forget the resolved tokenizer so env changes take effect."""
    global _encoder, _resolved
    with _lock:
        _encoder = None
        _resolved = False

"""Lexical scoring, used to narrow a wide level before the router sees it.

Deliberately not a vector store. BM25 over the candidates at the current level
is a few dozen lines, needs no dependencies, no embedding calls and no index to
keep in sync -- and for heading-structured documents it is remarkably effective,
because the heading path is itself a dense summary. Weighting heading terms
higher is the single highest-value tweak here: a question mentioning "SAP"
surfaces the SAP branch with zero LLM calls.

Add embeddings when this demonstrably stops being enough, not before.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.models import Chunk

_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")

_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by
    with without into over under again further is are was were be been being am do does
    did doing have has had having i you he she it we they me him her them my your our
    their as so such no nor not only own same too very can will just should now what
    which who whom when where why how about across after all also any because before
    between both during each few more most other some there here up down out off
    """.split()
)

# How many times heading-path terms are counted relative to body terms.
HEADING_WEIGHT = 3


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords and single characters removed."""
    return [
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 1 and word not in _STOPWORDS
    ]


def _chunk_terms(chunk: Chunk) -> list[str]:
    heading = tokenize(chunk.path_str)
    return heading * HEADING_WEIGHT + tokenize(chunk.full_text())


def bm25_scores(
    query: str,
    chunks: list[Chunk],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[Chunk, float]]:
    """Score each chunk against the query, highest first.

    The corpus is the candidate list at the current recursion level, so IDF is
    computed locally -- which is what we want: "SAP" is discriminative among four
    platform sections even though it is common in the document as a whole.
    """
    query_terms = set(tokenize(query))
    if not chunks:
        return []
    if not query_terms:
        return [(c, 0.0) for c in chunks]

    docs = [Counter(_chunk_terms(c)) for c in chunks]
    lengths = [sum(d.values()) or 1 for d in docs]
    avg_length = sum(lengths) / len(lengths)
    n_docs = len(docs)

    scored: list[tuple[Chunk, float]] = []
    for chunk, doc, length in zip(chunks, docs, lengths):
        score = 0.0
        for term in query_terms:
            freq = doc.get(term, 0)
            if not freq:
                continue
            containing = sum(1 for d in docs if term in d)
            idf = math.log(1 + (n_docs - containing + 0.5) / (containing + 0.5))
            score += idf * (freq * (k1 + 1)) / (
                freq + k1 * (1 - b + b * length / avg_length)
            )
        scored.append((chunk, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def rank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Reorder chunks by relevance without dropping any."""
    return [chunk for chunk, _ in bm25_scores(query, chunks)]


def prefilter(
    query: str,
    chunks: list[Chunk],
    threshold: int,
    keep: int,
    pinned: set[str] | None = None,
) -> tuple[list[Chunk], bool]:
    """Narrow a wide level to the best ``keep`` candidates.

    Returns ``(candidates, fired)``. Below ``threshold`` nothing is dropped and
    the list is merely reordered -- ranking is cheap and improves routing either
    way. Chunks named in ``pinned`` (for instance ones a previous round asked to
    look at) are never dropped.
    """
    scored = bm25_scores(query, chunks)
    if len(chunks) <= threshold:
        return [chunk for chunk, _ in scored], False

    pinned = pinned or set()
    kept: list[Chunk] = []
    for chunk, score in scored:
        if len(kept) < keep or chunk.id in pinned or (score > 0 and len(kept) < keep * 2):
            kept.append(chunk)
    return kept, True

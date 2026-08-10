from __future__ import annotations

from app.rlm.chunker import build_chunk_tree
from app.rlm.retrieval import bm25_scores, prefilter, rank, tokenize


def test_tokenize_drops_stopwords_and_single_characters():
    assert tokenize("The cost of a SAP licence is high") == ["cost", "sap", "licence", "high"]


def test_heading_terms_outweigh_body_terms(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    platforms = next(c for c in tree if c.path_str == "Details by Platform")

    ranked = rank("What does SAP charge for indirect access licensing?", platforms.children)
    assert ranked[0].path_str.endswith("1) SAP (S/4HANA / SAP Business AI Platform)")


def test_scores_are_sorted_high_to_low(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    scored = bm25_scores("Odoo pricing and IAP credits", tree)
    values = [score for _chunk, score in scored]
    assert values == sorted(values, reverse=True)
    assert values[0] > 0


def test_empty_query_is_harmless(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    scored = bm25_scores("", tree)
    assert len(scored) == len(tree)
    assert all(score == 0.0 for _chunk, score in scored)


def test_empty_corpus_is_harmless():
    assert bm25_scores("anything", []) == []


def test_prefilter_only_drops_above_the_threshold(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    question = "Which platform is most open to external agents?"

    kept, fired = prefilter(question, tree, threshold=12, keep=12)
    assert not fired
    assert len(kept) == len(tree), "below the threshold nothing is dropped, only reordered"

    kept, fired = prefilter(question, tree, threshold=3, keep=3)
    assert fired
    assert 3 <= len(kept) < len(tree)


def test_prefilter_never_drops_a_pinned_chunk(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    last = tree[-1]
    kept, fired = prefilter("Odoo pricing", tree, threshold=2, keep=1, pinned={last.id})
    assert fired
    assert last.id in {c.id for c in kept}

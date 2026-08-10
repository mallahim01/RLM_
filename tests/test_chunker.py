from __future__ import annotations

import re

from app.rlm.chunker import (
    build_chunk_tree,
    index_chunks,
    split_text,
    total_tokens,
    tree_depth,
)
from app.rlm.tokens import count_tokens


def test_ids_are_deterministic_and_hierarchical(nested_doc):
    first = build_chunk_tree(nested_doc.sections, target_tokens=200, overlap_tokens=20)
    second = build_chunk_tree(nested_doc.sections, target_tokens=200, overlap_tokens=20)
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.id for c in first] == ["c1", "c2", "c3", "c4"]

    by_id = index_chunks(first)
    assert "c3.1" in by_id
    assert by_id["c3.1"].parent_id == "c3"
    assert by_id["c3.1"].depth == 1


def test_a_lone_wrapping_h1_is_unwrapped(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    # Without unwrapping this would be a single-entry index, useless for routing.
    assert len(tree) == 7
    headings = [c.path_str for c in tree]
    assert "TL;DR" in headings
    assert "Details by Platform" in headings
    # The redundant document title is stripped from descendant paths.
    assert not any(h.startswith("Pre-Sales Dossier") for h in headings[1:])


def test_subtree_that_fits_is_collapsed_into_one_leaf(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    comparison = next(c for c in tree if c.path_str == "Cross-Platform Comparison")
    # It has three H3 children in the source, but the whole subtree fits.
    assert comparison.is_leaf
    assert comparison.token_count <= 600
    assert "Feasibility ranking" in comparison.text


def test_oversized_section_hard_splits_by_paragraph(nested_doc):
    tree = build_chunk_tree(nested_doc.sections, target_tokens=200, overlap_tokens=0)
    by_id = index_chunks(tree)
    huge = next(c for c in by_id.values() if c.path_str.endswith("Huge"))
    assert len(huge.children) >= 4
    for child in huge.children:
        assert count_tokens(child.text) <= 200
        assert re.match(r"part \d+/\d+", child.heading_path[-1])


def test_splitting_never_cuts_mid_word():
    words = [f"word{i}" for i in range(2000)]
    parts = split_text(" ".join(words), target=100, overlap=0)
    assert len(parts) > 1
    for part in parts:
        for token in part.split():
            assert token in words, f"{token!r} looks like a word cut in half"


def test_overlap_carries_the_previous_tail_forward():
    paragraphs = [f"Paragraph {i} sentence one. Paragraph {i} sentence two." for i in range(40)]
    text = "\n\n".join(paragraphs)
    with_overlap = split_text(text, target=120, overlap=30)
    assert len(with_overlap) > 1
    for previous, current in zip(with_overlap, with_overlap[1:]):
        head = current.split("\n\n")[0]
        assert head in previous, "each part should begin with the tail of the last"
        assert count_tokens(current) <= 120


def test_overlap_does_not_cross_heading_boundaries(nested_doc):
    tree = build_chunk_tree(nested_doc.sections, target_tokens=200, overlap_tokens=40)
    small = next(c for c in tree if c.path_str.endswith("Small"))
    tail = next(c for c in tree if c.path_str.endswith("Tail"))
    assert "small" in small.text
    assert "small" not in tail.text


def test_a_single_unsplittable_sentence_still_terminates():
    text = "word " * 5000  # no sentence or paragraph boundaries at all
    parts = split_text(text.strip(), target=100, overlap=0)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 100 for p in parts)


def test_tree_shape_of_the_real_document(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    assert tree_depth(tree) == 3
    assert len(index_chunks(tree)) > 20
    # Chunking adds overlap, so the tree is slightly larger than the source.
    assert total_tokens(tree) >= erp_doc.total_tokens

    platforms = next(c for c in tree if c.path_str == "Details by Platform")
    assert len(platforms.children) == 4
    assert all(child.children for child in platforms.children)

"""Merging several documents into one corpus.

The point is that merging costs the engine nothing new: each file becomes a
top-level branch, so choosing *which document* to look in is the same routing
decision as choosing which section -- one level higher.
"""

from __future__ import annotations

import pytest

from app.loaders import load_documents, merge_documents
from app.models import Document, Section
from app.rlm.chunker import build_chunk_tree
from app.rlm.context import build_index
from app.rlm.tokens import count_tokens
from tests.conftest import ERP_DOC, PROJECT_ROOT

DOCX = PROJECT_ROOT / "test_files" / "knowledge-product-pakistan.docx"


def _doc(title: str, headings: list[str]) -> Document:
    sections = [Section((h,), f"body of {h} " * 40, 1) for h in headings]
    return Document(
        path=f"{title}.md",
        title=title,
        sections=sections,
        total_tokens=sum(count_tokens(s.text) for s in sections),
    )


def test_each_document_becomes_its_own_branch():
    merged = merge_documents([_doc("Alpha", ["One", "Two"]), _doc("Beta", ["Three"])])

    assert [s.heading_path for s in merged.sections] == [
        ("Alpha", "One"),
        ("Alpha", "Two"),
        ("Beta", "Three"),
    ]
    assert all(s.level == 2 for s in merged.sections)


def test_merging_preserves_every_token():
    a, b = _doc("Alpha", ["One", "Two"]), _doc("Beta", ["Three"])
    merged = merge_documents([a, b])

    assert merged.total_tokens == a.total_tokens + b.total_tokens
    body = " ".join(s.text for s in merged.sections)
    for heading in ("One", "Two", "Three"):
        assert f"body of {heading}" in body


def test_a_flat_single_section_document_keeps_its_heading():
    """It shares its 'root' trivially; that must not be mistaken for a tree."""
    merged = merge_documents([_doc("Alpha", ["One"]), _doc("Beta", ["Three"])])

    assert [s.heading_path for s in merged.sections] == [("Alpha", "One"), ("Beta", "Three")]


def test_a_document_with_a_real_single_root_is_not_double_prefixed():
    """Markdown under one H1: prefixing would repeat the title on every path."""
    rooted = Document(
        path="rooted.md",
        title="The Title",
        sections=[
            Section(("The Title",), "intro", 1),
            Section(("The Title", "First"), "a", 2),
            Section(("The Title", "Second"), "b", 2),
        ],
        total_tokens=10,
    )
    merged = merge_documents([rooted, _doc("Beta", ["Three"])])

    assert [s.heading_path for s in merged.sections] == [
        ("The Title",),
        ("The Title", "First"),
        ("The Title", "Second"),
        ("Beta", "Three"),
    ]
    assert not any(p.count("The Title") > 1 for s in merged.sections for p in [str(s.heading_path)])


def test_a_single_document_is_returned_untouched():
    only = _doc("Alpha", ["One"])
    assert merge_documents([only]) is only


def test_merging_nothing_is_an_error():
    with pytest.raises(ValueError):
        merge_documents([])
    with pytest.raises(ValueError):
        load_documents([])


def test_the_index_lets_the_router_choose_a_document_first():
    merged = merge_documents(
        [_doc("Alpha", ["One", "Two"]), _doc("Beta", ["Three"]), _doc("Gamma", ["Four"])]
    )
    tree = build_chunk_tree(merged.sections, target_tokens=100, overlap_tokens=0)
    index = build_index(tree, max_tokens=800)

    assert len(tree) == 3, "one top-level entry per source document"
    for title in ("Alpha", "Beta", "Gamma"):
        assert title in index
    assert count_tokens(index) <= 800


def test_merging_real_files_of_different_formats():
    """Markdown plus docx: different loaders, one corpus."""
    if not (ERP_DOC.exists() and DOCX.exists()):  # pragma: no cover
        pytest.skip("sample files not present")

    merged = load_documents([ERP_DOC, DOCX])
    tree = build_chunk_tree(merged.sections, 600, 60)

    assert len(tree) == 2
    assert merged.total_tokens > 8_000
    # Every section is reachable under exactly one document branch.
    roots = {c.heading_path[0] for c in tree}
    assert len(roots) == 2
    index = build_index(tree, 800)
    assert count_tokens(index) <= 800

from __future__ import annotations

from app.models import Chunk
from app.rlm.chunker import build_chunk_tree
from app.rlm.context import build_index, format_findings, render_index
from app.rlm.tokens import count_tokens


def test_index_of_the_real_document_is_a_tiny_fraction_of_it(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    index = build_index(tree, max_tokens=800)
    index_tokens = count_tokens(index)

    assert index_tokens <= 800
    # This ratio is the entire argument for the approach.
    assert index_tokens < erp_doc.total_tokens * 0.10
    for chunk in tree:
        assert f"[{chunk.id}]" in index


def test_index_never_exceeds_its_budget_however_many_chunks():
    chunks = [
        Chunk(
            id=f"c{i}",
            heading_path=(f"Section number {i} with a fairly long heading",),
            text="body text " * 200,
            token_count=500,
            depth=0,
        )
        for i in range(200)
    ]
    index = build_index(chunks, max_tokens=800)
    assert count_tokens(index) <= 800
    # Previews are dropped before entries are, so the router still sees breadth.
    assert index.count("\n") > 20


def test_previews_shrink_rather_than_entries_disappearing():
    chunks = [
        Chunk(id=f"c{i}", heading_path=("Short",), text="x " * 500, token_count=250, depth=0)
        for i in range(30)
    ]
    generous = render_index(chunks, preview_chars=140)
    fitted = build_index(chunks, max_tokens=400)
    assert count_tokens(fitted) < count_tokens(generous)
    assert len(fitted.splitlines()) == len(chunks)


def test_index_marks_which_chunks_can_be_descended_into(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    index = build_index(tree)
    assert "subsections" in index  # oversized branches
    assert "leaf" in index  # readable sections


def test_empty_index_is_still_a_valid_string():
    assert build_index([]) == "(no sections)"


def test_findings_render_without_leaking_source_text():
    from app.models import Finding

    findings = [
        Finding(
            chunk_id="c1",
            heading_path="A > B",
            sub_question="q",
            found=True,
            answer="the answer",
            evidence=["a quote"],
            confidence=0.75,
        )
    ]
    rendered = format_findings(findings)
    assert "A > B" in rendered
    assert "the answer" in rendered
    assert "0.75" in rendered
    assert format_findings([]) == "(nothing yet)"

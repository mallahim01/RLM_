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


def _sibling(chunk_id: str, *path: str) -> Chunk:
    return Chunk(id=chunk_id, heading_path=path, text="body " * 50, token_count=100, depth=1)


def test_the_shared_ancestor_path_is_not_repeated_on_every_line():
    """Inside a branch the common prefix is pure token cost and no signal."""
    siblings = [
        _sibling("c1.1", "A Very Long Document Title", "Details", "SAP"),
        _sibling("c1.2", "A Very Long Document Title", "Details", "Odoo"),
        _sibling("c1.3", "A Very Long Document Title", "Details", "Oracle"),
    ]
    index = build_index(siblings)

    assert "A Very Long Document Title" not in index
    for name in ("SAP", "Odoo", "Oracle"):
        assert name in index
    for cid in ("c1.1", "c1.2", "c1.3"):
        assert f"[{cid}]" in index, "ids must stay intact -- they are what gets selected"


def test_stripping_never_removes_a_chunks_own_heading():
    identical = [_sibling("c1", "Same", "Path"), _sibling("c2", "Same", "Path")]
    index = build_index(identical)
    assert "Path" in index


def test_nothing_is_stripped_when_paths_diverge_at_the_top():
    mixed = [_sibling("c1", "Alpha", "One"), _sibling("c2", "Beta", "Two")]
    index = build_index(mixed)
    assert "Alpha" in index and "Beta" in index


def test_a_single_candidate_keeps_its_full_path():
    index = build_index([_sibling("c1", "Alpha", "One")])
    assert "Alpha" in index and "One" in index


def test_stripping_measurably_shrinks_a_deep_index(erp_doc):
    tree = build_chunk_tree(erp_doc.sections, 600, 60)
    platforms = next(c for c in tree if c.path_str == "Details by Platform")
    deep = platforms.children[0].children  # the "part n/m" leaves

    stripped = build_index(deep)
    full = render_index(deep, preview_chars=140)

    assert count_tokens(stripped) < count_tokens(full)


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

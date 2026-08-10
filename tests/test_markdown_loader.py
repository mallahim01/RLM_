from __future__ import annotations

from app.loaders.markdown_loader import PREAMBLE, load_markdown, parse_markdown


def test_heading_hierarchy_of_the_real_document(erp_doc):
    assert len(erp_doc.sections) == 14
    paths = [s.heading_path for s in erp_doc.sections]

    # The H1 wraps everything, so every deeper path carries it as a prefix.
    title = paths[0][0]
    assert title.startswith("Pre-Sales Dossier")
    assert (title, "Details by Platform") in paths
    assert (title, "Details by Platform", "1) SAP (S/4HANA / SAP Business AI Platform)") in paths
    assert max(len(p) for p in paths) == 3


def test_non_ascii_characters_survive_the_round_trip(erp_doc):
    body = "\n".join(s.text for s in erp_doc.sections)
    assert "→" in body or "—" in body  # arrows / em-dashes are in this file


def test_hash_inside_a_code_fence_is_not_a_heading():
    text = "# Real\n\nbody\n\n```python\n# not a heading\nx = 1\n```\n\nmore body\n"
    sections = parse_markdown(text)
    assert len(sections) == 1
    assert sections[0].heading_path == ("Real",)
    assert "# not a heading" in sections[0].text


def test_text_before_the_first_heading_becomes_a_preamble():
    sections = parse_markdown("loose intro text\n\n# Heading\n\nbody\n")
    assert sections[0].heading_path == (PREAMBLE,)
    assert sections[0].text == "loose intro text"


def test_nested_headings_build_full_paths():
    sections = parse_markdown("# A\n\na\n\n## B\n\nb\n\n### C\n\nc\n\n## D\n\nd\n")
    assert [s.heading_path for s in sections] == [
        ("A",),
        ("A", "B"),
        ("A", "B", "C"),
        ("A", "D"),
    ]


def test_loader_reads_utf8_explicitly(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Café → test\n\nbody with — dash\n", encoding="utf-8")
    document = load_markdown(path)
    assert document.title == "Café → test"
    assert "—" in document.sections[0].text
    assert document.total_tokens > 0

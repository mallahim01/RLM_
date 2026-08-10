"""The docx and pdf loaders, plus the dispatcher."""

from __future__ import annotations

import zipfile

import pytest

from app.loaders import UnsupportedFormatError, load_document
from app.loaders.docx_loader import DocxError, parse_docx_xml
from app.loaders.pdf_loader import _looks_like_heading, sections_from_pages
from tests.conftest import PROJECT_ROOT

TEST_FILES = PROJECT_ROOT / "test_files"
DOCX = TEST_FILES / "knowledge-product-pakistan.docx"
PDF = TEST_FILES / "WhatsApp Architecture and Technology Deep Dive.pdf"

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx_xml(paragraphs: list[tuple[str, int]]) -> bytes:
    """Build a minimal word/document.xml from (text, heading level) pairs."""
    body = []
    for text, level in paragraphs:
        style = f'<w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr>' if level else ""
        body.append(f"<w:p>{style}<w:r><w:t>{text}</w:t></w:r></w:p>")
    return (
        f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>'
        + "".join(body)
        + "</w:body></w:document>"
    ).encode("utf-8")


def test_docx_heading_styles_become_heading_paths():
    xml = _docx_xml([("Title", 1), ("intro body", 0), ("Sub", 2), ("sub body", 0), ("Next", 1)])
    sections = parse_docx_xml(xml)

    assert [s.heading_path for s in sections] == [("Title",), ("Title", "Sub"), ("Next",)]
    assert sections[0].text == "intro body"
    assert sections[1].level == 2


def test_docx_body_text_before_any_heading_is_a_preamble():
    sections = parse_docx_xml(_docx_xml([("loose text", 0), ("Heading", 1)]))
    assert sections[0].heading_path == ("(preamble)",)


def test_a_document_using_only_heading4_is_treated_as_top_level():
    """The real sample .docx does exactly this. Taking levels literally would
    manufacture three tiers of empty placeholder headings."""
    xml = _docx_xml([("First", 4), ("a", 0), ("Second", 4), ("b", 0)])
    sections = parse_docx_xml(xml)

    assert [s.heading_path for s in sections] == [("First",), ("Second",)]
    assert all(s.level == 1 for s in sections)
    assert "(untitled)" not in str([s.heading_path for s in sections])


def test_relative_nesting_survives_normalisation():
    xml = _docx_xml([("Top", 2), ("a", 0), ("Under", 4), ("b", 0), ("Next Top", 2)])
    sections = parse_docx_xml(xml)

    assert [s.heading_path for s in sections] == [("Top",), ("Top", "Under"), ("Next Top",)]


def test_repeated_heading_paths_do_not_lose_their_text():
    """PDF page sections and duplicate markdown headings both hit this."""
    from app.models import Section
    from app.rlm.chunker import build_chunk_tree

    sections = [
        Section(("Same",), "first body", 1),
        Section(("Other",), "middle", 1),
        Section(("Same",), "second body", 1),
    ]
    tree = build_chunk_tree(sections, target_tokens=600, overlap_tokens=0)
    combined = " ".join(c.full_text() for c in tree)

    assert "first body" in combined
    assert "second body" in combined


def test_a_non_docx_file_raises_a_clear_error(tmp_path):
    fake = tmp_path / "fake.docx"
    fake.write_text("definitely not a zip", encoding="utf-8")
    with pytest.raises(DocxError):
        load_document(fake)


def test_a_zip_without_document_xml_raises(tmp_path):
    path = tmp_path / "empty.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("other.txt", "x")
    with pytest.raises(DocxError):
        load_document(path)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("System Architecture Overview", True),
        ("SIGNAL PROTOCOL", True),
        ("This is a normal sentence that ends with a period.", False),
        ("- a bullet point item", False),
        ("1. numbered list entry", False),
        ("x", False),
    ],
)
def test_pdf_heading_heuristic(line, expected):
    assert _looks_like_heading(line) is expected


def test_pdf_pages_become_sections_with_promoted_headings():
    pages = ["Introduction\nSome body text here that goes on.\n\nMore text.", "Second page body."]
    sections = sections_from_pages(pages)

    assert any(s.heading_path == ("Introduction",) for s in sections)
    assert all(s.text.strip() for s in sections)


def test_dispatcher_rejects_unknown_extensions(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_document(path)


def test_dispatcher_reports_a_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("no/such/file.md")


@pytest.mark.skipif(not DOCX.exists(), reason="sample .docx not present")
def test_the_real_sample_docx_loads():
    document = load_document(DOCX)
    assert document.sections
    assert document.total_tokens > 100


@pytest.mark.skipif(not PDF.exists(), reason="sample .pdf not present")
def test_the_real_sample_pdf_loads():
    pytest.importorskip("pypdf")
    document = load_document(PDF)
    assert len(document.sections) > 5
    assert document.total_tokens > 1000

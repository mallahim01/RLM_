"""DOCX -> ``Section`` list, using only the standard library.

A `.docx` is a zip archive; `word/document.xml` holds `<w:p>` paragraphs, and a
paragraph's `<w:pStyle w:val="Heading2"/>` gives its heading level directly. That
is all this loader needs, so it costs zero dependencies -- notably avoiding
`python-docx`, which pulls in `lxml`, a multi-megabyte C extension.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from app.loaders._structure import PREAMBLE, Block, assemble_sections
from app.models import Document, Section
from app.rlm.tokens import count_tokens

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)

__all__ = ["PREAMBLE", "DocxError", "parse_docx_xml", "load_docx"]


class DocxError(ValueError):
    """Raised when a file is not a readable .docx."""


def _paragraph_text(para: ElementTree.Element) -> str:
    """Concatenate the runs of a paragraph, honouring explicit line breaks."""
    parts: list[str] = []
    for node in para.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
    return "".join(parts).strip()


def _heading_level(para: ElementTree.Element) -> int:
    """0 for body text, 1-6 for a Heading N style."""
    style = para.find(f"{_W}pPr/{_W}pStyle")
    if style is None:
        return 0
    match = _HEADING_STYLE.match(style.get(f"{_W}val", ""))
    if not match:
        return 0
    return min(6, int(match.group(1)))


def parse_docx_xml(xml_bytes: bytes) -> list[Section]:
    """Turn `word/document.xml` into sections, mirroring the markdown loader."""
    root = ElementTree.fromstring(xml_bytes)
    body = root.find(f"{_W}body")
    if body is None:
        return []

    blocks: list[Block] = []
    buffer: list[str] = []
    level, title = 0, ""

    def flush() -> None:
        blocks.append((level, title, "\n\n".join(buffer).strip()))
        buffer.clear()

    for para in body.findall(f"{_W}p"):
        text = _paragraph_text(para)
        para_level = _heading_level(para)
        if para_level == 0:
            if text:
                buffer.append(text)
            continue
        flush()
        level, title = para_level, text

    flush()
    return assemble_sections(blocks)


def load_docx(path: str | Path) -> Document:
    p = Path(path)
    try:
        with zipfile.ZipFile(p) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DocxError(f"{p.name} is not a readable .docx file: {exc}") from exc

    sections = parse_docx_xml(xml_bytes)
    title = sections[0].heading_path[0] if sections else p.name
    if title == PREAMBLE:
        title = p.stem
    return Document(
        path=str(p),
        title=title,
        sections=sections,
        total_tokens=sum(count_tokens(s.text) for s in sections),
    )

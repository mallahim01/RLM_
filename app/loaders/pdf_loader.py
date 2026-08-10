"""PDF -> ``Section`` list, via ``pypdf``.

Honest caveat, repeated in the README: PDFs carry no heading markup. Structure
recovery here is *best effort* -- a short, unpunctuated, title-cased line is
promoted to a heading, and everything else falls back to page boundaries. It is
noticeably worse than the markdown loader, which is itself instructive: the RLM
index is only as good as the structure it is given, and the lexical pre-filter
exists precisely to cope when that structure is weak.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.models import Document, Section
from app.rlm.tokens import count_tokens

_LOG = logging.getLogger("rlm")

_MAX_HEADING_CHARS = 80
_MAX_HEADING_WORDS = 8
_SENTENCE_END = re.compile(r"[.!?,;:]$")
_BULLET = re.compile(r"^\s*([-*•●]|\d+[.)])\s")

# If the heuristic fires more often than this (one heading per N words), it is
# almost certainly mistaking wrapped body lines for headings. Page sections are
# a worse index but an honest one; over-segmentation is worse than both.
_WORDS_PER_HEADING_FLOOR = 120


class PDFError(ValueError):
    """Raised when a PDF cannot be opened or has no extractable text."""


def _looks_like_heading(line: str) -> bool:
    """Short, unpunctuated, capitalised, not a bullet -- probably a heading.

    Tuned conservatively: PDF text extraction wraps body prose into short lines,
    and a loose rule turns every one of them into a section.
    """
    stripped = line.strip()
    if not (3 <= len(stripped) <= _MAX_HEADING_CHARS):
        return False
    if _SENTENCE_END.search(stripped) or _BULLET.match(stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False
    words = stripped.split()
    if len(words) > _MAX_HEADING_WORDS:
        return False
    if stripped == stripped.upper():
        return True
    # A wrapped body line usually starts lowercase or has mostly lowercase words.
    if not stripped[0].isupper():
        return False
    capitalised = sum(1 for w in words if w[:1].isupper())
    return capitalised / len(words) >= 0.8


def sections_from_pages(pages: list[str], detect_headings: bool = True) -> list[Section]:
    """Build sections from per-page text, promoting heading-like lines."""
    sections: list[Section] = []
    current_title = "Page 1"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n\n".join(b for b in buffer if b.strip()).strip()
        buffer.clear()
        if text:
            sections.append(Section(heading_path=(current_title,), text=text, level=1))

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        if not detect_headings:
            flush()
            current_title = f"Page {page_number}"
            buffer.append(page_text.strip())
            continue

        paragraph: list[str] = []
        for line in page_text.splitlines():
            if _looks_like_heading(line):
                if paragraph:
                    buffer.append(" ".join(paragraph).strip())
                    paragraph = []
                flush()
                current_title = line.strip()
                continue
            if line.strip():
                paragraph.append(line.strip())
            elif paragraph:
                buffer.append(" ".join(paragraph).strip())
                paragraph = []
        if paragraph:
            buffer.append(" ".join(paragraph).strip())

    flush()
    return sections


def load_pdf(path: str | Path) -> Document:
    p = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # keep the failure actionable
        raise PDFError(
            "Reading PDFs needs the `pypdf` package. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from exc

    try:
        reader = PdfReader(str(p))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise PDFError(f"Could not read {p.name}: {exc}") from exc

    sections = sections_from_pages(pages)

    # Guard against a heuristic that fired on wrapped body text: too many
    # sections means a shredded index, which is worse than plain page splits.
    words = sum(len(page.split()) for page in pages)
    if len(sections) > max(8, words // _WORDS_PER_HEADING_FLOOR):
        _LOG.info(
            "[RLM] pdf: heading heuristic produced %d fragments from %d words "
            "-> falling back to page sections",
            len(sections),
            words,
        )
        sections = sections_from_pages(pages, detect_headings=False)

    if not sections:
        raise PDFError(
            f"No extractable text in {p.name}. It may be a scanned image PDF, "
            "which would need OCR (out of scope for this demo)."
        )
    return Document(
        path=str(p),
        title=p.stem,
        sections=sections,
        total_tokens=sum(count_tokens(s.text) for s in sections),
    )

"""Document loaders. Every format produces the same ``Document``/``Section`` shape.

Imports are deferred so that a missing optional dependency surfaces as a clear
message when that format is actually used, rather than at startup.
"""

from __future__ import annotations

from pathlib import Path

from app.models import Document

MARKDOWN_SUFFIXES = {".md", ".markdown", ".txt"}


class UnsupportedFormatError(ValueError):
    """Raised for a file extension no loader handles."""


def load_document(path: str | Path) -> Document:
    """Load any supported document by file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such document: {p}")

    suffix = p.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        from app.loaders.markdown_loader import load_markdown

        return load_markdown(p)
    if suffix == ".docx":
        from app.loaders.docx_loader import load_docx

        return load_docx(p)
    if suffix == ".pdf":
        from app.loaders.pdf_loader import load_pdf

        return load_pdf(p)

    supported = ", ".join(sorted(MARKDOWN_SUFFIXES | {".docx", ".pdf"}))
    raise UnsupportedFormatError(f"Cannot load {p.name}: supported formats are {supported}")


__all__ = ["load_document", "UnsupportedFormatError", "MARKDOWN_SUFFIXES", "Document"]

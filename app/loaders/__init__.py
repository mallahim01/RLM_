"""Document loaders. Every format produces the same ``Document``/``Section`` shape.

Imports are deferred so that a missing optional dependency surfaces as a clear
message when that format is actually used, rather than at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.models import Document, Section

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


def merge_documents(documents: Sequence[Document], title: str = "Merged corpus") -> Document:
    """Combine several documents into one, each becoming a top-level branch.

    Every section's heading path is prefixed with its source document's title,
    so the router's index shows which file a section came from and can pick a
    document before picking a section within it. That is multi-document routing
    for free: the engine already recurses, and this just gives it one more level
    to recurse over.
    """
    if not documents:
        raise ValueError("merge_documents needs at least one document")
    if len(documents) == 1:
        return documents[0]

    sections: list[Section] = []
    for document in documents:
        label = document.title.strip() or Path(document.path).stem
        # A document whose sections all hang off one heading (markdown with a
        # single H1) already has a root. Prefixing it would duplicate the title
        # on every path and add a level of depth that says nothing -- replace
        # that root with the label instead of stacking a second one on top.
        roots = {s.heading_path[0] for s in document.sections if s.heading_path}
        has_descendants = any(len(s.heading_path) > 1 for s in document.sections)
        # One shared root is only a *root* if something hangs beneath it. A
        # document with a single flat section shares its "root" trivially, and
        # replacing that would throw the section's own heading away.
        already_rooted = len(roots) == 1 and has_descendants

        for section in document.sections:
            if already_rooted:
                path = (label,) + section.heading_path[1:]
                level = section.level
            else:
                path = (label,) + section.heading_path
                level = section.level + 1
            sections.append(Section(heading_path=path, text=section.text, level=level))

    return Document(
        path=" + ".join(Path(d.path).name for d in documents),
        title=title,
        sections=sections,
        total_tokens=sum(d.total_tokens for d in documents),
    )


def load_documents(paths: Sequence[str | Path]) -> Document:
    """Load one or many documents; many are merged into a single corpus."""
    if not paths:
        raise ValueError("no document paths given")
    return merge_documents([load_document(p) for p in paths])


__all__ = [
    "load_document",
    "load_documents",
    "merge_documents",
    "UnsupportedFormatError",
    "MARKDOWN_SUFFIXES",
    "Document",
]

"""Markdown -> ``Section`` list.

The reference loader. Markdown carries explicit heading levels, so the section
structure that the chunker later turns into a tree comes for free.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.loaders._structure import PREAMBLE, Block, assemble_sections
from app.models import Document, Section
from app.rlm.tokens import count_tokens

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

__all__ = ["PREAMBLE", "parse_markdown", "load_markdown"]


def parse_markdown(text: str) -> list[Section]:
    """Split markdown into one section per heading, tracking the heading stack."""
    blocks: list[Block] = []
    buffer: list[str] = []
    level, title = 0, ""
    in_fence = False

    def flush() -> None:
        blocks.append((level, title, "\n".join(buffer).strip()))
        buffer.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        # A '#' inside a fenced block is code, not a heading.
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue
        flush()
        level = len(match.group(1))
        title = match.group(2).strip()

    flush()
    return assemble_sections(blocks)


def load_markdown(path: str | Path) -> Document:
    """Read a markdown/plain-text file as a ``Document``. Always UTF-8."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    sections = parse_markdown(text)
    title = sections[0].heading_path[0] if sections else p.name
    if title == PREAMBLE:
        title = p.name
    return Document(
        path=str(p),
        title=title,
        sections=sections,
        total_tokens=sum(count_tokens(s.text) for s in sections),
    )

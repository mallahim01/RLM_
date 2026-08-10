"""Shared heading-stack logic for the loaders.

Both markdown and docx arrive at the same intermediate shape -- a flat list of
(level, title, body) blocks -- so the nesting rules live here once.

The important step is **level normalisation**. A document that only uses
``Heading 4``, or markdown that starts at ``###``, is perfectly ordinary; taking
its raw levels literally would manufacture three tiers of empty placeholder
headings and a tree that is deep but says nothing. Ranking the levels that
actually occur fixes that.
"""

from __future__ import annotations

from app.models import Section

PREAMBLE = "(preamble)"
UNTITLED = "(untitled)"

# (level, title, body); level 0 means "text before any heading".
Block = tuple[int, str, str]


def normalize_levels(blocks: list[Block]) -> list[Block]:
    """Rank the heading levels that occur to consecutive depths starting at 1."""
    present = sorted({level for level, _title, _body in blocks if level > 0})
    rank = {level: i + 1 for i, level in enumerate(present)}
    return [(rank.get(level, 0), title, body) for level, title, body in blocks]


def assemble_sections(blocks: list[Block]) -> list[Section]:
    """Turn (level, title, body) blocks into sections with full heading paths."""
    sections: list[Section] = []
    stack: list[str] = []

    for level, title, body in normalize_levels(blocks):
        if level == 0:
            if body.strip():
                sections.append(Section(heading_path=(PREAMBLE,), text=body.strip(), level=0))
            continue

        del stack[level - 1 :]
        while len(stack) < level - 1:
            stack.append(UNTITLED)  # a genuinely skipped level, e.g. H1 then H3
        stack.append(title.strip() or f"{UNTITLED} h{level}")
        sections.append(Section(heading_path=tuple(stack), text=body.strip(), level=level))

    return sections

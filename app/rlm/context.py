"""The index: a compact table of contents, and the only thing the router ever sees.

This is the idea the whole project is teaching. The router is never given
document text -- it is given one line per candidate section (id, heading path,
size, whether it can be descended into, and a short preview) and must decide
where to look from that alone.

The consequence is the point: the router's context is O(entries shown), not
O(document). A 10 MB corpus and a 10 KB one produce an index of the same size.
"""

from __future__ import annotations

from app.models import Chunk
from app.rlm.tokens import count_tokens

# Preview lengths tried in order until the rendered index fits its budget.
_PREVIEW_LADDER = (140, 100, 60, 30, 0)


def _entry(chunk: Chunk, preview_chars: int) -> str:
    shape = "leaf" if chunk.is_leaf else f"{len(chunk.children)} subsections"
    line = f"[{chunk.id}] {chunk.path_str} | {chunk.token_count} tok | {shape}"
    if preview_chars:
        line += f' | "{chunk.preview(preview_chars)}"'
    return line


def render_index(chunks: list[Chunk], preview_chars: int = 140) -> str:
    """Render one line per chunk at a fixed preview length."""
    return "\n".join(_entry(c, preview_chars) for c in chunks)


def build_index(chunks: list[Chunk], max_tokens: int = 800) -> str:
    """Render the index, shrinking previews until it fits ``max_tokens``.

    Guarantees the router prompt cannot overflow regardless of input size.
    """
    if not chunks:
        return "(no sections)"
    for preview_chars in _PREVIEW_LADDER:
        rendered = render_index(chunks, preview_chars)
        if count_tokens(rendered) <= max_tokens:
            return rendered
    # Even bare id + heading lines are too long: keep the head and say so.
    # The notice itself costs tokens, so reserve room for it before packing.
    lines = render_index(chunks, 0).splitlines()
    notice = f"... and {len(lines)} more sections omitted to fit the index budget"
    budget = max(0, max_tokens - count_tokens(notice) - 1)

    kept: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost

    dropped = len(lines) - len(kept)
    if dropped:
        kept.append(f"... and {dropped} more sections omitted to fit the index budget")
    return "\n".join(kept)


def render_tree(chunks: list[Chunk], indent: int = 0) -> str:
    """Human-readable chunk tree. Used by ``--show-tree`` and by tests."""
    lines: list[str] = []
    for chunk in chunks:
        shape = "leaf" if chunk.is_leaf else f"{len(chunk.children)} children"
        lines.append(
            f"{'  ' * indent}{chunk.id:<10} {chunk.token_count:>6} tok  {shape:<12} "
            f"{chunk.path_str}"
        )
        if chunk.children:
            lines.append(render_tree(chunk.children, indent + 1))
    return "\n".join(lines)


def format_findings(findings, max_chars: int = 600) -> str:
    """Render findings as the input to a routing or synthesis prompt.

    Only the *conclusions* travel upward, never the source text -- which is what
    keeps the parent level's context small no matter how much was read below.
    """
    if not findings:
        return "(nothing yet)"
    lines: list[str] = []
    for i, finding in enumerate(findings, start=1):
        status = "found" if finding.found else "not found"
        lines.append(
            f"[{i}] source: {finding.heading_path} ({status}, confidence "
            f"{finding.confidence:.2f})"
        )
        lines.append(f"    answer: {finding.answer[:max_chars]}")
        for quote in finding.evidence[:2]:
            lines.append(f'    evidence: "{quote[:240]}"')
    return "\n".join(lines)

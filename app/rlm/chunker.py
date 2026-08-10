"""``Section`` list -> ``Chunk`` tree.

The tree is what makes recursion possible: a node small enough to read in one
call becomes a leaf, and anything bigger keeps its children so the engine can
descend into it. Two rules do all the work:

* **Collapse** -- a subtree that fits the target becomes a single leaf, so the
  index is not cluttered with seven-token wrapper headings.
* **Hard-split** -- a leaf that is still too big is cut along paragraph, then
  sentence, then whitespace boundaries. Never mid-word.

Chunk ids are short, deterministic, and hierarchical (``c4``, ``c4.1``,
``c4.1.2``). They appear verbatim in prompts, so short and structured beats
random: long opaque ids measurably increase the rate at which a model invents
one that does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models import Chunk, Section
from app.rlm.tokens import count_tokens, truncate_to_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HORIZONTAL_RULE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)

# Own body text below this size is treated as structural (a `---` rule, a stray
# line) and left attached to its heading rather than promoted to an intro chunk.
_MIN_INTRO_TOKENS = 25


@dataclass
class _Node:
    """Mutable scratch tree, converted to immutable-ish ``Chunk``s at the end."""

    heading_path: tuple[str, ...]
    text: str = ""
    children: list["_Node"] = field(default_factory=list)

    def subtree_tokens(self) -> int:
        return count_tokens(self.text) + sum(c.subtree_tokens() for c in self.children)

    def collapsed_text(self) -> str:
        parts = [self.text.strip()] if self.text.strip() else []
        for child in self.children:
            # Re-introduce the heading so collapsed content keeps its labels.
            title = child.heading_path[-1]
            body = child.collapsed_text()
            parts.append(f"{title}\n{body}" if body else title)
        return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# splitting primitives
# --------------------------------------------------------------------------


def _sentence_units(paragraph: str, target: int) -> list[str]:
    """Break an oversized paragraph into sentence groups of at most ``target``."""
    sentences = [s for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    units: list[str] = []
    buf: list[str] = []
    size = 0
    for sentence in sentences:
        n = count_tokens(sentence)
        if n > target:
            if buf:
                units.append(" ".join(buf))
                buf, size = [], 0
            units.extend(_character_units(sentence, target))
            continue
        if size + n > target and buf:
            units.append(" ".join(buf))
            buf, size = [], 0
        buf.append(sentence)
        size += n
    if buf:
        units.append(" ".join(buf))
    return units


def _character_units(text: str, target: int) -> list[str]:
    """Last resort for a single sentence bigger than the target: window it.

    Always snaps to whitespace, so no chunk ever begins or ends mid-word.
    """
    units: list[str] = []
    remaining = text.strip()
    while remaining:
        head = truncate_to_tokens(remaining, target)
        if not head:  # pathological input; take a fixed slice to guarantee progress
            head = remaining[: target * 4]
        units.append(head.strip())
        remaining = remaining[len(head) :].strip()
    return units


def _split_units(text: str, target: int) -> list[str]:
    """Break text into semantic units, none larger than ``target``."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= target:
            units.append(paragraph)
        else:
            units.extend(_sentence_units(paragraph, target))
    return units


def _pack(units: list[str], budget: int) -> list[str]:
    """Greedily group units into parts of at most ``budget`` tokens."""
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for unit in units:
        n = count_tokens(unit)
        if buf and size + n > budget:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(unit)
        size += n
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def _tail(text: str, overlap_tokens: int) -> str:
    """The last ``overlap_tokens`` of ``text``, snapped forward to a sentence start."""
    if overlap_tokens <= 0 or not text:
        return ""
    total = count_tokens(text)
    if total <= overlap_tokens:
        return text
    # Walk back from the end until we have roughly the right number of tokens.
    approx_chars = overlap_tokens * 5
    tail = text[-approx_chars:]
    tail = truncate_to_tokens(tail[::-1], overlap_tokens)[::-1]
    boundary = _SENTENCE_SPLIT.search(tail)
    if boundary:
        tail = tail[boundary.end() :]
    else:
        space = tail.find(" ")
        tail = tail[space + 1 :] if space != -1 else tail
    return tail.strip()


def split_text(text: str, target: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping parts, each at most ``target`` tokens."""
    if count_tokens(text) <= target:
        return [text.strip()]
    overlap = max(0, min(overlap, target // 3))
    parts = _pack(_split_units(text, target - overlap), target - overlap)
    if overlap == 0 or len(parts) < 2:
        return parts
    out = [parts[0]]
    for previous, current in zip(parts, parts[1:]):
        carry = _tail(previous, overlap)
        out.append(f"{carry}\n\n{current}" if carry else current)
    return out


# --------------------------------------------------------------------------
# tree construction
# --------------------------------------------------------------------------


def _build_hierarchy(sections: list[Section]) -> list[_Node]:
    """Nest sections by heading-path prefix, creating missing intermediate nodes."""
    roots: list[_Node] = []
    by_path: dict[tuple[str, ...], _Node] = {}

    def ensure(path: tuple[str, ...]) -> _Node:
        if path in by_path:
            return by_path[path]
        node = _Node(heading_path=path)
        by_path[path] = node
        if len(path) == 1:
            roots.append(node)
        else:
            ensure(path[:-1]).children.append(node)
        return node

    for section in sections:
        node = ensure(section.heading_path)
        # Repeated heading paths are common in PDFs and not illegal in markdown.
        # Merge rather than overwrite, or the later occurrence's text is lost.
        node.text = f"{node.text}\n\n{section.text}".strip() if node.text else section.text
    return roots


def _strip_path_prefix(node: _Node, count: int) -> None:
    """Drop the first ``count`` heading levels from a subtree, in place."""
    node.heading_path = node.heading_path[count:]
    for child in node.children:
        _strip_path_prefix(child, count)


def _unwrap_single_root(roots: list[_Node]) -> list[_Node]:
    """Promote the children of a lone wrapping H1 to the top level.

    Documents that open with a single title heading would otherwise produce a
    one-entry index, which gives the router nothing to choose between. The
    now-redundant title is also stripped from every descendant path, because it
    would otherwise be repeated on every line of the index -- pure token cost
    with no discriminating power.
    """
    while len(roots) == 1 and roots[0].children:
        root = roots[0]
        promoted: list[_Node] = []
        if count_tokens(root.text) >= _MIN_INTRO_TOKENS:
            promoted.append(_Node(heading_path=(root.heading_path[-1],), text=root.text))
        for child in root.children:
            _strip_path_prefix(child, len(root.heading_path))
            promoted.append(child)
        roots = promoted
    return roots


def _finalize(node: _Node, target: int, overlap: int) -> None:
    """Apply the collapse and hard-split rules to a node, in place, top-down."""
    if node.subtree_tokens() <= target:
        node.text = node.collapsed_text()
        node.children = []
        return

    if node.children:
        own = node.text.strip()
        if count_tokens(own) >= _MIN_INTRO_TOKENS:
            # Substantial preamble under a heading that must be split: keep it
            # reachable by promoting it to the first child.
            intro = _Node(heading_path=node.heading_path + ("intro",), text=own)
            node.children.insert(0, intro)
            node.text = ""
        elif _HORIZONTAL_RULE.fullmatch(own) or not own:
            node.text = ""
        for child in node.children:
            _finalize(child, target, overlap)
        return

    parts = split_text(node.text, target, overlap)
    if len(parts) < 2:
        return  # unsplittable; the engine truncates it at inspect time
    total = len(parts)
    node.children = [
        _Node(heading_path=node.heading_path + (f"part {i}/{total}",), text=part)
        for i, part in enumerate(parts, start=1)
    ]
    node.text = ""


def _to_chunks(nodes: list[_Node], parent_id: str | None, depth: int, prefix: str) -> list[Chunk]:
    """Assign deterministic ids and convert the scratch tree to ``Chunk``s."""
    chunks: list[Chunk] = []
    for index, node in enumerate(nodes, start=1):
        chunk_id = f"{prefix}{index}" if prefix else f"c{index}"
        children = _to_chunks(node.children, chunk_id, depth + 1, f"{chunk_id}.")
        own = count_tokens(node.text)
        chunks.append(
            Chunk(
                id=chunk_id,
                heading_path=node.heading_path,
                text=node.text,
                token_count=own + sum(c.token_count for c in children),
                depth=depth,
                parent_id=parent_id,
                children=children,
            )
        )
    return chunks


def build_chunk_tree(
    sections: list[Section],
    target_tokens: int = 600,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Turn a document's sections into the tree the RLM engine navigates."""
    roots = _unwrap_single_root(_build_hierarchy(sections))
    for root in roots:
        _finalize(root, target_tokens, overlap_tokens)
    return _to_chunks(roots, parent_id=None, depth=0, prefix="")


def index_chunks(chunks: list[Chunk]) -> dict[str, Chunk]:
    """Flatten a tree into an id -> chunk lookup."""
    return {c.id: c for root in chunks for c in root.walk()}


def tree_depth(chunks: list[Chunk]) -> int:
    """Number of levels in the tree (1 for a flat list of leaves)."""
    if not chunks:
        return 0
    return 1 + max((tree_depth(c.children) for c in chunks), default=0)


def total_tokens(chunks: list[Chunk]) -> int:
    return sum(c.token_count for c in chunks)

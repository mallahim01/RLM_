"""Every data shape the project uses.

Kept in one module on purpose: these are six small dataclasses with almost no
behaviour, and splitting them across a package would add imports without adding
clarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Section:
    """One heading block from a source document, before any size-based splitting."""

    heading_path: tuple[str, ...]
    text: str
    level: int

    @property
    def path_str(self) -> str:
        return " > ".join(self.heading_path)


@dataclass(slots=True)
class Document:
    """A loaded source document as a flat list of heading-delimited sections."""

    path: str
    title: str
    sections: list[Section]
    total_tokens: int


@dataclass(slots=True)
class Chunk:
    """A node in the chunk tree.

    ``token_count`` is the *subtree* total (own body plus every descendant),
    because that is the number routing decisions must key on: it answers "is this
    branch small enough to read in one go, or must I descend into it?".
    """

    id: str
    heading_path: tuple[str, ...]
    text: str
    token_count: int
    depth: int
    parent_id: str | None = None
    children: list["Chunk"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def path_str(self) -> str:
        return " > ".join(self.heading_path)

    def preview(self, chars: int = 140) -> str:
        """A single-line, whitespace-collapsed opening excerpt for the index."""
        flat = _WHITESPACE.sub(" ", self.full_text()).strip()
        if len(flat) <= chars:
            return flat
        return flat[:chars].rstrip() + "..."

    def full_text(self) -> str:
        """This node's body plus every descendant's, in document order."""
        parts = [self.text] if self.text.strip() else []
        for child in self.children:
            child_text = child.full_text()
            if child_text.strip():
                parts.append(child_text)
        return "\n\n".join(parts)

    def walk(self):
        """Yield this chunk and all descendants, depth-first."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(slots=True)
class Selection:
    """One routing decision: which chunk to look at, and what to ask of it."""

    chunk_id: str
    sub_question: str
    why: str = ""


@dataclass(slots=True)
class Finding:
    """What came back from reading one chunk (or from collapsing a sub-level)."""

    chunk_id: str
    heading_path: str
    sub_question: str
    found: bool
    answer: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    needs_more: bool = False
    suggested_chunk_ids: list[str] = field(default_factory=list)
    depth: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "heading_path": self.heading_path,
            "sub_question": self.sub_question,
            "found": self.found,
            "answer": self.answer,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "needs_more": self.needs_more,
            "depth": self.depth,
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class TraceEvent:
    """One observable moment in a run. Mirrors exactly what is printed."""

    kind: str  # load|chunk|basecase|index|prefilter|route|inspect|recurse|compress|synthesize|limit|error
    depth: int
    message: str
    elapsed_s: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "depth": self.depth,
            "message": self.message,
            "elapsed_s": round(self.elapsed_s, 3),
            "data": self.data,
        }


@dataclass(slots=True)
class Stats:
    """Cost and shape of a run. ``context_efficiency`` is the headline number."""

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    max_depth_reached: int = 0
    chunks_total: int = 0
    chunks_inspected: int = 0
    route_calls: int = 0
    document_tokens: int = 0
    document_tokens_read: int = 0
    wall_time_s: float = 0.0

    @property
    def context_efficiency(self) -> float:
        """Fraction of the document's tokens that were actually sent to a model."""
        if self.document_tokens <= 0:
            return 0.0
        return self.document_tokens_read / self.document_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "route_calls": self.route_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "max_depth_reached": self.max_depth_reached,
            "chunks_total": self.chunks_total,
            "chunks_inspected": self.chunks_inspected,
            "document_tokens": self.document_tokens,
            "document_tokens_read": self.document_tokens_read,
            "context_efficiency": round(self.context_efficiency, 4),
            "wall_time_s": round(self.wall_time_s, 3),
        }


@dataclass(slots=True)
class RLMResult:
    """Everything a run produced: the answer, its provenance, and how it got there."""

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)
    stopped_reason: str = "completed"  # completed|max_iterations|budget_exhausted|router_empty|error
    gaps: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "gaps": self.gaps,
            "stopped_reason": self.stopped_reason,
            "stats": self.stats.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "trace": [e.to_dict() for e in self.trace],
        }

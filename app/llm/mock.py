"""Test doubles, and the engine behind ``--mock``.

Two shapes, because tests need two different things:

* ``ScriptedLLMClient`` returns queued replies in order. Use it when the test is
  *about* call order or call count -- budget limits, iteration caps.
* ``PatternLLMClient`` matches a substring of the prompt to a canned reply. Use
  it for happy-path tests, so reordering an internal call does not break them.

``PatternLLMClient`` is also what powers ``python -m app --mock``: a visitor can
watch a full recursive run without an API key.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.llm.base import LLMResponse
from app.rlm.tokens import estimate_tokens


def _render(reply: Any) -> str:
    return reply if isinstance(reply, str) else json.dumps(reply)


class ScriptedLLMClient:
    """Returns queued replies in order. Raises when the script runs out."""

    def __init__(self, replies: list[Any], loop: bool = False) -> None:
        self._replies = [_render(r) for r in replies]
        self._loop = loop
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def drained(self) -> bool:
        return self._index >= len(self._replies)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append({"system": system or "", "prompt": prompt, "json_mode": json_mode})
        if self._index >= len(self._replies):
            if not self._loop or not self._replies:
                raise AssertionError(
                    f"ScriptedLLMClient ran out of replies after {len(self.calls)} calls. "
                    "The engine made more calls than the test expected."
                )
            self._index = 0
        reply = self._replies[self._index]
        self._index += 1
        return LLMResponse(
            text=reply,
            tokens_in=estimate_tokens(prompt) + estimate_tokens(system or ""),
            tokens_out=estimate_tokens(reply),
        )


class PatternLLMClient:
    """Matches a regex against the prompt and returns the paired reply.

    Rules are tried in order; the first match wins. ``default`` covers anything
    unmatched, which keeps tests from breaking when a new call type is added.
    """

    def __init__(
        self,
        rules: list[tuple[str, Any]] | None = None,
        default: Any | Callable[[str], Any] | None = None,
    ) -> None:
        self._rules = [(re.compile(pattern, re.IGNORECASE | re.DOTALL), reply) for pattern, reply in (rules or [])]
        self._default = default
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append({"system": system or "", "prompt": prompt, "json_mode": json_mode})
        haystack = f"{system or ''}\n{prompt}"
        for pattern, reply in self._rules:
            if pattern.search(haystack):
                return self._respond(reply, prompt, system)
        if self._default is None:
            raise AssertionError(f"PatternLLMClient has no rule matching prompt: {prompt[:200]!r}")
        reply = self._default(prompt) if callable(self._default) else self._default
        return self._respond(reply, prompt, system)

    def _respond(self, reply: Any, prompt: str, system: str | None) -> LLMResponse:
        text = _render(reply)
        return LLMResponse(
            text=text,
            tokens_in=estimate_tokens(prompt) + estimate_tokens(system or ""),
            tokens_out=estimate_tokens(text),
        )


# --------------------------------------------------------------------------
# the offline demo client
# --------------------------------------------------------------------------

_INDEX_ENTRY = re.compile(
    r"^\[(?P<id>c[\d.]+)\]\s*(?P<heading>.*?)\s*\|\s*(?P<tokens>\d+) tok\s*\|"
    r"(?P<shape>[^|]*)(?:\|\s*\"(?P<preview>.*?)\")?\s*$",
    re.MULTILINE,
)
_QUESTION = re.compile(r"^QUESTION:\s*(.+)$", re.MULTILINE)
_SECTION_BODY = re.compile(r"^---\n(.*)\n---", re.MULTILINE | re.DOTALL)
_FINDING_SOURCE = re.compile(r"^\[\d+\] source: (.+?) \(", re.MULTILINE)
_FINDING_ANSWER = re.compile(r"^    answer: (.+)$", re.MULTILINE)


class DemoLLMClient:
    """A deterministic stand-in that produces a plausible recursive run offline.

    It does not understand language -- it picks sections by lexical overlap with
    the question and reports the opening sentences of whatever it is shown. That
    is enough to exercise every path in the engine (routing, descent, iteration,
    synthesis) so ``--mock`` shows the real control flow, with the reasoning
    replaced by something honest about being fake.
    """

    def __init__(self, max_selections: int = 2) -> None:
        self.max_selections = max_selections
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append({"system": system or "", "prompt": prompt, "json_mode": json_mode})
        # Key on the role sentence only. Matching the whole prompt would confuse
        # the synthesis system prompt, which mentions "readers", with the reader.
        role = (system or "").splitlines()[0].lower() if system else ""
        if "router" in role:
            payload = self._route(prompt)
        elif "careful reader" in role:
            payload = self._inspect(prompt)
        elif "summariser" in role:
            payload = self._compress(prompt)
        else:
            payload = self._synthesize(prompt)
        text = json.dumps(payload)
        return LLMResponse(
            text=text,
            tokens_in=estimate_tokens(prompt) + estimate_tokens(system or ""),
            tokens_out=estimate_tokens(text),
        )

    # -- individual call types -------------------------------------------

    def _question(self, prompt: str) -> str:
        match = _QUESTION.search(prompt)
        return match.group(1).strip() if match else ""

    def _route(self, prompt: str) -> dict[str, Any]:
        """Rank index entries by keyword overlap, breaking ties toward big sections.

        The size tiebreak matters: with no lexical signal, descending into the
        largest branch is the behaviour that actually exercises the recursion.
        """
        from app.rlm.retrieval import HEADING_WEIGHT, tokenize

        question_terms = set(tokenize(self._question(prompt)))
        scored: list[tuple[float, int, str]] = []
        for match in _INDEX_ENTRY.finditer(prompt):
            heading_terms = set(tokenize(match.group("heading")))
            preview_terms = set(tokenize(match.group("preview") or ""))
            score = (
                HEADING_WEIGHT * len(question_terms & heading_terms)
                + len(question_terms & preview_terms)
            )
            scored.append((score, int(match.group("tokens")), match.group("id")))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        question = self._question(prompt)
        return {
            "reasoning": "offline demo: keyword overlap, largest section wins ties",
            "selections": [
                {
                    "chunk_id": chunk_id,
                    "sub_question": question,
                    "why": f"overlap {score}, {tokens} tok",
                }
                for score, tokens, chunk_id in scored[: self.max_selections]
            ],
        }

    def _inspect(self, prompt: str) -> dict[str, Any]:
        match = _SECTION_BODY.search(prompt)
        body = (match.group(1) if match else prompt).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        excerpt = " ".join(sentences[:2])[:400]
        return {
            "found": bool(excerpt),
            "answer": f"(offline demo) This section begins: {excerpt}",
            "evidence": [excerpt[:240]] if excerpt else [],
            "confidence": 0.6,
            "needs_more": False,
            "suggested_chunk_ids": [],
        }

    def _compress(self, prompt: str) -> dict[str, Any]:
        answers = _FINDING_ANSWER.findall(prompt)
        return {
            "found": bool(answers),
            "answer": f"(offline demo) Merged {len(answers)} sub-section findings: "
            + " | ".join(a[:120] for a in answers[:3]),
            "evidence": [],
            "confidence": 0.5,
        }

    def _synthesize(self, prompt: str) -> dict[str, Any]:
        sources = _FINDING_SOURCE.findall(prompt)
        bullets = "\n".join(f"- {source}" for source in sources) or "- (nothing was read)"
        return {
            "answer": (
                "**This is the offline demo client -- no language model was involved, "
                "so there is no real reasoning here.**\n\n"
                "What *was* real is everything else: the engine chunked the document, "
                "built an index, routed on it, descended into oversized sections and "
                f"read {len(sources)} of them. Those sections were:\n\n{bullets}\n\n"
                "Set `OPENAI_API_KEY` in `.env` and drop `--mock` to get an actual answer."
            ),
            "citations": sources,
            "confidence": 0.5,
            "gaps": "no real reasoning was performed (mock client)",
        }

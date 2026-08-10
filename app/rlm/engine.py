"""The RLM engine: route -> inspect -> recurse -> synthesise.

The loop in ``_solve`` is the whole idea, and it is deliberately short enough to
read in one sitting:

1. Show the router an **index** of the sections at this level -- never their text.
2. It picks a few and writes a sub-question for each.
3. A pick that is small enough gets **read**; a pick that is too big gets
   **descended into**, which runs this same loop one level down.
4. If the readers came back short, route again over what is left (bounded).
5. Combine the findings -- and only the findings -- into an answer.

Three independent guards stop a run: ``max_depth`` (how far down),
``max_iterations`` (how wide per level) and ``max_llm_calls`` (total spend). The
call budget is enforced in exactly one place, ``_BudgetedClient.generate``, so it
cannot be bypassed by a code path that forgets to count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.base import (
    LLMClient,
    LLMError,
    LLMJSONError,
    LLMResponse,
    as_bool,
    as_float,
    as_str_list,
    as_text,
    generate_json,
)
from app.models import Chunk, Document, Finding, RLMResult, Selection, Stats
from app.rlm import prompts
from app.rlm.chunker import build_chunk_tree, index_chunks, total_tokens, tree_depth
from app.rlm.context import build_index, format_findings
from app.rlm.retrieval import prefilter
from app.rlm.tokens import count_tokens, truncate_to_tokens
from app.trace import Tracer


class BudgetExhausted(RuntimeError):
    """The run hit ``max_llm_calls``. Raised from the single accounting point."""


class _BudgetedClient:
    """Wraps an ``LLMClient`` to enforce the call budget and record token usage.

    Every model call in the project goes through here, which is the only reason
    the three guards can be trusted: there is no second place to count.
    """

    def __init__(self, client: LLMClient, max_calls: int, stats: Stats) -> None:
        self._client = client
        self._max_calls = max_calls
        self._stats = stats
        self.calls = 0

    @property
    def remaining(self) -> int:
        return max(0, self._max_calls - self.calls)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if self.calls >= self._max_calls:
            raise BudgetExhausted(f"reached max_llm_calls={self._max_calls}")
        self.calls += 1
        response = self._client.generate(prompt, **kwargs)
        self._stats.llm_calls += 1
        self._stats.tokens_in += response.tokens_in
        self._stats.tokens_out += response.tokens_out
        return response


@dataclass
class _Run:
    """Per-question state. Keeps the engine itself free of mutable run state."""

    question: str
    by_id: dict[str, Chunk]
    stats: Stats
    client: _BudgetedClient
    stopped_reason: str = "completed"
    inspected_ids: set[str] = field(default_factory=set)


class RLMEngine:
    """Answers questions about a document by recursively navigating its structure."""

    def __init__(self, client: LLMClient, settings: Settings, tracer: Tracer | None = None) -> None:
        self.client = client
        self.settings = settings
        self.tracer = tracer or Tracer()

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def answer(self, document: Document, question: str) -> RLMResult:
        s = self.settings
        started = time.perf_counter()
        self.tracer.reset()
        stats = Stats()

        self.tracer.event(
            "load",
            0,
            f"load: {_name(document.path)} ({len(document.sections)} sections, "
            f"{document.total_tokens} tokens)",
            path=document.path,
            sections=len(document.sections),
        )

        tree = build_chunk_tree(document.sections, s.chunk_target_tokens, s.chunk_overlap)
        by_id = index_chunks(tree)
        doc_tokens = total_tokens(tree)
        stats.chunks_total = len(by_id)
        stats.document_tokens = doc_tokens

        self.tracer.event(
            "chunk",
            0,
            f"chunk: {len(by_id)} chunks, {doc_tokens} tokens, tree depth "
            f"{tree_depth(tree)}, leaf target {s.chunk_target_tokens}",
            chunks=len(by_id),
            tokens=doc_tokens,
        )

        run = _Run(
            question=question,
            by_id=by_id,
            stats=stats,
            client=_BudgetedClient(self.client, s.max_llm_calls, stats),
        )

        try:
            if doc_tokens <= s.inspect_budget:
                result = self._direct_answer(run, tree, doc_tokens)
            else:
                self.tracer.event(
                    "limit",
                    0,
                    f"context limit {s.max_context_tokens} tok < document {doc_tokens} tok "
                    "-> recursive mode",
                )
                findings = self._solve(run, tree, question, depth=0)
                result = self._synthesize(run, findings)
        except BudgetExhausted as exc:
            self.tracer.event("limit", 0, f"stopped: {exc}")
            run.stopped_reason = "budget_exhausted"
            result = RLMResult(
                question=question,
                answer="The call budget was exhausted before an answer could be produced. "
                "Raise --max-llm-calls and try again.",
            )
        except LLMError as exc:
            self.tracer.event("error", 0, f"error: {exc}")
            run.stopped_reason = "error"
            result = RLMResult(question=question, answer=f"The run failed: {exc}")

        stats.wall_time_s = time.perf_counter() - started
        result.stats = stats
        result.trace = self.tracer.events
        result.stopped_reason = run.stopped_reason

        self.tracer.event(
            "done",
            0,
            f"done in {stats.wall_time_s:.1f}s | {stats.llm_calls} LLM calls | "
            f"{stats.tokens_in:,} in / {stats.tokens_out:,} out | max depth "
            f"{stats.max_depth_reached}",
        )
        self.tracer.event(
            "done",
            0,
            f"read {stats.document_tokens_read:,} of {stats.document_tokens:,} document "
            f"tokens ({stats.context_efficiency:.1%})",
        )
        return result

    # ------------------------------------------------------------------
    # base case
    # ------------------------------------------------------------------

    def _direct_answer(self, run: _Run, tree: list[Chunk], doc_tokens: int) -> RLMResult:
        """The document fits. Answer in one call and say so plainly."""
        self.tracer.event(
            "basecase",
            0,
            f"document fits the {self.settings.max_context_tokens} tok context "
            "-> answering directly, no recursion needed",
            document_tokens=doc_tokens,
        )
        text = "\n\n".join(f"## {c.path_str}\n{c.full_text()}" for c in tree)
        run.stats.document_tokens_read = doc_tokens
        run.stats.chunks_inspected = len(tree)

        payload = self._json_call(
            run,
            system=prompts.DIRECT_SYSTEM,
            prompt=prompts.DIRECT_USER.format(question=run.question, text=text),
            required_keys=("answer",),
            max_tokens=1200,
        )
        if payload is None:
            run.stopped_reason = "error"
            return RLMResult(question=run.question, answer="The model did not return a usable answer.")
        return RLMResult(
            question=run.question,
            answer=as_text(payload.get("answer")),
            citations=as_str_list(payload.get("citations")),
            gaps=as_text(payload.get("gaps")),
        )

    # ------------------------------------------------------------------
    # the recursive loop
    # ------------------------------------------------------------------

    def _solve(self, run: _Run, chunks: list[Chunk], question: str, depth: int) -> list[Finding]:
        """Route, read, descend and expand over one level of the tree."""
        s = self.settings
        run.stats.max_depth_reached = max(run.stats.max_depth_reached, depth)
        findings: list[Finding] = []
        inspected: set[str] = set()
        pinned: set[str] = set()

        for iteration in range(1, s.max_iterations + 1):
            candidates = [c for c in chunks if c.id not in inspected]
            if not candidates:
                break
            # Keep one call in reserve so synthesis can always run.
            if run.client.remaining <= 1:
                self.tracer.event("limit", depth, "call budget nearly spent -> stopping exploration")
                run.stopped_reason = "budget_exhausted"
                break

            self.tracer.event("route", depth, f"depth {depth} | iteration {iteration}")

            candidates, fired = prefilter(
                question, candidates, s.prefilter_threshold, s.max_index_entries, pinned
            )
            if fired:
                self.tracer.event(
                    "prefilter",
                    depth + 1,
                    f"prefilter: narrowed to {len(candidates)} candidates by lexical score",
                    kept=len(candidates),
                )
            else:
                self.tracer.event(
                    "prefilter",
                    depth + 1,
                    f"prefilter: skipped ({len(candidates)} candidates <= {s.prefilter_threshold})",
                )

            index = build_index(candidates, s.max_index_tokens)
            index_tokens = count_tokens(index)
            share = index_tokens / run.stats.document_tokens if run.stats.document_tokens else 0
            self.tracer.event(
                "index",
                depth + 1,
                f"index: {len(candidates)} chunks -> {index_tokens} tokens "
                f"({share:.1%} of document)",
                index_tokens=index_tokens,
            )

            selections = self._route(run, index, question, findings, inspected, candidates, depth)
            if not selections:
                self.tracer.event("route", depth + 1, "router selected nothing -> stopping this level")
                if not findings:
                    run.stopped_reason = "router_empty"
                break

            round_start = len(findings)
            for selection in selections:
                if run.client.remaining <= 1:
                    run.stopped_reason = "budget_exhausted"
                    break
                chunk = run.by_id[selection.chunk_id]
                inspected.add(chunk.id)
                findings.append(self._visit(run, chunk, selection, depth))

            new_findings = findings[round_start:]
            pinned = {cid for f in new_findings for cid in f.suggested_chunk_ids}
            if not any(f.needs_more or not f.found for f in new_findings):
                break
            if iteration == s.max_iterations:
                self.tracer.event(
                    "limit", depth + 1, f"reached max_iterations={s.max_iterations} at depth {depth}"
                )
                if run.stopped_reason == "completed":
                    run.stopped_reason = "max_iterations"

        return findings

    def _visit(self, run: _Run, chunk: Chunk, selection: Selection, depth: int) -> Finding:
        """Read a chunk, or descend into it when it is too big to read."""
        s = self.settings
        can_descend = bool(chunk.children) and depth < s.max_depth
        if can_descend:
            self.tracer.event(
                "recurse",
                depth + 1,
                f"recurse: {chunk.id} '{chunk.path_str}' is {chunk.token_count} tok "
                f"> {s.chunk_target_tokens} -> descending",
                chunk_id=chunk.id,
            )
            sub_findings = self._solve(run, chunk.children, selection.sub_question, depth + 1)
            return self._compress(run, chunk, selection, sub_findings, depth)
        return self._inspect(run, chunk, selection, depth)

    # ------------------------------------------------------------------
    # individual model calls
    # ------------------------------------------------------------------

    def _route(
        self,
        run: _Run,
        index: str,
        question: str,
        findings: list[Finding],
        inspected: set[str],
        candidates: list[Chunk],
        depth: int,
    ) -> list[Selection]:
        s = self.settings
        known = prompts.ROUTE_KNOWN_BLOCK.format(findings=format_findings(findings, 300)) if findings else ""
        already = prompts.ROUTE_INSPECTED_BLOCK.format(ids=", ".join(sorted(inspected))) if inspected else ""
        payload = self._json_call(
            run,
            system=prompts.ROUTE_SYSTEM.format(max_selections=s.max_selections_per_round),
            prompt=prompts.ROUTE_USER.format(
                question=question, known=known, index=index, inspected=already
            ),
            required_keys=("selections",),
            max_tokens=500,
        )
        run.stats.route_calls += 1
        if payload is None:
            return []

        valid_ids = {c.id for c in candidates}
        selections: list[Selection] = []
        for raw in payload.get("selections") or []:
            if not isinstance(raw, dict):
                continue
            chunk_id = as_text(raw.get("chunk_id"))
            if chunk_id not in valid_ids:
                self.tracer.event(
                    "route", depth + 1, f"route: dropped unknown section id '{chunk_id}'"
                )
                continue
            if chunk_id in inspected:
                continue
            if any(sel.chunk_id == chunk_id for sel in selections):
                continue
            selections.append(
                Selection(
                    chunk_id=chunk_id,
                    sub_question=as_text(raw.get("sub_question")) or question,
                    why=as_text(raw.get("why")),
                )
            )

        selections = selections[: s.max_selections_per_round]
        if selections:
            picked = ", ".join(f'{sel.chunk_id} "{_clip(sel.sub_question, 60)}"' for sel in selections)
            self.tracer.event(
                "route",
                depth + 1,
                f"route -> {picked}  [call {run.client.calls}]",
                selected=[sel.chunk_id for sel in selections],
            )
        return selections

    def _inspect(self, run: _Run, chunk: Chunk, selection: Selection, depth: int) -> Finding:
        s = self.settings
        text = chunk.full_text()
        sent_tokens = count_tokens(text)
        truncated = sent_tokens > s.inspect_budget
        if truncated:
            text = truncate_to_tokens(text, s.inspect_budget)
            sent_tokens = count_tokens(text)
            self.tracer.event(
                "limit",
                depth + 1,
                f"truncate: {chunk.id} exceeds the {s.inspect_budget} tok inspect budget "
                "(at max depth) -> reading the first part only",
                chunk_id=chunk.id,
            )

        siblings = [c.id for c in run.by_id.values() if c.parent_id == chunk.parent_id and c.id != chunk.id]
        others = prompts.INSPECT_OTHERS_BLOCK.format(ids=", ".join(siblings[:12])) if siblings else ""

        payload = self._json_call(
            run,
            system=prompts.INSPECT_SYSTEM,
            prompt=prompts.INSPECT_USER.format(
                question=selection.sub_question,
                chunk_id=chunk.id,
                heading_path=chunk.path_str,
                text=text,
                others=others,
            ),
            required_keys=("found", "answer"),
            max_tokens=700,
        )
        run.stats.chunks_inspected += 1
        run.stats.document_tokens_read += sent_tokens
        run.inspected_ids.add(chunk.id)

        if payload is None:
            return Finding(
                chunk_id=chunk.id,
                heading_path=chunk.path_str,
                sub_question=selection.sub_question,
                found=False,
                answer="(the reader's reply could not be parsed)",
                confidence=0.0,
                depth=depth,
                truncated=truncated,
            )

        known_ids = set(run.by_id)
        finding = Finding(
            chunk_id=chunk.id,
            heading_path=chunk.path_str,
            sub_question=selection.sub_question,
            found=as_bool(payload.get("found")),
            answer=as_text(payload.get("answer")),
            evidence=as_str_list(payload.get("evidence"))[:3],
            confidence=as_float(payload.get("confidence")),
            needs_more=as_bool(payload.get("needs_more")),
            suggested_chunk_ids=[i for i in as_str_list(payload.get("suggested_chunk_ids")) if i in known_ids],
            depth=depth,
            truncated=truncated,
        )
        self.tracer.event(
            "inspect",
            depth + 1,
            f"inspect {chunk.id} ({sent_tokens} tok) [call {run.client.calls}] -> "
            f"found={str(finding.found).lower()} conf={finding.confidence:.2f} "
            f'"{_clip(finding.answer, 70)}"',
            chunk_id=chunk.id,
            found=finding.found,
        )
        return finding

    def _compress(
        self, run: _Run, chunk: Chunk, selection: Selection, sub: list[Finding], depth: int
    ) -> Finding:
        """Fold a sub-level's findings into one, so the parent's context stays small."""
        useful = [f for f in sub if f.found]
        if not sub:
            return Finding(
                chunk_id=chunk.id,
                heading_path=chunk.path_str,
                sub_question=selection.sub_question,
                found=False,
                answer="(nothing was found in this section's subsections)",
                confidence=0.0,
                depth=depth,
            )
        if len(sub) == 1:
            only = sub[0]
            only.chunk_id = chunk.id
            only.heading_path = only.heading_path or chunk.path_str
            return only

        payload = self._json_call(
            run,
            system=prompts.COMPRESS_SYSTEM,
            prompt=prompts.COMPRESS_USER.format(
                question=selection.sub_question,
                heading_path=chunk.path_str,
                findings=format_findings(sub),
            ),
            required_keys=("answer",),
            max_tokens=700,
        )
        if payload is None:  # fall back to concatenation rather than losing the work
            return Finding(
                chunk_id=chunk.id,
                heading_path=chunk.path_str,
                sub_question=selection.sub_question,
                found=bool(useful),
                answer=" ".join(f.answer for f in useful)[:1200],
                evidence=[q for f in useful for q in f.evidence][:3],
                confidence=max((f.confidence for f in useful), default=0.0),
                depth=depth,
            )

        self.tracer.event(
            "compress",
            depth + 1,
            f"compress: {len(sub)} findings from {chunk.id} -> 1  [call {run.client.calls}]",
            chunk_id=chunk.id,
        )
        return Finding(
            chunk_id=chunk.id,
            heading_path=chunk.path_str,
            sub_question=selection.sub_question,
            found=as_bool(payload.get("found"), default=bool(useful)),
            answer=as_text(payload.get("answer")),
            evidence=as_str_list(payload.get("evidence"))[:3],
            confidence=as_float(payload.get("confidence")),
            depth=depth,
        )

    def _synthesize(self, run: _Run, findings: list[Finding]) -> RLMResult:
        if not findings:
            return RLMResult(
                question=run.question,
                answer="No relevant sections were found for this question.",
                findings=findings,
            )

        self.tracer.event(
            "synthesize", 0, f"synthesize: {len(findings)} findings -> answer  [call {run.client.calls + 1}]"
        )
        payload = self._json_call(
            run,
            system=prompts.SYNTHESIS_SYSTEM,
            prompt=prompts.SYNTHESIS_USER.format(
                question=run.question, findings=format_findings(findings)
            ),
            required_keys=("answer",),
            max_tokens=1200,
        )
        if payload is None:
            self.tracer.event("synthesize", 0, "synthesis failed -> assembling findings directly")
            return RLMResult(
                question=run.question,
                answer=_fallback_answer(findings),
                citations=[f.heading_path for f in findings if f.found],
                findings=findings,
            )
        return RLMResult(
            question=run.question,
            answer=as_text(payload.get("answer")),
            citations=as_str_list(payload.get("citations")) or [f.heading_path for f in findings if f.found],
            gaps=as_text(payload.get("gaps")),
            findings=findings,
        )

    # ------------------------------------------------------------------

    def _json_call(
        self,
        run: _Run,
        *,
        system: str,
        prompt: str,
        required_keys: tuple[str, ...],
        max_tokens: int,
    ) -> dict | None:
        """One JSON model call. Returns ``None`` when the reply is unusable.

        Unusable replies are non-fatal by design: a failed route ends that level
        and synthesis proceeds with whatever was found. A demo that dies on one
        malformed reply is not a demo.
        """
        try:
            payload, _responses = generate_json(
                run.client,
                prompt,
                system=system,
                required_keys=required_keys,
                temperature=self.settings.temperature,
                max_tokens=max_tokens,
            )
            return payload
        except LLMJSONError as exc:
            self.tracer.event("error", 0, f"unusable model reply, continuing without it: {exc}")
            return None


# --------------------------------------------------------------------------


def _fallback_answer(findings: list[Finding]) -> str:
    """Deterministic answer assembled in Python when synthesis fails."""
    lines = ["The final synthesis call failed, so here are the raw findings:", ""]
    for finding in findings:
        if finding.found:
            lines.append(f"- **{finding.heading_path}** — {finding.answer}")
    if len(lines) == 2:
        lines.append("- No section reported a confident answer.")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


def _name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]

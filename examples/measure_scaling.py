"""Measure how the router's context scales with document size.

    python examples/measure_scaling.py

No API key and no model calls: this measures the *structure* of the approach,
not the quality of any answer. Everything here is deterministic and reproducible.

The question it answers: as a document grows, how much of it does the routing
step actually have to look at? That number is what bounds the reasoning context,
and it is the architectural claim the whole design rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings  # noqa: E402
from app.loaders.markdown_loader import parse_markdown  # noqa: E402
from app.rlm.chunker import build_chunk_tree, index_chunks, tree_depth  # noqa: E402
from app.rlm.context import build_index  # noqa: E402
from app.rlm.tokens import count_tokens  # noqa: E402

LOREM = (
    "The platform exposes a governed integration surface for external agents, "
    "metered per document and per action, with human approval gates available "
    "on write paths and regional data residency options in selected markets. "
)


def synthetic_markdown(sections: int, paragraphs_per_section: int) -> str:
    """A document with a realistic two-level heading structure."""
    out = ["# Synthetic Corpus\n"]
    for s in range(1, sections + 1):
        out.append(f"\n## Section {s}: platform capability area {s}\n")
        for p in range(1, paragraphs_per_section + 1):
            out.append(f"\n### {s}.{p} Subtopic {p}\n")
            out.append("\n" + (LOREM * 6) + "\n")
    return "".join(out)


def measure(markdown: str, settings) -> dict:
    sections = parse_markdown(markdown)
    tree = build_chunk_tree(sections, settings.chunk_target_tokens, settings.chunk_overlap)
    index = build_index(tree, settings.max_index_tokens)
    doc_tokens = sum(count_tokens(s.text) for s in sections)
    return {
        "doc_tokens": doc_tokens,
        "chunks": len(index_chunks(tree)),
        "depth": tree_depth(tree),
        "top_level": len(tree),
        "index_tokens": count_tokens(index),
        "index_share": count_tokens(index) / doc_tokens if doc_tokens else 0.0,
    }


def main() -> int:
    settings = load_settings(api_key="not-needed")

    print("How the routing context scales with document size")
    print("(model-independent: no API calls, deterministic)\n")
    print(f"  working budget      : {settings.max_context_tokens:,} tokens per call")
    print(f"  index cap           : {settings.max_index_tokens:,} tokens")
    print(f"  max doc tokens/call : {settings.inspect_budget:,} (inspect budget)")
    print(f"  max calls/question  : {settings.max_llm_calls}\n")

    header = f"{'document':>12} {'chunks':>8} {'depth':>6} {'index':>8} {'index as % of doc':>20}"
    print(header)
    print("-" * len(header))

    shapes = [(4, 2), (8, 3), (20, 4), (60, 6), (150, 8), (400, 10)]
    for sections, paras in shapes:
        stats = measure(synthetic_markdown(sections, paras), settings)
        print(
            f"{stats['doc_tokens']:>12,} {stats['chunks']:>8,} {stats['depth']:>6} "
            f"{stats['index_tokens']:>8,} {stats['index_share']:>19.2%}"
        )

    print(
        "\nThe index is capped, so its share of the document falls as the document grows."
        "\nRouting cost is a function of how many sections are shown, not of document size."
    )

    real = Path(__file__).resolve().parents[1] / "test_files" / "erp-ai-capabilities.md"
    if real.exists():
        stats = measure(real.read_text(encoding="utf-8"), settings)
        print(
            f"\nSample document ({real.name}):"
            f"\n  {stats['doc_tokens']:,} tokens -> {stats['chunks']} chunks, depth "
            f"{stats['depth']}, {stats['top_level']} top-level entries"
            f"\n  index = {stats['index_tokens']:,} tokens ({stats['index_share']:.1%} of the document)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

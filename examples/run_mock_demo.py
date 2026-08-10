"""End-to-end RLM run with no API key and no network.

    python examples/run_mock_demo.py

The reasoning is fake -- the offline client just reports the opening lines of
whatever it is shown. Everything around it is real: the chunk tree, the index,
the routing decisions, the descent into oversized sections, the budget guards
and the token accounting. This is the fastest way to see the control flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import render_result  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm.mock import DemoLLMClient  # noqa: E402
from app.loaders import load_document  # noqa: E402
from app.rlm.chunker import build_chunk_tree  # noqa: E402
from app.rlm.context import render_tree  # noqa: E402
from app.rlm.engine import RLMEngine  # noqa: E402
from app.trace import Tracer, configure_logging  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "test_files" / "erp-ai-capabilities.md"
QUESTION = "What does SAP charge for indirect access, and how does it gate external agents?"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings(openai_api_key="not-needed-for-mock")
    logger = configure_logging(settings.log_level, stream=sys.stdout)

    document = load_document(DOC)
    print(f"Document : {DOC.name}  ({document.total_tokens:,} tokens)")
    print(f"Budget   : {settings.max_context_tokens:,} document tokens per call")
    print(f"Question : {QUESTION}\n")

    tree = build_chunk_tree(document.sections, settings.chunk_target_tokens, settings.chunk_overlap)
    print("Chunk tree")
    print("----------")
    print(render_tree(tree))
    print()

    engine = RLMEngine(DemoLLMClient(), settings, Tracer(logger))
    result = engine.answer(document, QUESTION)
    print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

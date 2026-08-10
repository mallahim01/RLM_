"""A real run against the OpenAI API, over a set of increasingly demanding questions.

    python examples/run_erp_question.py                 # all questions
    python examples/run_erp_question.py 3               # just question 3
    python examples/run_erp_question.py "your question"

Needs OPENAI_API_KEY in `.env`. Each question costs a handful of small calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import render_result  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import build_openai_client  # noqa: E402
from app.loaders import load_document  # noqa: E402
from app.rlm.engine import RLMEngine  # noqa: E402
from app.trace import Tracer, configure_logging  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "test_files" / "erp-ai-capabilities.md"

# Each of these exercises a different shape of retrieval.
QUESTIONS = [
    ("breadth", "What AI capabilities are available across these platforms?"),
    ("filtering", "Which capabilities relate specifically to ERP transactions rather than chat?"),
    ("summary", "Summarise the most important capabilities in five bullets."),
    ("comparison", "Compare SAP and Odoo on how open they are to third-party agents."),
    ("deep fact", "How many free AI Units per month does Oracle include, and with what?"),
    ("multi-section", "If we can only build for one platform first, which and why? Use the "
                      "recommendations and the cost analysis together."),
]


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings()
    if not settings.has_api_key:
        print("No OPENAI_API_KEY found. Copy .env.example to .env and set it,", file=sys.stderr)
        print("or run examples/run_mock_demo.py to see the flow without a key.", file=sys.stderr)
        return 2

    logger = configure_logging(settings.log_level, stream=sys.stdout)
    document = load_document(DOC)
    engine = RLMEngine(
        build_openai_client(settings.openai_api_key, settings.model, settings.request_timeout),
        settings,
        Tracer(logger),
    )

    if argv and not argv[0].isdigit():
        selected = [("custom", " ".join(argv))]
    elif argv:
        selected = [QUESTIONS[int(argv[0]) - 1]]
    else:
        selected = QUESTIONS

    print(f"Document: {DOC.name} ({document.total_tokens:,} tokens)")
    print(f"Model: {settings.model} | window: {settings.max_context_tokens:,} tokens\n")

    for label, question in selected:
        print("=" * 78)
        print(f"[{label}] {question}")
        print("=" * 78)
        result = engine.answer(document, question)
        print(render_result(result))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

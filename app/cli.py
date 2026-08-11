"""Command-line interface.

    python -m app --mock                       # full recursive trace, no API key
    python -m app --question "..."             # one question, real model
    python -m app                              # interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from app import __version__
from app.config import PROVIDERS, ConfigError, Settings, load_settings
from app.loaders import UnsupportedFormatError, load_documents
from app.llm.base import LLMError, LLMClient
from app.llm.mock import DemoLLMClient
from app.models import Document, RLMResult
from app.rlm.chunker import build_chunk_tree
from app.rlm.context import render_tree
from app.rlm.engine import RLMEngine
from app.trace import Tracer, configure_logging

TEST_FILES = Path("test_files")
DEFAULT_DOC = TEST_FILES / "erp-ai-capabilities.md"
ALL_DOCS = [
    TEST_FILES / "erp-ai-capabilities.md",
    TEST_FILES / "WhatsApp Architecture and Technology Deep Dive.pdf",
    TEST_FILES / "knowledge-product-pakistan.docx",
]

_BANNER = """RLM Demo -- Recursive Language Model
===================================="""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Answer questions about a long document by recursively navigating it "
        "instead of stuffing it into one context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m app --mock                       run the whole flow offline, no API key
  python -m app --question "Which ERP is most open to third-party agents?"
  python -m app --all-docs --question "..."  merge every sample file into one corpus
  python -m app --doc a.md --doc b.pdf -Q "..."   merge specific files
  python -m app --provider gemini -Q "..."   pick a vendor explicitly
  python -m app --show-tree                  print the chunk tree and exit
  python -m app --json --question "..." | jq .stats
""",
    )
    parser.add_argument(
        "--doc",
        action="append",
        metavar="PATH",
        help=f"document to query; repeat to merge several into one corpus "
        f"(default: {DEFAULT_DOC.as_posix()})",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="query every sample file in test_files/ as a single merged corpus",
    )
    parser.add_argument("--question", "-Q", help="ask one question and exit; omit for an interactive session")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        help="model vendor (default: auto-detected from whichever API key is set)",
    )
    parser.add_argument("--model", help="model name (default: the provider's own default, or RLM_MODEL)")

    limits = parser.add_argument_group("recursion budget")
    limits.add_argument(
        "--max-context-tokens",
        type=int,
        help="working context budget: document tokens per call, not the model's window (default: 1500)",
    )
    limits.add_argument("--chunk-target-tokens", type=int, help="target leaf chunk size (default: 600)")
    limits.add_argument("--max-depth", type=int, help="deepest recursion level (default: 3)")
    limits.add_argument("--max-iterations", type=int, help="routing rounds per level (default: 2)")
    limits.add_argument("--max-llm-calls", type=int, help="total model calls per question (default: 25)")

    output = parser.add_argument_group("output")
    output.add_argument("--verbose", "-v", action="store_true", help="show per-call detail")
    output.add_argument("--quiet", "-q", action="store_true", help="hide the [RLM] trace")
    output.add_argument("--json", action="store_true", dest="as_json", help="print the full result as JSON")
    output.add_argument("--show-tree", action="store_true", help="print the chunk tree and exit")
    output.add_argument("--mock", action="store_true", help="use the offline demo client (no API key needed)")
    output.add_argument("--version", action="version", version=f"rlm {__version__}")
    return parser


def _settings_from(args: argparse.Namespace) -> Settings:
    overrides = {
        "provider": args.provider,
        "model": args.model,
        "max_context_tokens": args.max_context_tokens,
        "chunk_target_tokens": args.chunk_target_tokens,
        "max_depth": args.max_depth,
        "max_iterations": args.max_iterations,
        "max_llm_calls": args.max_llm_calls,
    }
    if args.verbose:
        overrides["log_level"] = "DEBUG"
    elif args.quiet:
        overrides["log_level"] = "WARNING"
    return load_settings(**overrides)


def _build_client(args: argparse.Namespace, settings: Settings) -> LLMClient | None:
    """Returns the client, or ``None`` when no key is configured."""
    if args.mock:
        return DemoLLMClient()
    if not settings.has_api_key:
        return None
    from app.llm import build_client

    return build_client(settings)


def _doc_paths(args: argparse.Namespace) -> list[Path]:
    if args.all_docs:
        return list(ALL_DOCS)
    if args.doc:
        return [Path(p) for p in args.doc]
    return [DEFAULT_DOC]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_result(result: RLMResult) -> str:
    lines = ["", "Answer", "------", result.answer.strip() or "(no answer)"]
    if result.citations:
        lines += ["", "Sources", "-------"]
        lines += [f"  - {c}" for c in dict.fromkeys(result.citations)]
    if result.gaps:
        lines += ["", "Gaps", "----", f"  {result.gaps}"]

    s = result.stats
    lines += [
        "",
        f"Stats: {s.llm_calls} LLM calls | {s.tokens_in:,} in / {s.tokens_out:,} out | "
        f"depth {s.max_depth_reached} | {s.wall_time_s:.1f}s",
        f"       read {s.document_tokens_read:,} of {s.document_tokens:,} document tokens "
        f"({s.context_efficiency:.1%}) across {s.chunks_inspected} of {s.chunks_total} chunks",
    ]
    if result.stopped_reason != "completed":
        lines.append(f"       stopped early: {result.stopped_reason}")
    return "\n".join(lines)


def _emit(result: RLMResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_result(result))


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

_REPL_HELP = """commands:
  :help     show this
  :tree     print the chunk tree
  :stats    stats from the last answer
  :trace    replay the last trace
  :quit     exit  (Ctrl-D also works)
anything else is treated as a question."""


def repl(engine: RLMEngine, document: Document, as_json: bool) -> int:
    print(f"\nAsk questions about {_short(document.path)}. Type :help for commands, :quit to exit.\n")
    last: RLMResult | None = None
    while True:
        try:
            line = input("rlm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            return 0
        if line == ":help":
            print(_REPL_HELP)
            continue
        if line == ":tree":
            tree = build_chunk_tree(
                document.sections, engine.settings.chunk_target_tokens, engine.settings.chunk_overlap
            )
            print(render_tree(tree))
            continue
        if line == ":stats":
            print(render_result(last).split("Stats:")[-1].strip() if last else "No answer yet.")
            continue
        if line == ":trace":
            for event in last.trace if last else []:
                print(f"[RLM] {'  ' * event.depth}{event.message}")
            if last is None:
                print("No answer yet.")
            continue

        try:
            last = engine.answer(document, line)
        except KeyboardInterrupt:
            print("\n(cancelled)")
            continue
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue
        _emit(last, as_json)
        print()


def main(argv: Sequence[str] | None = None) -> int:
    # Trace lines are ASCII, but document text is not; legacy Windows consoles
    # default to cp1252 and would raise on the first en-dash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    args = build_parser().parse_args(argv)

    try:
        settings = _settings_from(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    # In --json mode stdout must stay pure JSON, so the trace goes to stderr.
    configure_logging(settings.log_level, stream=sys.stderr)
    tracer = Tracer(logging.getLogger("rlm"), enabled=not args.quiet)

    try:
        document = load_documents(_doc_paths(args))
    except (FileNotFoundError, UnsupportedFormatError, ValueError) as exc:
        print(f"could not load document: {exc}", file=sys.stderr)
        return 2

    if args.show_tree:
        tree = build_chunk_tree(document.sections, settings.chunk_target_tokens, settings.chunk_overlap)
        print(render_tree(tree))
        return 0

    try:
        client = _build_client(args, settings)
    except LLMError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    if client is None:
        keys = " or ".join(sorted(p.env_key for p in PROVIDERS.values()))
        print(
            f"No API key found. Set {keys} in .env,\n"
            "  or run with --mock to use the offline demo client.",
            file=sys.stderr,
        )
        return 2

    engine = RLMEngine(client=client, settings=settings, tracer=tracer)

    if not args.question:
        if not args.as_json:
            print(_BANNER)
            print(f"\nDocument: {_short(document.path)}  ({document.total_tokens:,} tokens)")
            engine_label = "mock (offline)" if args.mock else f"{settings.provider}/{settings.model}"
            print(f"Model: {engine_label}")
            print(
                f"Working budget: {settings.max_context_tokens:,} document tokens per call"
            )
        return repl(engine, document, args.as_json)

    try:
        result = engine.answer(document, args.question)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _emit(result, args.as_json)
    return 0


def _short(path: str) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return path

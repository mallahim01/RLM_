"""Run observability.

One object, two outputs. Every interesting moment goes through ``Tracer.event``,
which both prints a ``[RLM]`` line and appends a ``TraceEvent`` to the structured
trace returned in ``RLMResult``. A single call site means the console output and
the ``--json`` trace can never silently drift apart.

Indentation by recursion depth is the point: it is what makes the recursion
visible at a glance. Output stays ASCII-only so it survives legacy Windows
terminals.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from app.models import TraceEvent

LOGGER_NAME = "rlm"


def configure_logging(level: str = "INFO", stream: Any = None) -> logging.Logger:
    """Set up the dedicated `rlm` logger with a bare message format.

    Trace lines carry their own ``[RLM]`` marker, so no timestamps or level
    prefixes -- the output is meant to be read, and pasted into a README.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


class Tracer:
    """Collects trace events and mirrors them to the log."""

    def __init__(self, logger: logging.Logger | None = None, enabled: bool = True) -> None:
        self._logger = logger or logging.getLogger(LOGGER_NAME)
        self._enabled = enabled
        self._events: list[TraceEvent] = []
        self._start = time.perf_counter()

    @property
    def events(self) -> list[TraceEvent]:
        return self._events

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self) -> None:
        self._events = []
        self._start = time.perf_counter()

    def event(self, kind: str, depth: int, message: str, **data: Any) -> TraceEvent:
        """Record one moment: append it, and print it indented by depth."""
        evt = TraceEvent(kind=kind, depth=depth, message=message, elapsed_s=self.elapsed, data=data)
        self._events.append(evt)
        if self._enabled:
            self._logger.info("[RLM] %s%s", "  " * depth, message)
        return evt

    def detail(self, depth: int, message: str) -> None:
        """Verbose-only line. Not recorded -- use ``event`` for anything durable."""
        if self._enabled:
            self._logger.debug("[RLM] %s%s", "  " * depth, message)

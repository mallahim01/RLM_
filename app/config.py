"""Settings, loaded from `.env` / environment with CLI overrides on top.

Precedence: CLI flag > environment variable > dataclass default.

The API key lives here but is never printed: ``__repr__`` masks it, so no
accidental ``print(settings)`` or exception dump can leak it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

# Rough cost of the fixed scaffolding around a leaf call: system prompt,
# instructions, heading path, the sub-question, and room for the JSON reply.
# Kept as a named constant so the budget arithmetic below is readable.
PROMPT_OVERHEAD_TOKENS = 350


class ConfigError(ValueError):
    """Raised when settings contradict each other. Fails at startup, not mid-run."""


@dataclass(frozen=True, slots=True, repr=False)
class Settings:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"

    # The headline knob: the simulated context window.
    max_context_tokens: int = 1500

    # Chunking.
    chunk_target_tokens: int = 600
    chunk_overlap: int = 60

    # Recursion guards -- three independent ways for a run to terminate.
    max_depth: int = 3
    max_iterations: int = 2
    max_llm_calls: int = 25
    max_selections_per_round: int = 3

    # What the router is allowed to see, and when lexical pre-filtering kicks in.
    max_index_tokens: int = 800
    max_index_entries: int = 12
    prefilter_threshold: int = 12

    temperature: float = 0.0
    request_timeout: float = 60.0
    log_level: str = "INFO"
    tokenizer: str = "auto"  # auto | heuristic

    @property
    def inspect_budget(self) -> int:
        """Max document tokens allowed in a single leaf (inspect) call."""
        return max(200, self.max_context_tokens - PROMPT_OVERHEAD_TOKENS)

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key.strip())

    def validate(self) -> None:
        if self.max_context_tokens < 400:
            raise ConfigError(
                f"max_context_tokens={self.max_context_tokens} is too small to fit a "
                "prompt; use 400 or more."
            )
        if self.chunk_target_tokens < 50:
            raise ConfigError("chunk_target_tokens must be at least 50.")
        if self.chunk_overlap >= self.chunk_target_tokens:
            raise ConfigError(
                f"chunk_overlap={self.chunk_overlap} must be smaller than "
                f"chunk_target_tokens={self.chunk_target_tokens}."
            )
        if self.chunk_overlap < 0:
            raise ConfigError("chunk_overlap cannot be negative.")
        if self.chunk_target_tokens > self.inspect_budget:
            raise ConfigError(
                f"chunk_target_tokens={self.chunk_target_tokens} exceeds the inspect "
                f"budget of {self.inspect_budget} implied by "
                f"max_context_tokens={self.max_context_tokens}. Lower the chunk size "
                "or raise the context window."
            )
        if self.max_depth < 1:
            raise ConfigError("max_depth must be at least 1.")
        if self.max_iterations < 1:
            raise ConfigError("max_iterations must be at least 1.")
        if self.max_llm_calls < 2:
            raise ConfigError("max_llm_calls must be at least 2 (one route, one answer).")
        if self.max_selections_per_round < 1:
            raise ConfigError("max_selections_per_round must be at least 1.")
        if self.tokenizer not in ("auto", "heuristic"):
            raise ConfigError("tokenizer must be 'auto' or 'heuristic'.")

    def masked_key(self) -> str:
        key = self.openai_api_key.strip()
        if not key:
            return "(unset)"
        tail = key[-4:] if len(key) > 8 else "****"
        return f"sk-...{tail} (set)"

    def __repr__(self) -> str:  # never leak the key
        shown = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if f.name != "openai_api_key"
        )
        return f"Settings(openai_api_key={self.masked_key()!r}, {shown})"


_ENV_PREFIX = "RLM_"


def _env(name: str) -> str | None:
    raw = os.getenv(_ENV_PREFIX + name.upper())
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _coerce(name: str, raw: str, target_type: Any) -> Any:
    try:
        if target_type is int:
            return int(raw)
        if target_type is float:
            return float(raw)
        return raw
    except ValueError as exc:
        raise ConfigError(
            f"{_ENV_PREFIX}{name.upper()}={raw!r} is not a valid {target_type.__name__}."
        ) from exc


def load_settings(**overrides: Any) -> Settings:
    """Build settings from `.env` + environment, then apply non-None overrides."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # python-dotenv is optional at import time
        pass

    values: dict[str, Any] = {}
    for f in fields(Settings):
        if f.name == "openai_api_key":
            continue
        raw = _env(f.name)
        if raw is not None:
            # `from __future__ import annotations` makes f.type a string, so infer
            # the target type from the default instead. Every field has one.
            values[f.name] = _coerce(f.name, raw, type(f.default))

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    values["openai_api_key"] = api_key

    for key, value in overrides.items():
        if value is None:
            continue
        if key not in {f.name for f in fields(Settings)}:
            raise ConfigError(f"Unknown setting override: {key!r}")
        values[key] = value

    settings = Settings(**values)
    settings.validate()

    # The tokenizer module reads this env var directly, so keep the two in sync
    # when the choice arrived via a CLI override rather than the environment.
    os.environ["RLM_TOKENIZER"] = settings.tokenizer
    return settings

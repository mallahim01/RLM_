"""Settings, loaded from `.env` / environment with CLI overrides on top.

Precedence: CLI flag > environment variable > dataclass default.

The API key lives here but is never printed: ``__repr__`` masks it, so no
accidental ``print(settings)`` or exception dump can leak it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, NamedTuple

# Rough cost of the fixed scaffolding around a leaf call: system prompt,
# instructions, heading path, the sub-question, and room for the JSON reply.
# Kept as a named constant so the budget arithmetic below is readable.
PROMPT_OVERHEAD_TOKENS = 350


class ConfigError(ValueError):
    """Raised when settings contradict each other. Fails at startup, not mid-run."""


class Provider(NamedTuple):
    """Everything that differs between one model vendor and another."""

    name: str
    env_key: str
    default_model: str
    base_url: str  # "" means the SDK's own default


# Gemini publishes an OpenAI-compatible endpoint, so a base_url is the entire
# difference. That keeps this project at one HTTP client and no extra
# dependency -- adding a third provider should be one more line here.
PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        name="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        base_url="",
    ),
    "gemini": Provider(
        name="gemini",
        env_key="GEMINI_API_KEY",
        # A rolling alias rather than a pinned version: Google retires numbered
        # Gemini models fairly quickly, and a pinned id turns into a 404 for
        # anyone cloning this later. Lite is the cheapest tier, which suits a
        # demo that makes a dozen small calls per question.
        default_model="gemini-flash-lite-latest",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
}

# Checked in order when no provider is named explicitly.
_AUTODETECT_ORDER = ("openai", "gemini")


@dataclass(frozen=True, slots=True, repr=False)
class Settings:
    provider: str = "openai"
    api_key: str = ""
    model: str = ""  # empty means "the provider's default"
    base_url: str = ""

    # The working context budget: how much document text goes into one call.
    # Not the model's context window -- the amount we choose to show it at once,
    # because reasoning quality degrades with context length long before the
    # window is full. See README, "The working context budget".
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
    # How many times to wait out a rate limit. Free tiers meter per minute, and
    # an RLM makes many small calls quickly, so this is the difference between
    # a run completing slowly and failing outright.
    rate_limit_retries: int = 3
    log_level: str = "INFO"
    tokenizer: str = "auto"  # auto | heuristic

    @property
    def inspect_budget(self) -> int:
        """Max document tokens allowed in a single leaf (inspect) call."""
        return max(200, self.max_context_tokens - PROMPT_OVERHEAD_TOKENS)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def provider_spec(self) -> Provider:
        return PROVIDERS[self.provider]

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            known = ", ".join(sorted(PROVIDERS))
            raise ConfigError(f"Unknown provider {self.provider!r}. Known providers: {known}.")
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
                "or raise the working budget."
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
        key = self.api_key.strip()
        if not key:
            return "(unset)"
        tail = key[-4:] if len(key) > 8 else "****"
        return f"...{tail} (set, {len(key)} chars)"

    def __repr__(self) -> str:  # never leak the key
        shown = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if f.name != "api_key"
        )
        return f"Settings(api_key={self.masked_key()!r}, {shown})"


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


def detect_provider() -> str:
    """Pick a provider from whichever vendor key is present."""
    named = _env("provider")
    if named:
        if named.lower() not in PROVIDERS:
            known = ", ".join(sorted(PROVIDERS))
            raise ConfigError(f"Unknown RLM_PROVIDER={named!r}. Known providers: {known}.")
        return named.lower()
    for name in _AUTODETECT_ORDER:
        if os.getenv(PROVIDERS[name].env_key, "").strip():
            return name
    return "openai"


def load_settings(**overrides: Any) -> Settings:
    """Build settings from `.env` + environment, then apply non-None overrides."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # python-dotenv is optional at import time
        pass

    values: dict[str, Any] = {}
    for f in fields(Settings):
        if f.name in ("api_key", "provider"):
            continue
        raw = _env(f.name)
        if raw is not None:
            # `from __future__ import annotations` makes f.type a string, so infer
            # the target type from the default instead. Every field has one.
            values[f.name] = _coerce(f.name, raw, type(f.default))

    provider_name = overrides.pop("provider", None) or detect_provider()
    if provider_name not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"Unknown provider {provider_name!r}. Known providers: {known}.")
    spec = PROVIDERS[provider_name]

    values["provider"] = provider_name
    values["api_key"] = os.getenv(spec.env_key, "").strip()
    values.setdefault("base_url", spec.base_url)

    for key, value in overrides.items():
        if value is None:
            continue
        if key not in {f.name for f in fields(Settings)}:
            raise ConfigError(f"Unknown setting override: {key!r}")
        values[key] = value

    # A model was not chosen anywhere -> use the provider's own default, so
    # switching provider does not silently keep the other vendor's model name.
    if not values.get("model"):
        values["model"] = spec.default_model

    settings = Settings(**values)
    settings.validate()

    # The tokenizer module reads this env var directly, so keep the two in sync
    # when the choice arrived via a CLI override rather than the environment.
    os.environ["RLM_TOKENIZER"] = settings.tokenizer
    return settings

"""LLM clients. ``openai_client`` is imported lazily so the SDK stays optional."""

from __future__ import annotations

from app.llm.base import (
    LLMClient,
    LLMError,
    LLMJSONError,
    LLMResponse,
    extract_json,
    generate_json,
)
from app.llm.mock import DemoLLMClient, PatternLLMClient, ScriptedLLMClient

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMJSONError",
    "LLMResponse",
    "extract_json",
    "generate_json",
    "DemoLLMClient",
    "PatternLLMClient",
    "ScriptedLLMClient",
    "build_client",
]


def build_client(settings):
    """Construct the real client for the configured provider.

    Imported lazily so the SDK stays off the hot path and out of the tests.
    """
    from app.llm.openai_client import OpenAIClient

    spec = settings.provider_spec
    return OpenAIClient(
        api_key=settings.api_key,
        model=settings.model,
        timeout=settings.request_timeout,
        base_url=settings.base_url,
        provider=settings.provider,
        env_key=spec.env_key,
        rate_limit_retries=settings.rate_limit_retries,
    )

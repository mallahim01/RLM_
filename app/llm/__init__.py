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
    "build_openai_client",
]


def build_openai_client(api_key: str, model: str, timeout: float = 60.0):
    """Construct the real client. Imported here to keep the SDK off the hot path."""
    from app.llm.openai_client import OpenAIClient

    return OpenAIClient(api_key=api_key, model=model, timeout=timeout)

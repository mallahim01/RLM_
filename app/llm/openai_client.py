"""OpenAI implementation of ``LLMClient``.

Isolated in its own module so that importing the engine never imports the SDK --
which is why the whole test suite runs without network access or an API key.

No retry library here. The OpenAI SDK already retries connection failures, 429s
and 5xx with backoff; the only retry this project adds is semantic (asking the
model to repair malformed JSON), and that lives in ``llm/base.py``.
"""

from __future__ import annotations

from app.llm.base import LLMError, LLMResponse


class OpenAIClient:
    """Thin wrapper over ``chat.completions``. One method, no cleverness."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError(
                "No OpenAI API key. Copy .env.example to .env and set OPENAI_API_KEY, "
                "or run with --mock to use the offline demo client."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The `openai` package is not installed. Run:\n"
                "    pip install -r requirements.txt"
            ) from exc

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # keep vendor exception types out of the engine
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = completion.usage
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )

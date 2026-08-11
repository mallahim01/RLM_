"""Chat-completions implementation of ``LLMClient``.

Named for OpenAI because that is the API shape, but it serves any vendor
exposing an OpenAI-compatible ``/chat/completions`` endpoint -- Gemini does,
which is why supporting a second provider here costs a ``base_url`` rather
than a second SDK.

Isolated in its own module so that importing the engine never imports the SDK --
which is why the whole test suite runs without network access or an API key.

No retry library here. The OpenAI SDK already retries connection failures, 429s
and 5xx with backoff; the only retry this project adds is semantic (asking the
model to repair malformed JSON), and that lives in ``llm/base.py``.
"""

from __future__ import annotations

import logging
import re
import time

from app.llm.base import LLMError, LLMResponse

_LOG = logging.getLogger("rlm")

# Free tiers meter by requests-per-minute, and an RLM makes a dozen small calls
# in a few seconds -- precisely the shape a per-minute quota punishes. The SDK's
# own backoff tops out in single-digit seconds, while these APIs routinely ask
# for 20-60. So rate limits get their own wait, honouring the delay the server
# actually asked for.
# Providers phrase it differently in the same message: Gemini says both
# "Please retry in 24.35s" and "'retryDelay': '24s'". Accept either.
_RETRY_DELAY = re.compile(r"retry(?:_?delay|[-_ ]after|\s+in)\D{0,4}(\d+(?:\.\d+)?)", re.I)
_MAX_RATE_LIMIT_WAIT = 65.0


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    return type(exc).__name__ == "RateLimitError"


def _retry_after_seconds(exc: Exception, default: float = 10.0) -> float:
    """How long the server asked us to wait, from the header or the message."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return min(float(raw), _MAX_RATE_LIMIT_WAIT)
            except (TypeError, ValueError):
                pass
    match = _RETRY_DELAY.search(str(exc))
    if match:
        return min(float(match.group(1)), _MAX_RATE_LIMIT_WAIT)
    return default


class OpenAIClient:
    """Thin wrapper over ``chat.completions``. One method, no cleverness."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 3,
        base_url: str = "",
        provider: str = "openai",
        env_key: str = "OPENAI_API_KEY",
        rate_limit_retries: int = 3,
        sleep = time.sleep,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError(
                f"No API key for provider {provider!r}. Copy .env.example to .env and "
                f"set {env_key}, or run with --mock to use the offline demo client."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The `openai` package is not installed. Run:\n"
                "    pip install -r requirements.txt"
            ) from exc

        self.model = model
        self.provider = provider
        self.rate_limit_retries = rate_limit_retries
        self._sleep = sleep
        kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

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

        completion = self._create_with_rate_limit_retry(kwargs)

        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = completion.usage
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )

    def _create_with_rate_limit_retry(self, kwargs: dict[str, object]):
        """Send the request, waiting out rate limits for as long as asked.

        Transport errors are already handled by the SDK. This adds only the
        rate-limit case, where the right wait is the one the server names and
        is far longer than any generic backoff would choose.
        """
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # keep vendor exception types out of the engine
                if attempt < self.rate_limit_retries and _is_rate_limit(exc):
                    delay = _retry_after_seconds(exc)
                    _LOG.info(
                        "[RLM] rate limited by %s, waiting %.0fs (attempt %d of %d)",
                        self.provider,
                        delay,
                        attempt + 1,
                        self.rate_limit_retries,
                    )
                    self._sleep(delay)
                    continue
                raise LLMError(f"{self.provider} request failed: {exc}") from exc
        raise LLMError(f"{self.provider} request failed: rate limit not cleared")

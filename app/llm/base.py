"""The LLM boundary.

The engine depends on this ``Protocol`` and nothing else, so it never imports a
vendor SDK and never sees a vendor exception type. Swapping OpenAI for another
provider means writing one class with one method.

The other job of this module is making JSON replies survivable. Models return
malformed or fence-wrapped JSON often enough that a demo which crashes on it is
not a demo. Four layers defend against that: JSON response mode, an extraction
ladder, required-key validation, and exactly one repair retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class LLMError(RuntimeError):
    """Any failure talking to a model. Vendor exceptions are re-raised as this."""


class LLMJSONError(LLMError):
    """The model's reply could not be read as the JSON object we asked for."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    """One method. Implemented by ``OpenAIClient`` and by the test doubles."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]  # drop the opening ``` or ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _first_json_object(text: str) -> str | None:
    """Find the outermost balanced ``{...}``, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply. Raises ``LLMJSONError``."""
    for candidate in (text.strip(), _strip_fences(text), _first_json_object(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMJSONError(f"No JSON object found in model reply: {text[:200]!r}")


_REPAIR_TEMPLATE = (
    "\n\nYour previous reply could not be used: {problem}\n"
    "Reply again with a single valid JSON object and nothing else. "
    "It must contain these keys: {keys}."
)


def generate_json(
    client: LLMClient,
    prompt: str,
    *,
    system: str,
    required_keys: Sequence[str] = (),
    temperature: float = 0.0,
    max_tokens: int = 800,
    retries: int = 1,
) -> tuple[dict[str, Any], list[LLMResponse]]:
    """Ask for JSON, validate it has the keys we need, repair once if not.

    Returns the parsed object and every response involved, so the caller can
    account for the tokens spent on a repair attempt as well as the first try.
    """
    responses: list[LLMResponse] = []
    attempt_prompt = prompt
    problem = ""

    for attempt in range(retries + 1):
        response = client.generate(
            attempt_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        responses.append(response)
        try:
            payload = extract_json(response.text)
        except LLMJSONError as exc:
            problem = str(exc)
        else:
            missing = [key for key in required_keys if key not in payload]
            if not missing:
                return payload, responses
            problem = f"missing required key(s): {', '.join(missing)}"

        if attempt < retries:
            attempt_prompt = prompt + _REPAIR_TEMPLATE.format(
                problem=problem, keys=", ".join(required_keys) or "the ones requested"
            )

    raise LLMJSONError(problem or "model did not return usable JSON")


# --------------------------------------------------------------------------
# small coercion helpers -- never trust the shape of a model's reply
# --------------------------------------------------------------------------


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def as_float(value: Any, default: float = 0.5, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def as_text(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return default
    return str(value).strip()

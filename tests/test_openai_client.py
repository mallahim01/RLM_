"""``OpenAIClient`` request construction and response handling, with a stub SDK.

The real API is covered by ``test_integration_openai.py``, which costs money and
is opt-in. These tests pin the parts that would otherwise only ever be exercised
by a paid call: which arguments get sent, how usage is read back, and that
vendor exceptions never escape as vendor types.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.llm.base import LLMError


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Completion:
    def __init__(self, content="{}", tokens_in=11, tokens_out=7):
        self.choices = [_Choice(content)]
        self.usage = _Usage(tokens_in, tokens_out)


class _FakeOpenAI:
    """Records the kwargs it was constructed and called with."""

    last_init: dict = {}
    last_call: dict = {}
    raises: Exception | None = None
    completion = _Completion()

    def __init__(self, **kwargs):
        type(self).last_init = kwargs
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        type(self).last_call = kwargs
        if type(self).raises:
            raise type(self).raises
        return type(self).completion


@pytest.fixture
def client(monkeypatch):
    _FakeOpenAI.raises = None
    _FakeOpenAI.completion = _Completion()
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))

    from app.llm.openai_client import OpenAIClient

    return OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", timeout=12.0)


def test_constructor_passes_timeout_and_retries_to_the_sdk(client):
    assert _FakeOpenAI.last_init["api_key"] == "sk-fake"
    assert _FakeOpenAI.last_init["timeout"] == 12.0
    assert _FakeOpenAI.last_init["max_retries"] == 3, "the SDK does our transport retries"


def test_system_prompt_becomes_the_first_message(client):
    client.generate("the question", system="the role")

    messages = _FakeOpenAI.last_call["messages"]
    assert messages == [
        {"role": "system", "content": "the role"},
        {"role": "user", "content": "the question"},
    ]


def test_no_system_message_when_none_is_given(client):
    client.generate("only a prompt")
    assert [m["role"] for m in _FakeOpenAI.last_call["messages"]] == ["user"]


def test_json_mode_sets_the_response_format(client):
    client.generate("p", json_mode=True)
    assert _FakeOpenAI.last_call["response_format"] == {"type": "json_object"}

    client.generate("p", json_mode=False)
    assert "response_format" not in _FakeOpenAI.last_call


def test_model_and_sampling_arguments_are_forwarded(client):
    client.generate("p", temperature=0.3, max_tokens=123)

    assert _FakeOpenAI.last_call["model"] == "gpt-4o-mini"
    assert _FakeOpenAI.last_call["temperature"] == 0.3
    assert _FakeOpenAI.last_call["max_tokens"] == 123


def test_usage_is_read_from_the_response_not_estimated(client):
    _FakeOpenAI.completion = _Completion(content='{"a": 1}', tokens_in=321, tokens_out=45)

    response = client.generate("p")

    assert response.text == '{"a": 1}'
    assert response.tokens_in == 321
    assert response.tokens_out == 45


def test_an_empty_content_field_yields_an_empty_string_not_none(client):
    _FakeOpenAI.completion = _Completion(content=None)
    assert client.generate("p").text == ""


def test_vendor_exceptions_are_re_raised_as_llm_error(client):
    _FakeOpenAI.raises = RuntimeError("429 insufficient_quota")

    with pytest.raises(LLMError) as excinfo:
        client.generate("p")

    assert "OpenAI request failed" in str(excinfo.value)
    assert "insufficient_quota" in str(excinfo.value)


def test_a_missing_key_is_refused_before_the_sdk_is_touched(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    from app.llm.openai_client import OpenAIClient

    with pytest.raises(LLMError) as excinfo:
        OpenAIClient(api_key="   ")

    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "--mock" in str(excinfo.value)


def test_a_missing_sdk_gives_an_actionable_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    from app.llm.openai_client import OpenAIClient

    with pytest.raises(LLMError) as excinfo:
        OpenAIClient(api_key="sk-fake")

    assert "requirements.txt" in str(excinfo.value)

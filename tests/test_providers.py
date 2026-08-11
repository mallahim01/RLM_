"""Provider selection, and that the right key and endpoint reach the client.

Gemini is reached through its OpenAI-compatible endpoint, so "adding a provider"
is a base_url plus a default model. These tests pin that wiring, because getting
it wrong shows up only as a live API failure -- which is exactly what a free-tier
key cannot afford to spend calls discovering.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.config import PROVIDERS, ConfigError, detect_provider, load_settings
from app.llm.base import LLMError
from tests.test_openai_client import _FakeOpenAI


@pytest.fixture(autouse=True)
def no_ambient_keys(monkeypatch):
    """`.env` is real on a dev machine -- neutralise it for these tests."""
    for spec in PROVIDERS.values():
        monkeypatch.delenv(spec.env_key, raising=False)
    monkeypatch.delenv("RLM_PROVIDER", raising=False)
    monkeypatch.delenv("RLM_MODEL", raising=False)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: None, raising=False)
    # load_settings imports dotenv lazily inside the function; block the file read.
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda **k: None))


def test_every_provider_declares_what_it_needs():
    for name, spec in PROVIDERS.items():
        assert spec.name == name
        assert spec.env_key
        assert spec.default_model


def test_openai_is_detected_from_its_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert detect_provider() == "openai"

    settings = load_settings()
    assert settings.provider == "openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.base_url == "", "the SDK's own default endpoint"


def test_gemini_is_detected_from_its_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert detect_provider() == "gemini"

    settings = load_settings()
    assert settings.provider == "gemini"
    assert settings.api_key == "gemini-key"
    assert settings.model.startswith("gemini-")
    assert "generativelanguage.googleapis.com" in settings.base_url


def test_openai_wins_when_both_keys_are_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert load_settings().provider == "openai"


def test_an_explicit_provider_overrides_detection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    settings = load_settings(provider="gemini")

    assert settings.provider == "gemini"
    assert settings.api_key == "gemini-key", "the key must follow the provider"
    assert settings.model.startswith("gemini-")


def test_rlm_provider_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("RLM_PROVIDER", "gemini")
    assert detect_provider() == "gemini"


def test_an_unknown_provider_is_rejected(monkeypatch):
    with pytest.raises(ConfigError, match="Unknown provider"):
        load_settings(provider="llamafile")

    monkeypatch.setenv("RLM_PROVIDER", "nonsense")
    with pytest.raises(ConfigError, match="Unknown RLM_PROVIDER"):
        load_settings()


def test_an_explicit_model_survives_a_provider_switch(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    settings = load_settings(model="gemini-2.5-flash")
    assert settings.model == "gemini-2.5-flash"


def test_switching_provider_does_not_keep_the_other_vendors_model(monkeypatch):
    """The failure this prevents: asking Gemini for 'gpt-4o-mini'."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert load_settings(provider="openai").model == "gpt-4o-mini"
    assert load_settings(provider="gemini").model.startswith("gemini-")


def test_a_missing_key_is_reported_against_the_right_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    settings = load_settings(provider="gemini")

    assert not settings.has_api_key
    assert settings.provider_spec.env_key == "GEMINI_API_KEY"


def test_the_key_is_masked_in_repr(monkeypatch):
    # Deliberately not shaped like a real key prefix: a realistic-looking
    # literal in a public repo trips other people's secret scanners.
    monkeypatch.setenv("GEMINI_API_KEY", "NOT-A-REAL-KEY-PLACEHOLDER-1234")
    settings = load_settings()

    text = repr(settings)
    assert "SUPERSECRET" not in text
    assert "1234" in text, "a short suffix is fine for identifying which key is loaded"
    assert settings.masked_key().startswith("...")


# --------------------------------------------------------------------------
# what actually reaches the SDK
# --------------------------------------------------------------------------


@pytest.fixture
def fake_sdk(monkeypatch):
    _FakeOpenAI.raises = None
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    return _FakeOpenAI


def test_gemini_client_is_pointed_at_the_compatible_endpoint(monkeypatch, fake_sdk):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    from app.llm import build_client

    client = build_client(load_settings())

    assert fake_sdk.last_init["api_key"] == "gemini-key"
    assert fake_sdk.last_init["base_url"].endswith("/v1beta/openai/")
    assert client.model.startswith("gemini-")


def test_openai_client_gets_no_base_url_override(monkeypatch, fake_sdk):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    from app.llm import build_client

    build_client(load_settings())

    assert "base_url" not in fake_sdk.last_init, "must not pin OpenAI to a hardcoded host"


def test_provider_name_appears_in_error_messages(monkeypatch, fake_sdk):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    from app.llm import build_client

    client = build_client(load_settings())
    fake_sdk.raises = RuntimeError("401 API_KEY_INVALID")

    with pytest.raises(LLMError, match="gemini request failed"):
        client.generate("p")


def test_a_missing_key_names_the_providers_env_var(fake_sdk):
    from app.llm.openai_client import OpenAIClient

    with pytest.raises(LLMError) as excinfo:
        OpenAIClient(api_key="", provider="gemini", env_key="GEMINI_API_KEY")

    assert "GEMINI_API_KEY" in str(excinfo.value)
    assert "gemini" in str(excinfo.value)

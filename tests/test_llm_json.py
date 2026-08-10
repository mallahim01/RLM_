from __future__ import annotations

import pytest

from app.llm.base import (
    LLMJSONError,
    as_bool,
    as_float,
    as_str_list,
    as_text,
    extract_json,
    generate_json,
)
from app.llm.mock import ScriptedLLMClient


def test_bare_json_parses():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_parses():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_wrapped_in_prose_parses():
    reply = 'Sure! Here is the result:\n{"a": 1, "b": [2, 3]}\nHope that helps.'
    assert extract_json(reply) == {"a": 1, "b": [2, 3]}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    assert extract_json('prefix {"a": "a } brace", "b": 2} suffix') == {"a": "a } brace", "b": 2}


def test_unparseable_replies_raise():
    for reply in ("not json at all", "", '{"unbalanced": '):
        with pytest.raises(LLMJSONError):
            extract_json(reply)


def test_a_bare_json_array_is_not_accepted():
    with pytest.raises(LLMJSONError):
        extract_json("[1, 2, 3]")


def test_generate_json_repairs_a_malformed_first_reply():
    client = ScriptedLLMClient(["sorry, no JSON here", {"selections": []}])
    payload, responses = generate_json(client, "prompt", system="sys", required_keys=("selections",))

    assert payload == {"selections": []}
    assert len(responses) == 2
    assert "could not be used" in client.calls[1]["prompt"]


def test_generate_json_repairs_a_missing_required_key():
    client = ScriptedLLMClient([{"wrong": 1}, {"answer": "ok"}])
    payload, _ = generate_json(client, "prompt", system="sys", required_keys=("answer",))
    assert payload["answer"] == "ok"
    assert "missing required key" in client.calls[1]["prompt"]


def test_generate_json_gives_up_after_one_repair():
    client = ScriptedLLMClient(["nope", "still nope"])
    with pytest.raises(LLMJSONError):
        generate_json(client, "prompt", system="sys", required_keys=("answer",))
    assert client.call_count == 2, "exactly one repair attempt, not an unbounded loop"


def test_json_mode_is_always_requested():
    client = ScriptedLLMClient([{"answer": "ok"}])
    generate_json(client, "prompt", system="sys", required_keys=("answer",))
    assert client.calls[0]["json_mode"] is True


@pytest.mark.parametrize(
    "value,expected",
    [(True, True), ("true", True), ("YES", True), (1, True), ("false", False), (None, False), ({}, False)],
)
def test_bool_coercion(value, expected):
    assert as_bool(value) is expected


@pytest.mark.parametrize("value,expected", [(0.5, 0.5), ("0.8", 0.8), (5, 1.0), (-2, 0.0), ("x", 0.5)])
def test_float_coercion_clamps_to_the_unit_interval(value, expected):
    assert as_float(value) == expected


def test_string_list_coercion_tolerates_a_bare_string():
    assert as_str_list("one") == ["one"]
    assert as_str_list(["a", " b ", ""]) == ["a", "b"]
    assert as_str_list(None) == []
    assert as_str_list(42) == []


def test_text_coercion():
    assert as_text("  hi  ") == "hi"
    assert as_text(None) == ""
    assert as_text(7) == "7"

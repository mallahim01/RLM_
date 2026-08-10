from __future__ import annotations

import json
import subprocess
import sys

import pytest

from app.cli import main
from tests.conftest import ERP_DOC, PROJECT_ROOT

QUESTION = "Which platform is most open to third-party agents?"


def test_mock_mode_runs_end_to_end_without_a_key(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(["--mock", "--doc", str(ERP_DOC), "--question", QUESTION])

    captured = capsys.readouterr()
    assert code == 0
    assert "Answer" in captured.out
    assert "[RLM]" in captured.err, "the trace belongs on stderr"


def test_json_output_is_valid_and_stdout_stays_pure(capsys):
    code = main(["--mock", "--json", "--doc", str(ERP_DOC), "--question", QUESTION])

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert {"question", "answer", "stats", "trace", "findings"} <= payload.keys()
    assert payload["stats"]["llm_calls"] > 0
    assert payload["stats"]["context_efficiency"] < 1.0


def test_show_tree_prints_the_chunk_hierarchy(capsys):
    code = main(["--doc", str(ERP_DOC), "--show-tree", "--mock"])

    captured = capsys.readouterr()
    assert code == 0
    assert "c4.1" in captured.out
    assert "Details by Platform" in captured.out


def test_missing_key_without_mock_is_a_clear_error(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("app.cli.load_settings", lambda **kw: _keyless_settings(**kw))

    code = main(["--doc", str(ERP_DOC), "--question", QUESTION])

    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def _keyless_settings(**overrides):
    from app.config import load_settings

    overrides["openai_api_key"] = ""
    return load_settings(**overrides)


def test_a_missing_document_fails_with_exit_code_2(capsys):
    code = main(["--mock", "--doc", "does-not-exist.md", "--question", QUESTION])
    assert code == 2
    assert "could not load document" in capsys.readouterr().err


def test_an_unsupported_format_is_reported(capsys, tmp_path):
    bad = tmp_path / "notes.xlsx"
    bad.write_text("x", encoding="utf-8")

    code = main(["--mock", "--doc", str(bad), "--question", QUESTION])

    assert code == 2
    assert "supported formats" in capsys.readouterr().err


def test_a_contradictory_budget_is_rejected_before_any_work(capsys):
    code = main(["--mock", "--max-context-tokens", "500", "--chunk-target-tokens", "5000"])
    assert code == 2
    assert "configuration error" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["run_mock_demo.py"])
def test_the_documented_example_still_runs(script):
    """Keeps examples/ from rotting: it is executed, not just described."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "examples" / script)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

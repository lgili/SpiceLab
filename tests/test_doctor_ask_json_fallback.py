import json
import os
import typing as t

import pytest


def _fake_generate_nonjson(schema: t.Any, system_prompt: str, user_prompt: str, **kwargs: t.Any):
    class _Res:
        model = kwargs.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        # Intentionally non-JSON content
        content = "Here are some tips: 1) run ngspice, 2) check PATH, 3) reinstall"
        parsed = None

    return _Res()


@pytest.mark.parametrize("fmt", ["json"])  # validate fallback in JSON mode
def test_doctor_ask_json_fallback(monkeypatch, capsys, fmt):
    import importlib

    importlib.import_module("spicelab.doctor")
    import spicelab.doctor as doctor

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "dummy-model")
    monkeypatch.setattr(
        "spicelab.ai.llm.generate_structured_openai", _fake_generate_nonjson, raising=True
    )

    rc = doctor._doctor_ask("help me fix ngspice", out_format=fmt, model="dummy-model")
    assert rc == 0

    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    # Contract: model present, and we fall back to raw when not parsed
    assert data["model"] == "dummy-model"
    assert "raw" in data
    assert "tips" in data["raw"].lower()

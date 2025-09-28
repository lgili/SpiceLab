import json
import os
import typing as t
from types import SimpleNamespace

import pytest

# We won't actually call the network. Instead, we'll monkeypatch the LLM wrapper
# to return a deterministic structured response.


class _DummyParsed(t.TypedDict, total=False):
    intent: str
    summary: str
    steps: list[str]
    commands: list[str]
    circuit_snippet: str | None
    notes: list[str]


def _fake_generate(schema: t.Any, system_prompt: str, user_prompt: str, **kwargs: t.Any):
    model = kwargs.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    data = {
        "intent": "howto",
        "summary": "How to measure GBW",
        "steps": ["Run AC analysis", "Interpolate unity gain"],
        "commands": ["spicelab-measure --list"],
        "circuit_snippet": None,
        "notes": ["Use log-frequency interpolation"],
    }
    content = json.dumps(data)
    parsed = schema.model_validate(data)  # pydantic BaseModel
    return SimpleNamespace(model=model, content=content, parsed=parsed)


@pytest.mark.parametrize("fmt", ["json"])  # we validate only JSON mode here
def test_doctor_ask_json_contract(monkeypatch, capsys, fmt):
    import importlib

    # Reload module to ensure a clean import state
    importlib.import_module("spicelab.doctor")
    import spicelab.doctor as doctor

    # Monkeypatch generator to avoid network
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "dummy-model")
    monkeypatch.setattr("spicelab.ai.llm.generate_structured_openai", _fake_generate, raising=True)

    # Run ask in JSON format
    rc = doctor._doctor_ask("how to measure gbw?", out_format=fmt, model="dummy-model")
    assert rc == 0

    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    # Contract: model present, and structured keys when parse succeeded
    assert data["model"] == "dummy-model"
    for key in ["intent", "summary", "steps", "commands", "notes"]:
        assert key in data
    # circuit_snippet may be null
    assert "circuit_snippet" in data

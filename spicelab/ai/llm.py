"""Lightweight LLM wrapper for optional AI features.

This module is intentionally small and optional. It uses pydantic for
structured outputs and can call an OpenAI-compatible endpoint if the
"openai" package is available and OPENAI_API_KEY is configured.

The goal is to provide a simple interface the rest of spicelab can use
without hard-coding a specific provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError


class LLMError(RuntimeError):
    pass


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    model: str
    content: str
    parsed: T | None


def _load_openai() -> Any:
    try:
        import os

        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client
    except ImportError as exc:  # pragma: no cover - optional
        raise LLMError(
            "The 'openai' package is not installed. Install with 'pip install openai' or"
            " add the optional extra: pip install spicelab[ai]"
        ) from exc


def generate_structured_openai(
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> GenerationResult[T]:
    """Call an OpenAI-compatible endpoint and try to parse JSON into schema.

    The model can be configured via OPENAI_MODEL environment variable; otherwise
    a small, cost-effective default is used.
    """
    import json
    import os

    client = _load_openai()
    mdl = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if mdl is None:
        mdl = "gpt-4o-mini"
    # Nudge the model to emit JSON matching the schema
    json_instruction = (
        "Respond only with a single JSON object that matches the provided schema. "
        "Do not include any explanatory text."
    )
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + json_instruction},
        {"role": "user", "content": user_prompt},
    ]
    try:
        create_kwargs: dict[str, Any] = {
            "model": mdl,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            create_kwargs["timeout"] = timeout
        resp = client.chat.completions.create(**create_kwargs)
    except Exception as exc:  # pragma: no cover - network/SDK errors
        raise LLMError(f"OpenAI request failed: {exc}") from exc

    # Defensive extraction of assistant message content
    content: str
    try:
        if getattr(resp, "choices", None) and getattr(resp.choices[0], "message", None):
            content = resp.choices[0].message.content or "{}"
        else:
            content = "{}"
    except Exception:
        content = "{}"
    parsed: T | None
    try:
        data = json.loads(content)
        parsed = schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        parsed = None
    return GenerationResult(model=mdl, content=content, parsed=parsed)


__all__ = [
    "LLMError",
    "GenerationResult",
    "generate_structured_openai",
]

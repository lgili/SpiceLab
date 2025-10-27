from __future__ import annotations

import re
from typing import Any

_ALLOWED_CHARS_RE = re.compile(r"^[A-Za-z0-9_()+\-*/.^, ]+$")
_ALLOWED_NAMES = {
    "pi",
    "e",
    "ln",
    "exp",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "abs",
    "min",
    "max",
}


class ExpressionError(ValueError):
    """Raised when a DSL expression contains unsupported tokens."""


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    magnitude = abs(value)
    if magnitude >= 1e4 or magnitude <= 1e-4:
        return f"{value:.12g}"
    return f"{value:.12g}"


def normalize_expression(value: Any, *, allow_unit_suffix: bool = True) -> str:
    """Return a sanitized SPICE expression string.

    Numbers are rendered in a compact format. Strings are validated to ensure
    they only contain safe characters (letters, digits, parentheses, math
    operators and decimal separators). Unit suffixes such as ``1k`` or ``10u``
    are allowed when ``allow_unit_suffix`` is true.
    """

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return _format_number(float(value))
    if isinstance(value, str):
        expr = value.strip()
        if not expr:
            raise ExpressionError("expression cannot be empty")
        if "\n" in expr or "\r" in expr:
            raise ExpressionError("multiline expressions are not allowed")
        if any(ch in expr for ch in {";", "#"}):
            raise ExpressionError("characters ';' and '#' are not allowed in expressions")
        if allow_unit_suffix and _looks_like_unit_literal(expr):
            return expr
        if not _ALLOWED_CHARS_RE.match(expr):
            raise ExpressionError(f"unsupported characters in expression: {expr!r}")
        tokens = [tok for tok in re.split(r"[^A-Za-z_]+", expr) if tok]
        for name in tokens:
            if name.lower() in _ALLOWED_NAMES:
                continue
            if name[0].isalpha() or name.startswith("_"):
                # treat as parameter/variable reference; keep as-is
                continue
        return expr
    raise ExpressionError(f"unsupported expression type: {type(value)!r}")


_UNIT_SUFFIXES = {
    "t",
    "g",
    "meg",
    "k",
    "",
    "m",
    "u",
    "µ",
    "n",
    "p",
    "f",
}


def _looks_like_unit_literal(expr: str) -> bool:
    match = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?([A-Za-zµ]{0,3})", expr)
    if not match:
        return False
    suffix = match.group(1) or ""
    return suffix.lower() in _UNIT_SUFFIXES


__all__ = ["normalize_expression", "ExpressionError"]

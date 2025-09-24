# mypy: ignore-errors
"""SPICE netlist preprocessing helpers.

Functions available:

* ``preprocess_netlist(text)`` – join continuation lines, keep ``.subckt`` blocks
  together and return logical lines.
* ``tokenize_card(line)`` – split a card while preserving parenthesis groups.
* ``parse_with_lark(lines)`` – optional AST parsing if `lark` is installed.

The Lark dependency is optional; preprocessing/tokenizing work standalone and are
used by :func:`spicelab.core.circuit.Circuit.from_netlist`.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _LarkProtocol(Protocol):
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def parse(self, text: str) -> object: ...
else:  # pragma: no cover - runtime fallback
    _LarkProtocol = object


def preprocess_netlist(text: str) -> list[str]:
    raw_lines = text.splitlines()
    # join continuations starting with '+'
    joined: list[str] = []
    for ln in raw_lines:
        if not ln.strip():
            continue
        if ln.lstrip().startswith("+") and joined:
            cont = ln.lstrip()[1:].lstrip()
            joined[-1] = joined[-1] + " " + cont
        else:
            joined.append(ln.rstrip("\n"))

    # capture .subckt blocks
    lines: list[str] = []
    i = 0
    while i < len(joined):
        ln = joined[i]
        lstrip = ln.lstrip()
        if lstrip.lower().startswith(".subckt"):
            block = [ln]
            i += 1
            while i < len(joined):
                block.append(joined[i])
                if joined[i].lstrip().lower().startswith(".ends"):
                    break
                i += 1
            lines.append("\n".join(block))
            i += 1
            continue
        lines.append(ln.strip())
        i += 1
    return lines


def tokenize_card(s: str) -> list[str]:
    toks: list[str] = []
    cur: list[str] = []
    in_paren = 0
    for ch in s:
        if ch == "(":
            in_paren += 1
            cur.append(ch)
            continue
        if ch == ")":
            in_paren = max(0, in_paren - 1)
            cur.append(ch)
            continue
        if in_paren > 0:
            cur.append(ch)
            continue
        if ch.isspace():
            if cur:
                toks.append("".join(cur))
                cur = []
            continue
        cur.append(ch)
    if cur:
        toks.append("".join(cur))
    return toks


def parse_with_lark(lines: list[str]) -> Any:
    try:
        lark_module = import_module("lark")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "lark-parser not installed; install dev-requirements to enable full parsing"
        ) from None

    lark_cls = cast("type[_LarkProtocol]", lark_module.Lark)

    grammar = r"""
        start: line+
        line: /[^\n]+/ "\n"?
        %ignore /\s+/
    """

    parser = lark_cls(grammar, start="start")
    joined = "\n".join(lines) + "\n"
    tree = parser.parse(joined)
    return tree


def parse_lines_to_ast(lines: list[str]) -> list[dict[str, object]]:
    """Return a simple AST (list of dict nodes) from logical lines.

    Node types: comment, directive, subckt (with block raw), component
    """
    ast: list[dict[str, object]] = []
    for ln in lines:
        if not ln:
            continue
        lstrip = ln.lstrip()
        if lstrip.startswith("*"):
            ast.append({"type": "comment", "raw": ln})
            continue
        if lstrip.startswith("."):
            # directive (including .include, .model, .param, .end, .endc etc.)
            ast.append({"type": "directive", "raw": ln, "name": lstrip.split()[0].lstrip(".")})
            continue
        if lstrip.lower().startswith(".subckt"):
            # subckt blocks should have been joined previously; ln may contain newlines
            ast.append({"type": "subckt", "raw": ln})
            continue
        # component card or unknown
        toks = tokenize_card(ln)
        if not toks:
            continue
        card = toks[0]
        letter = card[0].upper()
        ref = card[1:]
        ast.append(
            {
                "type": "component",
                "raw": ln,
                "tokens": toks,
                "card": card,
                "letter": letter,
                "ref": ref,
            }
        )
    return ast

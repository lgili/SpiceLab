"""SPICE syntax compatibility layer between different simulators.

This module provides functions to convert SPICE expressions between different
simulator syntaxes, primarily LTspice to ngspice.

Key conversions:
- IF(cond, true, false) -> (cond) ? (true) : (false)  [ngspice ternary]
- Nested IF() functions are handled recursively

Reference:
- LTspice: Uses IF(cond, true_val, false_val) function
- ngspice: Uses C-style ternary operator (cond) ? (true_val) : (false_val)
- ngspice also supports ternary_fcn(cond, true_val, false_val)
"""

from __future__ import annotations

import re
from typing import Literal

__all__ = [
    "convert_if_to_ternary",
    "convert_expression_for_engine",
    "convert_param_directive",
    "convert_directives_for_engine",
]


def _find_matching_paren(s: str, start: int) -> int:
    """Find the matching closing parenthesis for an opening one.

    Args:
        s: The string to search
        start: Index of the opening parenthesis

    Returns:
        Index of the matching closing parenthesis, or -1 if not found
    """
    if start >= len(s) or s[start] != "(":
        return -1

    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
        i += 1

    return i - 1 if depth == 0 else -1


def _split_if_args(args_str: str) -> list[str]:
    """Split IF() function arguments respecting nested parentheses.

    Args:
        args_str: The argument string (without outer parentheses)

    Returns:
        List of argument strings
    """
    args: list[str] = []
    current: list[str] = []
    depth = 0

    for char in args_str:
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            current.append(char)

    # Add the last argument
    if current:
        args.append("".join(current).strip())

    return args


def convert_if_to_ternary(expr: str) -> str:
    """Convert LTspice IF(cond, true, false) to ngspice ternary operator.

    LTspice syntax: IF(condition, true_value, false_value)
    ngspice syntax: (condition) ? (true_value) : (false_value)

    This function handles:
    - Case-insensitive IF matching
    - Nested IF() functions
    - Complex expressions with multiple IF() calls

    Args:
        expr: Expression string potentially containing IF() functions

    Returns:
        Expression with IF() converted to ternary operators

    Examples:
        >>> convert_if_to_ternary("IF(T1<0,1,0)")
        '((T1<0) ? (1) : (0))'

        >>> convert_if_to_ternary("R0*(1+A*T1+IF(T1<0,C*T1,0))")
        'R0*(1+A*T1+((T1<0) ? (C*T1) : (0)))'

        >>> convert_if_to_ternary("IF(x>0, IF(x>10, 2, 1), 0)")
        '((x>0) ? (((x>10) ? (2) : (1))) : (0))'
    """
    result = expr

    # Pattern to find IF( - case insensitive
    if_pattern = re.compile(r"\bIF\s*\(", re.IGNORECASE)

    # Process from the innermost IF to outermost
    max_iterations = 100  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        match = if_pattern.search(result)
        if not match:
            break

        # Find the position of the opening parenthesis
        if_start = match.start()
        paren_start = match.end() - 1  # Position of '('

        # Find the matching closing parenthesis
        paren_end = _find_matching_paren(result, paren_start)
        if paren_end == -1:
            # Malformed IF - skip
            break

        # Extract the arguments
        args_str = result[paren_start + 1 : paren_end]
        args = _split_if_args(args_str)

        if len(args) != 3:
            # IF() requires exactly 3 arguments
            break

        cond, true_val, false_val = args

        # Recursively convert any nested IF() in the arguments
        cond = convert_if_to_ternary(cond)
        true_val = convert_if_to_ternary(true_val)
        false_val = convert_if_to_ternary(false_val)

        # Build the ternary expression
        ternary = f"(({cond}) ? ({true_val}) : ({false_val}))"

        # Replace the IF(...) with the ternary
        result = result[:if_start] + ternary + result[paren_end + 1 :]

        iteration += 1

    return result


def convert_expression_for_engine(
    expr: str,
    target_engine: Literal["ngspice", "ltspice", "xyce"],
) -> str:
    """Convert a SPICE expression for a specific simulation engine.

    Args:
        expr: The expression string
        target_engine: Target simulator ("ngspice", "ltspice", "xyce")

    Returns:
        Converted expression string
    """
    if target_engine == "ngspice":
        # Convert IF() to ternary for ngspice
        return convert_if_to_ternary(expr)
    elif target_engine == "xyce":
        # Xyce also uses IF() but with slightly different syntax
        # For now, keep as-is (Xyce supports IF)
        return expr
    else:
        # LTspice - no conversion needed
        return expr


def _remove_spaces_in_expression(expr: str) -> str:
    """Remove spaces around operators in SPICE expressions.

    ngspice doesn't allow spaces around operators in .param expressions.
    This function removes spaces while preserving them in strings/comments.

    Args:
        expr: Expression string

    Returns:
        Expression with spaces removed around operators
    """
    # Remove spaces around operators: + - * / ** = < > <= >= == != ( )
    # Be careful not to remove spaces in strings
    result = expr

    # Remove spaces around operators
    # Order matters - do ** before * to avoid issues
    operators = ["**", "*", "/", "+", "-", "<=", ">=", "==", "!=", "<", ">", "="]
    for op in operators:
        # Remove space before operator
        result = re.sub(rf"\s+({re.escape(op)})", r"\1", result)
        # Remove space after operator
        result = re.sub(rf"({re.escape(op)})\s+", r"\1", result)

    # Also remove spaces around parentheses
    result = re.sub(r"\s+\(", "(", result)
    result = re.sub(r"\(\s+", "(", result)
    result = re.sub(r"\s+\)", ")", result)
    result = re.sub(r"\)\s+", ")", result)

    # Remove spaces around ? and : for ternary operator
    result = re.sub(r"\s+\?", "?", result)
    result = re.sub(r"\?\s+", "?", result)
    result = re.sub(r"\s+:", ":", result)
    result = re.sub(r":\s+", ":", result)

    return result


def convert_param_directive(
    directive: str,
    target_engine: Literal["ngspice", "ltspice", "xyce"],
) -> str:
    """Convert a .param directive for a specific simulation engine.

    Args:
        directive: The directive string (e.g., ".param X=IF(a,b,c)")
        target_engine: Target simulator

    Returns:
        Converted directive string
    """
    # Check if this is a .param directive
    if not directive.lower().startswith(".param"):
        return directive

    # Extract the parameter definition part
    param_match = re.match(r"(\.param\s+)(.+)", directive, re.IGNORECASE)
    if not param_match:
        return directive

    prefix = param_match.group(1)
    param_def = param_match.group(2)

    # Convert the parameter definition
    converted_def = convert_expression_for_engine(param_def, target_engine)

    # For ngspice, remove spaces in the expression
    if target_engine == "ngspice":
        converted_def = _remove_spaces_in_expression(converted_def)

    return prefix + converted_def


def convert_directives_for_engine(
    directives: list[str],
    target_engine: Literal["ngspice", "ltspice", "xyce"],
) -> list[str]:
    """Convert a list of SPICE directives for a specific simulation engine.

    Args:
        directives: List of directive strings
        target_engine: Target simulator

    Returns:
        List of converted directive strings
    """
    converted = []
    for directive in directives:
        if directive.lower().startswith(".param"):
            converted.append(convert_param_directive(directive, target_engine))
        else:
            converted.append(directive)
    return converted

"""Safe expression evaluator for parameter dependencies.

Phase 2 (P2.5): Expression evaluation with DAG validation to prevent
circular dependencies.

Example:
    >>> from spicelab.core.expressions import safe_eval_expression
    >>>
    >>> context = {"R1": 1000, "R2": 2000}
    >>> safe_eval_expression("R1 + R2", context)
    3000.0
    >>> safe_eval_expression("2 * R1 + R2", context)
    4000.0
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["safe_eval_expression", "validate_expression_dependencies", "ExpressionError"]


class ExpressionError(ValueError):
    """Error in parameter expression evaluation."""

    pass


# Safe math functions allowed in expressions
_SAFE_FUNCTIONS = {
    # Basic math
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    # Math module
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "pow": math.pow,
    # Trigonometric
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # Special
    "ceil": math.ceil,
    "floor": math.floor,
    # Constants
    "pi": math.pi,
    "e": math.e,
}


def safe_eval_expression(expression: str, context: Mapping[str, float]) -> float:
    """Safely evaluate parameter expression with context.

    Supports:
    - Arithmetic: +, -, *, /, **, %
    - Comparison: <, >, <=, >=, ==, !=
    - Boolean: and, or, not
    - Math functions: sqrt, exp, log, sin, cos, etc.
    - Constants: pi, e
    - Parameter references from context

    Security:
    - No eval() or exec()
    - Whitelist of safe functions
    - AST-based parsing (no code injection)

    Args:
        expression: Expression string (e.g., "2*R1 + R2")
        context: Dict mapping parameter names to values

    Returns:
        Evaluated result (float)

    Raises:
        ExpressionError: If expression is invalid or references undefined parameter

    Example:
        >>> context = {"R1": 1000, "R2": 2000}
        >>> safe_eval_expression("sqrt(R1**2 + R2**2)", context)
        2236.067977...
    """
    try:
        # Parse expression to AST
        tree = ast.parse(expression, mode="eval")

        # Evaluate AST safely
        return _eval_node(tree.body, context)

    except KeyError as exc:
        raise ExpressionError(f"Undefined parameter in expression '{expression}': {exc}") from exc
    except Exception as exc:
        raise ExpressionError(f"Invalid expression '{expression}': {exc}") from exc


def _eval_node(node: ast.expr, context: Mapping[str, float]) -> float:
    """Recursively evaluate AST node.

    This is a safe evaluator that only supports whitelisted operations.
    """
    # Literals
    if isinstance(node, ast.Constant):
        return float(node.value)

    if isinstance(node, ast.Num):  # Python 3.7 compat
        return float(node.n)

    # Variable lookup (parameter reference)
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        if node.id in context:
            return float(context[node.id])
        raise KeyError(node.id)

    # Binary operations
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right

        raise ExpressionError(f"Unsupported binary operator: {node.op.__class__.__name__}")

    # Unary operations
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context)

        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return float(not operand)

        raise ExpressionError(f"Unsupported unary operator: {node.op.__class__.__name__}")

    # Comparison operations
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)

        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _eval_node(comparator, context)

            if isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            elif isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            else:
                raise ExpressionError(f"Unsupported comparison: {op.__class__.__name__}")

            if not result:
                return 0.0
            left = right

        return 1.0

    # Boolean operations
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = 1.0
            for value in node.values:
                result = result and _eval_node(value, context)
                if not result:
                    return 0.0
            return result

        if isinstance(node.op, ast.Or):
            result = 0.0
            for value in node.values:
                result = result or _eval_node(value, context)
                if result:
                    return result
            return result

        raise ExpressionError(f"Unsupported boolean operator: {node.op.__class__.__name__}")

    # Function calls
    if isinstance(node, ast.Call):
        func_node = node.func

        if not isinstance(func_node, ast.Name):
            raise ExpressionError("Only simple function calls supported")

        func_name = func_node.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ExpressionError(f"Function '{func_name}' not allowed")

        func = _SAFE_FUNCTIONS[func_name]
        args = [_eval_node(arg, context) for arg in node.args]

        try:
            return float(func(*args))
        except Exception as exc:
            raise ExpressionError(f"Error calling {func_name}: {exc}") from exc

    # Conditional expression (ternary)
    if isinstance(node, ast.IfExp):
        test = _eval_node(node.test, context)
        if test:
            return _eval_node(node.body, context)
        return _eval_node(node.orelse, context)

    raise ExpressionError(f"Unsupported expression node: {node.__class__.__name__}")


def validate_expression_dependencies(
    expressions: Mapping[str, str], detect_cycles: bool = True
) -> list[str]:
    """Validate parameter expression dependencies and detect cycles.

    Args:
        expressions: Dict mapping parameter names to expression strings
        detect_cycles: If True, raise error on circular dependencies

    Returns:
        List of parameter names in dependency order (topological sort)

    Raises:
        ExpressionError: If circular dependency detected

    Example:
        >>> expressions = {
        ...     "R_total": "R1 + R2",
        ...     "tau": "R_total * C1",
        ...     "R1": "1000",
        ...     "R2": "2000",
        ...     "C1": "1e-6"
        ... }
        >>> validate_expression_dependencies(expressions)
        ['R1', 'R2', 'C1', 'R_total', 'tau']
    """
    # Build dependency graph
    dependencies: dict[str, set[str]] = {}

    for param_name, expr in expressions.items():
        deps = _extract_dependencies(expr)
        dependencies[param_name] = deps

    # Topological sort (Kahn's algorithm)
    if not detect_cycles:
        return list(expressions.keys())

    # Compute in-degrees
    in_degree = {name: 0 for name in expressions}
    for name, deps in dependencies.items():
        for dep in deps:
            if dep in expressions:  # Only count dependencies within this set
                in_degree[name] += 1

    # Start with nodes that have no dependencies
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        # Reduce in-degree of dependent nodes
        for name, deps in dependencies.items():
            if current in deps and name in expressions:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)

    # If not all nodes processed, there's a cycle
    if len(result) != len(expressions):
        unprocessed = set(expressions.keys()) - set(result)
        raise ExpressionError(
            f"Circular dependency detected in parameters: {', '.join(unprocessed)}"
        )

    return result


def _extract_dependencies(expression: str) -> set[str]:
    """Extract parameter names referenced in expression.

    Returns set of variable names (excludes functions and constants).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set()

    dependencies = set()

    def visit_node(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id not in _SAFE_FUNCTIONS:
                dependencies.add(node.id)

        for child in ast.iter_child_nodes(node):
            visit_node(child)

    visit_node(tree)
    return dependencies

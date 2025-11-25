# Contributing to SpiceLab

Thank you for your interest in contributing to SpiceLab! This document provides guidelines and information for contributors.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Format](#commit-message-format)
- [Review Process](#review-process)
- [Questions](#questions)

## Ways to Contribute

There are many ways to contribute to SpiceLab:

1. **Report Bugs** - Found a bug? [Open an issue](https://github.com/your-org/spicelab/issues/new?template=bug_report.md)
2. **Request Features** - Have an idea? [Start a discussion](https://github.com/your-org/spicelab/discussions)
3. **Write Code** - Fix bugs or implement features via Pull Requests
4. **Improve Documentation** - Fix typos, add examples, improve clarity
5. **Answer Questions** - Help others in GitHub Discussions or Discord
6. **Share Examples** - Contribute circuit examples and tutorials

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A SPICE simulator (NGSpice, LTspice, or Xyce)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then:
   git clone https://github.com/YOUR-USERNAME/spicelab.git
   cd spicelab
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync --all-extras --dev

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Verify installation**
   ```bash
   # Run tests
   pytest

   # Run linting
   ruff check .
   ruff format --check .

   # Run type checking
   mypy spicelab
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

## Code Style

We use automated tools to maintain consistent code style:

### Formatting

```bash
# Format code
ruff format .

# Check formatting without changes
ruff format --check .
```

### Linting

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking

```bash
# Run mypy
mypy spicelab

# Strict mode (for new code)
mypy --strict spicelab/your_module.py
```

### Style Guidelines

- **Line length:** 100 characters maximum
- **Imports:** Use `ruff` to sort imports automatically
- **Docstrings:** Use Google-style docstrings
- **Type hints:** Required for all public APIs
- **Comments:** Explain "why", not "what"

### Docstring Example

```python
def simulate_circuit(
    circuit: Circuit,
    analysis: AnalysisType,
    *,
    timeout: float = 60.0,
) -> SimulationResult:
    """Run a SPICE simulation on the given circuit.

    Args:
        circuit: The circuit to simulate.
        analysis: Type of analysis to perform (DC, AC, TRAN, etc.).
        timeout: Maximum time in seconds to wait for simulation.

    Returns:
        SimulationResult containing output data and metadata.

    Raises:
        SimulationError: If the simulation fails to converge.
        TimeoutError: If simulation exceeds the timeout.

    Example:
        >>> circuit = Circuit("example")
        >>> result = simulate_circuit(circuit, AnalysisType.DC)
        >>> print(result.data["V(out)"])
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spicelab --cov-report=html

# Run specific test file
pytest tests/test_circuit.py

# Run tests matching a pattern
pytest -k "test_validation"

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested
- Aim for >90% code coverage for new code

```python
import pytest
from spicelab.core.circuit import Circuit

class TestCircuitValidation:
    """Tests for circuit validation functionality."""

    def test_empty_circuit_is_invalid(self):
        """An empty circuit should fail validation."""
        circuit = Circuit("empty")
        result = circuit.validate()
        assert not result.is_valid

    def test_circuit_with_floating_node_warns(self):
        """A circuit with floating nodes should produce warnings."""
        circuit = create_circuit_with_floating_node()
        result = circuit.validate()
        assert any("floating" in w.message.lower() for w in result.warnings)
```

### Test Categories

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests
- `tests/property/` - Property-based tests (hypothesis)
- `tests/regression/` - Regression tests for known issues
- `tests/stress/` - Performance and stress tests

## Pull Request Process

### Before Submitting

1. **Create an issue first** for large changes to discuss the approach
2. **Keep PRs focused** - one feature or fix per PR
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Run all checks locally**:
   ```bash
   ruff format .
   ruff check .
   mypy spicelab
   pytest
   ```

### Submitting a PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**
   - Use a clear, descriptive title
   - Fill out the PR template
   - Link related issues

3. **Wait for review**
   - CI must pass (tests, linting, type checking)
   - At least one maintainer approval required
   - Address review feedback

### PR Template

When you open a PR, you'll see a template. Please fill it out completely:

- **Description**: What does this PR do?
- **Related Issues**: Link to related issues
- **Type of Change**: Bug fix, feature, docs, etc.
- **Testing**: How was this tested?
- **Checklist**: Confirm all requirements are met

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Feature
feat(templates): add LDO regulator template

# Bug fix
fix(monte-carlo): resolve convergence issue with high iteration count

Fixes #123

# Documentation
docs(tutorial): add chapter 5 on Monte Carlo analysis

# Refactor
refactor(netlist): optimize string building performance
```

### Scope (optional)

Use the module or area being changed:
- `core`, `circuit`, `components`
- `templates`, `library`
- `monte-carlo`, `validation`
- `docs`, `examples`, `tests`

## Review Process

### Timeline

- **Initial response:** Within 3 business days
- **Full review:** Within 1 week for small PRs
- **Large PRs:** May take longer, consider breaking into smaller PRs

### What Reviewers Look For

1. **Correctness** - Does the code work as intended?
2. **Tests** - Are there adequate tests?
3. **Style** - Does it follow our guidelines?
4. **Documentation** - Are changes documented?
5. **Performance** - Any performance implications?
6. **Security** - Any security concerns?

### Responding to Reviews

- Be respectful and constructive
- Address all comments
- Mark conversations as resolved when addressed
- Ask for clarification if needed

## Questions?

- **GitHub Discussions:** For general questions and ideas
- **GitHub Issues:** For bugs and feature requests
- **Discord:** For real-time chat (link in README)

## Thank You!

Every contribution matters, whether it's fixing a typo or implementing a major feature. Thank you for helping make SpiceLab better!

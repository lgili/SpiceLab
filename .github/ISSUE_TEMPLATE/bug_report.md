---
name: Bug Report
about: Report a bug to help us improve SpiceLab
title: '[BUG] '
labels: bug, triage
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Create circuit with '...'
2. Run simulation '...'
3. See error

## Minimal Reproducible Example

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc

# Minimal code that reproduces the issue
circuit = Circuit("bug_example")
# ...
```

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

What actually happened. Include any error messages or tracebacks:

```
Paste error message here
```

## Environment

- **SpiceLab version:** (e.g., 0.1.0)
- **Python version:** (e.g., 3.11.5)
- **OS:** (e.g., Windows 11, macOS 14, Ubuntu 22.04)
- **SPICE simulator:** (e.g., NGSpice 42, LTspice XVII)

## Additional Context

Add any other context about the problem here (screenshots, circuit files, etc.).

## Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided a minimal reproducible example
- [ ] I have included the full error traceback (if applicable)

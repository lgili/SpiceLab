"""Helper for GitHub Actions: update pyproject.toml 'version' field.

Reads VERSION from the environment and replaces the first
PEP 621 'version = "x.y.z"' occurrence in pyproject.toml.
This script is intentionally small, synchronous, and safe.
"""

import os
import pathlib
import re
import sys

VERSION = os.environ.get("VERSION")
if not VERSION:
    print("Environment variable VERSION is not set", file=sys.stderr)
    sys.exit(2)

pp = pathlib.Path("pyproject.toml")
if not pp.exists():
    print("pyproject.toml not found", file=sys.stderr)
    sys.exit(3)

text = pp.read_text(encoding="utf-8")
pattern = re.compile(r'(?m)^(\s*version\s*=\s*)"[^\"]+"')
new, n = pattern.subn(rf'\1"{VERSION}"', text, count=1)
if n == 0:
    print("Warning: no 'version' field updated in pyproject.toml", file=sys.stderr)
else:
    pp.write_text(new, encoding="utf-8")
    print("pyproject.toml updated to", VERSION)

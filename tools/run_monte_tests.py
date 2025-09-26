"""Helper to run the small monte-carlo related tests locally.

This script is intentionally tiny: it runs pytest for the new tests only so
contributors can validate the example/test changes quickly.

Usage:
    PYTHONPATH=. python tools/run_monte_tests.py

"""

import subprocess
import sys

TESTS = [
    "tests/test_examples_monte_carlo_metric.py",
    "tests/test_orchestrator_monte_carlo_demo.py",
    "tests/test_examples_monte_carlo_pandas.py",
]

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "pytest", "-q"] + TESTS
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)

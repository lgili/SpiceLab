"""Pytest configuration for property-based tests.

This module provides shared fixtures and Hypothesis strategies
for property-based testing.
"""

from hypothesis import HealthCheck, settings

# Configure Hypothesis to suppress some health checks that are too strict
settings.register_profile(
    "default",
    max_examples=100,
    deadline=200,  # ms
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=1000,  # ms
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=100,  # ms
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

# Activate default profile
settings.load_profile("default")

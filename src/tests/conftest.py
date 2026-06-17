"""Shared pytest configuration and fixtures for the ScribbleDB test suite.

Layout:
  src/tests/unit/         fully offline, deterministic, fast (no LLM, no network)
  src/tests/integration/  LIVE - call the real OpenAI API (marked `integration`)
  src/tests/fixtures/      reusable sample-data builders (importable, not tests)

Integration tests are auto-skipped unless OPENAI_API_KEY is set, so a plain
`pytest` run stays offline and green. Run live tests explicitly with:
    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from src.tests.fixtures import sample_data


# --------------------------------------------------------------------------- #
# Skip live integration tests when no API key is available
# --------------------------------------------------------------------------- #


def pytest_collection_modifyitems(config, items):
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_live = pytest.mark.skip(
        reason="OPENAI_API_KEY not set - skipping live integration test"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_live)


# --------------------------------------------------------------------------- #
# Convert transient network errors in integration tests to skips, not failures.
# A connection error mid-run means the test couldn't execute, not that code is
# broken -- treating it as a failure blocks CI/hooks on infrastructure flakes.
# --------------------------------------------------------------------------- #


def pytest_runtest_makereport(item, call):
    if call.when != "call":
        return
    if "integration" not in item.keywords:
        return
    if call.excinfo is None:
        return
    exc = call.excinfo.value
    _TRANSIENT_TYPES = (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ConnectionError",
        "TimeoutError",
    )
    if type(exc).__name__ in _TRANSIENT_TYPES or (
        hasattr(exc, "__cause__")
        and type(getattr(exc, "__cause__", None)).__name__ in _TRANSIENT_TYPES
    ):
        pytest.skip(f"Skipped: transient network error ({type(exc).__name__})")
        call.excinfo = None


# --------------------------------------------------------------------------- #
# Shared sample-data fixtures (thin wrappers over fixtures/sample_data.py)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fintech_nl() -> str:
    return sample_data.FINTECH_NL


@pytest.fixture
def fintech_facts():
    return sample_data.fintech_facts()


@pytest.fixture
def fintech_schema():
    return sample_data.fintech_schema()


@pytest.fixture
def fintech_registry():
    return sample_data.fintech_registry()


@pytest.fixture
def simple_schema():
    return sample_data.simple_two_table_schema()


@pytest.fixture
def simple_facts():
    return sample_data.simple_facts()

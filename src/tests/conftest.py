"""Shared pytest configuration and fixtures for the ScribbleDB test suite.

Layout:
  src/tests/unit/         fully offline, deterministic, fast (no LLM, no network)
  src/tests/integration/  LIVE - call the real OpenAI API (marked `integration`)
  src/tests/fixtures/      reusable sample-data builders (importable, not tests)

Integration tests require an EXPLICIT opt-in (`pytest --live`), so a plain
`pytest` run always stays offline and green:
    pytest --live -m integration
"""

from __future__ import annotations

import os

import pytest

from src.tests.fixtures import sample_data


# --------------------------------------------------------------------------- #
# Live integration tests are opt-IN, never opt-out
# --------------------------------------------------------------------------- #
#
# This used to gate purely on `os.environ.get("OPENAI_API_KEY")`: if a key was
# present, the live tests ran. That is the wrong default and it silently
# misfired -- on a developer machine that exports OPENAI_API_KEY (the normal
# case here), a bare `pytest` ran 15 tests against the real provider API and
# the real DuckDuckGo endpoint, spending money and hanging on the network,
# while the docstring above promised "a plain pytest run stays offline".
#
# Having a key is not consent to spend it. The gate is now an explicit `--live`
# flag. A marker expression is deliberately NOT used as the opt-in signal --
# `-m "not integration"` contains the substring "integration", so keying off
# `-m` would mean the very expression meant to EXCLUDE these tests could be
# read as selecting them.

_PROVIDER_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "DEEPSEEK_API_KEY",
    "VLLM_BASE_URL",
)


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Run tests marked `integration` against real provider APIs. These "
            "cost money and hit the network; without this flag they are skipped."
        ),
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip = pytest.mark.skip(
            reason="live test - pass --live to opt in (real API calls, costs money)"
        )
    elif not any(os.environ.get(v) for v in _PROVIDER_KEY_ENV_VARS):
        skip = pytest.mark.skip(
            reason=(
                "--live given but no provider key set (one of: "
                + ", ".join(_PROVIDER_KEY_ENV_VARS)
                + ")"
            )
        )
    else:
        return
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- #
# Convert transient network errors in integration tests to skips, not failures.
# A connection error mid-run means the test couldn't execute, not that code is
# broken -- treating it as a failure blocks CI/hooks on infrastructure flakes.
# --------------------------------------------------------------------------- #


_TRANSIENT_TYPES = (
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "ConnectionError",
    "TimeoutError",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Must be a hookwrapper: `yield` runs pytest's own makereport first to
    # build the real TestReport, then we mutate its outcome IN PLACE below.
    # Calling `pytest.skip()` directly here (the previous approach) raises
    # `Skipped` from inside a report hook, which nothing downstream catches --
    # it crashes the whole run (and the whole xdist session) instead of just
    # marking this one test skipped.
    outcome = yield
    if call.when != "call":
        return
    if "integration" not in item.keywords:
        return
    if call.excinfo is None:
        return
    exc = call.excinfo.value
    if type(exc).__name__ in _TRANSIENT_TYPES or (
        hasattr(exc, "__cause__")
        and type(getattr(exc, "__cause__", None)).__name__ in _TRANSIENT_TYPES
    ):
        report = outcome.get_result()
        report.outcome = "skipped"
        # Pytest's terminal reporter (short_test_summary -> _folded_skips)
        # asserts a skipped report's longrepr is a (path, lineno, reason)
        # tuple, not a bare string -- matching the shape pytest's own
        # skipping.py plugin uses internally for a real skip.
        path, lineno, _ = item.location
        report.longrepr = (
            path,
            lineno,
            f"Skipped: transient network error ({type(exc).__name__})",
        )


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

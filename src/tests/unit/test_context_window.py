"""Tests for src/util/core/context_window.py's get_context_window().

Mocks requests.get entirely -- no real network calls. Covers: direct
resolution for the 3 providers that expose context length themselves
(OpenRouter, Gemini, Groq), the OpenRouter-catalog fallback for the 3
that don't (DeepSeek/OpenAI/Cerebras -- exercised generically via a fake
provider name to avoid depending on any specific provider string), the
"no silent guessing" behavior (unresolvable -> raises, ambiguous ->
raises), and the in-memory cache actually avoiding a second HTTP call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.util.core import context_window as cw
from src.util.core.context_window import (
    ContextWindowUnresolvedError,
    get_context_window,
)


@pytest.fixture(autouse=True)
def _reset_module_caches():
    cw._openrouter_catalog_cache = None
    cw._resolved_cache = {}
    yield
    cw._openrouter_catalog_cache = None
    cw._resolved_cache = {}


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestOpenRouterDirect:
    def test_exact_catalog_match_resolves(self):
        catalog_resp = _mock_response(
            {"data": [{"id": "openai/gpt-4o", "context_length": 128000}]}
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp) as mock_get:
            window = get_context_window("openrouter", "openai/gpt-4o")
        assert window == 128000
        assert mock_get.call_count == 1

    def test_no_match_raises(self):
        catalog_resp = _mock_response(
            {"data": [{"id": "other/model", "context_length": 1}]}
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp):
            with pytest.raises(ContextWindowUnresolvedError):
                get_context_window("openrouter", "openai/gpt-4o")


class TestGeminiDirect:
    def test_input_token_limit_used(self):
        resp = _mock_response({"inputTokenLimit": 1048576})
        with patch.object(cw.requests, "get", return_value=resp):
            window = get_context_window("gemini", "gemini-2.5-flash", api_key="fake")
        assert window == 1048576

    def test_missing_api_key_raises(self):
        with pytest.raises(ContextWindowUnresolvedError):
            get_context_window("gemini", "gemini-2.5-flash")

    def test_non_200_raises_unresolved(self):
        resp = _mock_response({}, status_code=404)
        with patch.object(cw.requests, "get", return_value=resp):
            with pytest.raises(ContextWindowUnresolvedError):
                get_context_window("gemini", "nonexistent-model", api_key="fake")


class TestGroqDirect:
    def test_context_window_field_used(self):
        resp = _mock_response(
            {"data": [{"id": "llama-3.3-70b-versatile", "context_window": 131072}]}
        )
        with patch.object(cw.requests, "get", return_value=resp):
            window = get_context_window(
                "groq", "llama-3.3-70b-versatile", api_key="fake"
            )
        assert window == 131072

    def test_missing_api_key_raises(self):
        with pytest.raises(ContextWindowUnresolvedError):
            get_context_window("groq", "llama-3.3-70b-versatile")


class TestOpenRouterCatalogFallback:
    """Covers the DeepSeek/OpenAI/Cerebras path generically -- any
    provider not in the direct-lookup set falls back to the catalog."""

    def test_exact_match_via_vendor_prefixed_id(self):
        catalog_resp = _mock_response(
            {
                "data": [
                    {"id": "deepseek/deepseek-v4-flash", "context_length": 1048576},
                    {"id": "unrelated/model-x", "context_length": 4096},
                ]
            }
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp):
            window = get_context_window("deepseek", "deepseek-v4-flash")
        assert window == 1048576

    def test_free_suffix_stripped_before_matching(self):
        catalog_resp = _mock_response(
            {"data": [{"id": "openai/gpt-oss-20b:free", "context_length": 131072}]}
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp):
            window = get_context_window("cerebras", "gpt-oss-20b")
        assert window == 131072

    def test_ambiguous_matches_raise_not_guess(self):
        catalog_resp = _mock_response(
            {
                "data": [
                    {"id": "vendor-a/shared-name", "context_length": 8192},
                    {"id": "vendor-b/shared-name", "context_length": 16384},
                ]
            }
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp):
            with pytest.raises(ContextWindowUnresolvedError):
                get_context_window("openai", "shared-name")

    def test_no_match_raises(self):
        catalog_resp = _mock_response(
            {"data": [{"id": "vendor/other", "context_length": 1}]}
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp):
            with pytest.raises(ContextWindowUnresolvedError):
                get_context_window("openai", "totally-unknown-model")


class TestCaching:
    def test_second_call_does_not_hit_network_again(self):
        catalog_resp = _mock_response(
            {"data": [{"id": "openai/gpt-4o", "context_length": 128000}]}
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp) as mock_get:
            first = get_context_window("openrouter", "openai/gpt-4o")
            second = get_context_window("openrouter", "openai/gpt-4o")
        assert first == second == 128000
        assert mock_get.call_count == 1

    def test_different_model_still_triggers_lookup(self):
        catalog_resp = _mock_response(
            {
                "data": [
                    {"id": "openai/gpt-4o", "context_length": 128000},
                    {"id": "openai/gpt-4o-mini", "context_length": 128000},
                ]
            }
        )
        with patch.object(cw.requests, "get", return_value=catalog_resp) as mock_get:
            get_context_window("openrouter", "openai/gpt-4o")
            get_context_window("openrouter", "openai/gpt-4o-mini")
        # Catalog itself is cached (1 HTTP call) even across 2 distinct
        # resolved models.
        assert mock_get.call_count == 1

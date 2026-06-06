"""Unit tests for src.util.invoke._extract_token_usage.

Deterministic, offline. We only test the pure token-summing helper; we never
call get_response (which invokes a live agent).

The helper sums total_tokens from a heterogeneous list of messages:
  - langchain BaseMessage objects with .usage_metadata (dict) -> "total_tokens"
  - BaseMessage objects falling back to .response_metadata["token_usage"]["total_tokens"]
  - plain dicts with "usage_metadata" -> "total_tokens"
  - plain dicts falling back to ["response_metadata"]["token_usage"]["total_tokens"]
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.util.invoke import _extract_token_usage


def test_empty_list_returns_zero():
    assert _extract_token_usage([]) == 0


def test_basemessage_with_usage_metadata():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    assert _extract_token_usage([msg]) == 15


def test_basemessage_falls_back_to_response_metadata():
    # No usage_metadata -> read response_metadata.token_usage.total_tokens
    msg = AIMessage(
        content="hi",
        response_metadata={"token_usage": {"total_tokens": 42}},
    )
    assert _extract_token_usage([msg]) == 42


def test_basemessage_without_any_usage_contributes_zero():
    msg = HumanMessage(content="no usage here")
    assert _extract_token_usage([msg]) == 0


def test_plain_dict_with_usage_metadata():
    msg = {"usage_metadata": {"total_tokens": 7}}
    assert _extract_token_usage([msg]) == 7


def test_plain_dict_falls_back_to_response_metadata():
    msg = {"response_metadata": {"token_usage": {"total_tokens": 33}}}
    assert _extract_token_usage([msg]) == 33


def test_plain_dict_with_no_usage_contributes_zero():
    assert _extract_token_usage([{}]) == 0


def test_sums_across_mixed_messages():
    messages = [
        AIMessage(content="a", usage_metadata={"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}),
        AIMessage(content="b", response_metadata={"token_usage": {"total_tokens": 20}}),
        {"usage_metadata": {"total_tokens": 30}},
        {"response_metadata": {"token_usage": {"total_tokens": 40}}},
        HumanMessage(content="no usage"),
        {},
    ]
    assert _extract_token_usage(messages) == 100


def test_usage_metadata_missing_total_tokens_key_treated_as_zero():
    # usage_metadata present but lacks total_tokens -> .get default 0
    msg = {"usage_metadata": {"input_tokens": 5}}
    assert _extract_token_usage([msg]) == 0


def test_empty_usage_metadata_falls_through_to_response_metadata():
    # An empty/falsy usage_metadata dict is falsy, so the helper uses the
    # response_metadata fallback branch for plain dicts.
    msg = {"usage_metadata": {}, "response_metadata": {"token_usage": {"total_tokens": 9}}}
    assert _extract_token_usage([msg]) == 9

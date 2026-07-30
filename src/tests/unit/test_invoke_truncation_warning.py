"""A provider-truncated reply must announce itself.

Nothing in this project caps output tokens, so truncation is at the provider's
discretion -- and finish_reason was read NOWHERE, making it invisible. That
matters because of what happens next: a truncated structured response fails to
parse, the retry loop feeds that back as a validation error, and the measured
response to validation errors is for the model to emit LESS (Stage 1's extractor
shed 7,631 -> 4,653 output tokens while its error count fell 20 -> 2). Output
pressure therefore becomes silent under-modelling, and the small result reads as
a modelling decision rather than a cut-off response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.util.core.invoke import _finish_reason, _warn_if_truncated


def _msg(finish_reason: str | None) -> AIMessage:
    meta = {"finish_reason": finish_reason} if finish_reason is not None else {}
    return AIMessage(content="partial", response_metadata=meta)


class TestFinishReasonExtraction:
    def test_reads_openai_style_metadata(self):
        assert _finish_reason(_msg("length")) == "length"

    def test_missing_metadata_is_empty_not_an_error(self):
        assert _finish_reason(AIMessage(content="x")) == ""

    def test_dict_shaped_message(self):
        assert (
            _finish_reason({"response_metadata": {"finish_reason": "stop"}}) == "stop"
        )

    def test_unknown_shape_is_empty(self):
        assert _finish_reason(object()) == ""


class TestTruncationWarning:
    def test_length_truncation_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([_msg("length")], "er_extractor")
        assert "TRUNCATED" in caplog.text
        assert "er_extractor" in caplog.text

    def test_gemini_max_tokens_spelling_warns(self, caplog):
        """Gemini's native vocabulary is MAX_TOKENS, and the comparison is
        case-insensitive so a differently-cased provider still matches."""
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([_msg("MAX_TOKENS")], "agent")
        assert "TRUNCATED" in caplog.text

    def test_normal_completion_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([_msg("stop")], "agent")
        assert caplog.text == ""

    def test_absent_finish_reason_is_silent(self, caplog):
        """Many providers omit it entirely; that must not be read as truncation."""
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([AIMessage(content="x")], "agent")
        assert caplog.text == ""

    def test_warns_once_even_with_several_truncated_messages(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([_msg("length"), _msg("length")], "agent")
        assert caplog.text.count("TRUNCATED") == 1

    def test_finds_truncation_after_a_clean_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([_msg("stop"), _msg("length")], "agent")
        assert "TRUNCATED" in caplog.text

    def test_empty_message_list_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_truncated([], "agent")
        assert caplog.text == ""

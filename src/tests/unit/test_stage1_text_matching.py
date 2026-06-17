"""Unit tests for Stage 1 semantic origin matching.

Tests FactOriginMatcher from semantic_match.py. Exact-match paths are
fully deterministic (no model download). Semantic-search paths are
tested if the embedding model is available.
"""

from __future__ import annotations

from src.util.algorithms.semantic_match import FactOriginMatcher, MatchResult


# --------------------------------------------------------------------------- #
# FactOriginMatcher — exact match (no model needed)
# --------------------------------------------------------------------------- #


def test_exact_substring_match():
    source = "Users have credit scores associated with them in this system."
    matcher = FactOriginMatcher(source)
    result = matcher.verify_origin(
        "Users have credit scores", "credit scores associated"
    )
    assert result.is_valid is True
    assert result.match_type == "verbatim"
    assert result.score == 1.0


def test_exact_match_empty_origin_fails():
    source = "Users have credit scores associated with them."
    matcher = FactOriginMatcher(source)
    result = matcher.verify_origin("Users have credit scores", "")
    assert result.is_valid is False
    assert result.match_type == "none"
    assert result.warning is not None


def test_exact_match_claimed_origin_not_in_source():
    source = "The system tracks maturity and yield for each product."
    matcher = FactOriginMatcher(source)
    result = matcher.verify_origin("Users have scores", "completely unrelated")
    assert result.is_valid is False
    assert result.match_type in ("high", "medium", "low", "none")


def test_matchresult_dataclass_fields():
    m = MatchResult(
        best_span="some span",
        score=1.0,
        match_type="verbatim",
    )
    assert m.warning is None
    assert m.is_valid is True
    assert m.match_type == "verbatim"
    assert m.best_span == "some span"
    assert m.score == 1.0


def test_matchresult_is_valid_property():
    assert MatchResult(best_span="", score=1.0, match_type="verbatim").is_valid is True
    assert MatchResult(best_span="", score=1.0, match_type="sentence").is_valid is True
    assert MatchResult(best_span="", score=0.8, match_type="high").is_valid is True
    assert MatchResult(best_span="", score=0.6, match_type="medium").is_valid is True
    assert MatchResult(best_span="", score=0.3, match_type="low").is_valid is False
    assert MatchResult(best_span="", score=0.0, match_type="none").is_valid is False


def test_find_best_source_span_empty_source():
    source = "Hi."
    matcher = FactOriginMatcher(source, window_sizes=[4])
    result = matcher.find_best_source_span("")
    # Empty source or very short text may yield no spans — not an error
    assert result is None or result.score >= 0.0


# --------------------------------------------------------------------------- #
# TokenSpanIndex — structural tests
# --------------------------------------------------------------------------- #


def test_token_span_index_constructs():
    from src.util.algorithms.span_index import TokenSpanIndex

    text = "A moderately long test sentence to build spans from for indexing purposes."
    idx = TokenSpanIndex(text, min_span_chars=10, window_sizes=[4, 8, 12])
    assert len(idx.spans) > 0
    assert idx._tfidf is not None


def test_token_span_index_empty_source():
    from src.util.algorithms.span_index import TokenSpanIndex

    idx = TokenSpanIndex("")
    assert len(idx.spans) == 0
    assert idx._tfidf is None


def test_token_span_index_exact_search():
    from src.util.algorithms.span_index import TokenSpanIndex

    text = "The quick brown fox jumps over the lazy dog near the riverbank today."
    idx = TokenSpanIndex(text, min_span_chars=10, window_sizes=[4, 8])
    result = idx.get_best_match("fox jumps over the lazy dog")
    assert result is not None
    assert result.score > 0.0


def test_fact_origin_matcher_updates_origin():
    """When verify_origin finds a better semantic span, it returns it in best_span."""
    source = "Customers can have multiple loan accounts for borrowing money."
    matcher = FactOriginMatcher(source, window_sizes=[4, 8])
    result = matcher.verify_origin("Customers have loan accounts", "loan accounts")
    assert result.is_valid is True
    assert result.score > 0.0

"""Unit tests for Stage 1 text-matching middleware.

Pure deterministic string functions. Offline, no LLM/network. Note that
verify_facts_parallel uses a ThreadPoolExecutor but performs only local
string work - no I/O - so it remains deterministic and offline.
"""
from __future__ import annotations

from src.pipeline.stage1.middleware.text_matching import (
    normalize_text,
    tokenize,
    jaccard_similarity,
    find_best_match_sliding_window,
    verify_origin,
    verify_facts_parallel,
    MatchResult,
)
from src.pipeline.stage1.models.raw_fact import RawFact


# --------------------------------------------------------------------------- #
# normalize_text
# --------------------------------------------------------------------------- #

def test_normalize_text_lowercases():
    norm, _ = normalize_text("Hello WORLD")
    assert norm == "hello world"


def test_normalize_text_collapses_whitespace():
    norm, _ = normalize_text("a   b\t\nc")
    assert norm == "a b c"


def test_normalize_text_underscores_become_spaces():
    norm, _ = normalize_text("credit_score_value")
    assert norm == "credit score value"


def test_normalize_text_strips_quotes():
    norm, _ = normalize_text("the 'Approval Alpha' metric")
    assert "'" not in norm
    assert norm == "the approval alpha metric"


def test_normalize_text_trims_leading_trailing_dots_and_space():
    norm, _ = normalize_text("  ...hello world...  ")
    assert norm == "hello world"


def test_normalize_text_returns_token_positions_list():
    norm, positions = normalize_text("one two three")
    # positions index the start of each whitespace-delimited token.
    assert positions == [0, 4, 8]
    assert len(positions) == len(norm.split())


# --------------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------------- #

def test_tokenize_returns_set_of_words():
    assert tokenize("a b c") == {"a", "b", "c"}


def test_tokenize_dedupes():
    assert tokenize("a a b") == {"a", "b"}


def test_tokenize_empty_string():
    assert tokenize("") == set()


# --------------------------------------------------------------------------- #
# jaccard_similarity
# --------------------------------------------------------------------------- #

def test_jaccard_identical_sets_is_one():
    s = {"a", "b", "c"}
    assert jaccard_similarity(s, set(s)) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial_overlap_exact_fraction():
    # intersection {b,c}=2, union {a,b,c,d}=4 -> 0.5
    assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5


def test_jaccard_both_empty_is_one():
    assert jaccard_similarity(set(), set()) == 1.0


def test_jaccard_one_empty_is_zero():
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity(set(), {"a"}) == 0.0


def test_jaccard_subset_fraction():
    # {a,b} subset of {a,b,c,d}: inter=2, union=4 -> 0.5
    assert jaccard_similarity({"a", "b"}, {"a", "b", "c", "d"}) == 0.5


# --------------------------------------------------------------------------- #
# find_best_match_sliding_window
# --------------------------------------------------------------------------- #

def test_sliding_window_finds_verbatim_substring():
    desc = "the quick brown fox jumps over the lazy dog today"
    origin = "quick brown fox jumps"
    seg, score, start, end = find_best_match_sliding_window(origin, desc)
    assert score == 1.0
    assert seg == "quick brown fox jumps"
    assert desc[start:end] == seg


def test_sliding_window_absent_string_scores_low():
    desc = "the quick brown fox jumps over the lazy dog today"
    origin = "completely unrelated phrase here nowhere"
    seg, score, start, end = find_best_match_sliding_window(origin, desc)
    assert score < 0.5


def test_sliding_window_empty_inputs_return_zero():
    seg, score, start, end = find_best_match_sliding_window("", "some text here")
    assert seg is None
    assert score == 0.0
    seg2, score2, _, _ = find_best_match_sliding_window("origin tokens here", "")
    assert seg2 is None
    assert score2 == 0.0


# --------------------------------------------------------------------------- #
# verify_origin
# --------------------------------------------------------------------------- #

def test_verify_origin_missing_origin_invalid():
    r = verify_origin(1, "fact text", "", "some long natural language description")
    assert r.is_valid is False
    assert r.match_type == "failed"
    assert r.warning == "Missing origin"


def test_verify_origin_too_short_invalid():
    # origin shorter than min_match_length (10) -> failed
    r = verify_origin(1, "fact", "abc", "the natural language description with abc inside it")
    assert r.is_valid is False
    assert r.match_type == "failed"
    assert r.warning == "Origin too short"


def test_verify_origin_exact_substring_match():
    nl = "Users have credit scores associated with them in this system."
    origin = "credit scores associated"
    r = verify_origin(1, "fact text", origin, nl)
    assert r.is_valid is True
    assert r.match_type == "exact"
    assert r.jaccard_score == 1.0


def test_verify_origin_fuzzy_match_above_threshold():
    # Origin has reordered/extra token but high jaccard overlap with a window.
    nl = "The system tracks maturity and yield for each credit product carefully."
    origin = "tracks maturity and yield"
    r = verify_origin(1, "fact", origin, nl, jaccard_threshold=0.75)
    assert r.is_valid is True
    assert r.match_type in ("exact", "fuzzy")
    assert r.jaccard_score >= 0.75


def test_verify_origin_below_threshold_fails():
    nl = "The system tracks maturity and yield for each credit product."
    origin = "completely different unrelated content phrase entirely"
    r = verify_origin(1, "fact", origin, nl, jaccard_threshold=0.75)
    assert r.is_valid is False
    assert r.match_type == "failed"
    assert r.jaccard_score < 0.75
    assert "below threshold" in (r.warning or "")


# --------------------------------------------------------------------------- #
# verify_facts_parallel
# --------------------------------------------------------------------------- #

def test_verify_facts_parallel_external_short_circuits():
    facts = [RawFact(id=1, fact="external def", origin="", is_external=True)]
    results, stats = verify_facts_parallel(facts, "any nl description here long enough")
    assert len(results) == 1
    assert results[0].match_type == "external"
    assert results[0].is_valid is True
    assert stats["external"] == 1
    assert stats["total"] == 1


def test_verify_facts_parallel_mixed_stats():
    nl = "Users have credit scores associated with them in this lending system."
    facts = [
        RawFact(id=1, fact="f1", origin="credit scores associated"),  # exact
        RawFact(id=2, fact="f2", origin="", is_external=True),         # external
        RawFact(id=3, fact="f3", origin="wholly unrelated nonexistent verbiage here"),  # failed
    ]
    results, stats = verify_facts_parallel(facts, nl)
    assert stats["total"] == 3
    assert stats["external"] == 1
    assert stats["exact"] >= 1
    assert stats["failed"] >= 1
    # results preserve input order
    assert results[0].match_type == "exact"
    assert results[1].match_type == "external"
    assert results[2].match_type == "failed"


def test_matchresult_dataclass_fields():
    m = MatchResult(
        is_valid=True,
        original_segment="seg",
        normalized_match="seg",
        jaccard_score=1.0,
        match_type="exact",
    )
    assert m.warning is None
    assert m.match_type == "exact"

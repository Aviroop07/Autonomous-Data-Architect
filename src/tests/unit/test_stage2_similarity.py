"""Unit tests for Stage 2 TokenSimilarity (Jaccard-based).

Deterministic, offline. No ML models loaded. The similarity engine is a
pure keyword-overlap implementation that requires no downloads.
"""
from __future__ import annotations

import pytest

from src.pipeline.stage2.middleware.schema_merging.similarity import (
    TokenSimilarity,
    get_similarity_score,
    get_similarity_matrix,
)


@pytest.fixture
def sim():
    return TokenSimilarity()


# --------------------------------------------------------------------------- #
# _tokenize
# --------------------------------------------------------------------------- #

def test_tokenize_basic(sim):
    assert sim._tokenize("hello world") == {"hello", "world"}


def test_tokenize_underscores_to_spaces(sim):
    tokens = sim._tokenize("MARKET_DATA")
    assert "market" in tokens
    assert "data" in tokens


def test_tokenize_lowercases(sim):
    assert "Hello" not in sim._tokenize("Hello World")
    assert "hello" in sim._tokenize("Hello World")


def test_tokenize_empty(sim):
    assert sim._tokenize("") == set()


def test_tokenize_none_returns_empty(sim):
    assert sim._tokenize(None) == set()


# --------------------------------------------------------------------------- #
# _jaccard_similarity
# --------------------------------------------------------------------------- #

def test_jaccard_identical_strings(sim):
    assert sim._jaccard_similarity("credit score", "credit score") == 1.0


def test_jaccard_disjoint_strings(sim):
    assert sim._jaccard_similarity("apple banana", "cherry orange") == 0.0


def test_jaccard_partial_overlap(sim):
    # "credit score" vs "credit product": inter={"credit"}, union={"credit","score","product"} -> 1/3
    score = sim._jaccard_similarity("credit score", "credit product")
    assert 0.0 < score < 1.0


def test_jaccard_empty_first(sim):
    assert sim._jaccard_similarity("", "hello") == 0.0


def test_jaccard_empty_second(sim):
    assert sim._jaccard_similarity("hello", "") == 0.0


# --------------------------------------------------------------------------- #
# _context_overlap
# --------------------------------------------------------------------------- #

def test_context_overlap_superset_returns_one(sim):
    # "price" is fully contained in "volume price rate"
    score = sim._context_overlap("price", "volume price rate")
    assert score == 1.0


def test_context_overlap_disjoint_returns_zero(sim):
    assert sim._context_overlap("apple", "orange cherry") == 0.0


# --------------------------------------------------------------------------- #
# get_score
# --------------------------------------------------------------------------- #

def test_get_score_exact_match_is_one(sim):
    assert sim.get_score("customer_id", "customer_id") == 1.0


def test_get_score_case_insensitive_match_is_one(sim):
    assert sim.get_score("  Customer_ID  ", "customer_id") == 1.0


def test_get_score_empty_strings_returns_zero(sim):
    assert sim.get_score("", "anything") == 0.0
    assert sim.get_score("anything", "") == 0.0


def test_get_score_similar_names_higher_than_random(sim):
    similar = sim.get_score("loan_amount", "total_loan_amount")
    unrelated = sim.get_score("loan_amount", "zip_code")
    assert similar > unrelated


def test_get_score_underscore_tables(sim):
    score = sim.get_score("MARKET_DATA", "market_data")
    assert score == 1.0


# --------------------------------------------------------------------------- #
# get_matrix_scores
# --------------------------------------------------------------------------- #

def test_get_matrix_scores_shape(sim):
    matrix = sim.get_matrix_scores(["a", "b", "c"], ["x", "y"])
    assert len(matrix) == 3
    assert all(len(row) == 2 for row in matrix)


def test_get_matrix_scores_diagonal_is_one_for_identical(sim):
    names = ["customer_id", "order_date", "amount"]
    matrix = sim.get_matrix_scores(names, names)
    for i in range(len(names)):
        assert matrix[i][i] == 1.0


def test_get_matrix_scores_empty_list_a_returns_empty(sim):
    assert sim.get_matrix_scores([], ["a", "b"]) == []


def test_get_matrix_scores_empty_list_b_returns_empty(sim):
    assert sim.get_matrix_scores(["a"], []) == []


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #

def test_module_level_get_similarity_score_exact():
    assert get_similarity_score("user_id", "user_id") == 1.0


def test_module_level_get_similarity_matrix_shape():
    matrix = get_similarity_matrix(["a", "b"], ["x", "y", "z"])
    assert len(matrix) == 2
    assert all(len(row) == 3 for row in matrix)

"""Unit tests for Stage 2 SemanticSimilarity.

Deterministic, offline. Embedding behavior is tested through an injected fake
model so no model download is required.
"""
from __future__ import annotations

from typing import List

import pytest

from src.pipeline.stage2.middleware.schema_merging.similarity import (
    SemanticSimilarity,
    configure_default_similarity,
    get_similarity_matrix,
    get_similarity_score,
)


class FakeEmbeddingModel:
    def encode(self, sentences: List[str], **kwargs: object) -> List[List[float]]:
        vectors = {
            "customer": [1.0, 0.0, 0.0],
            "client": [1.0, 0.0, 0.0],
            "physician": [0.0, 1.0, 0.0],
            "doctor": [0.0, 1.0, 0.0],
            "movie": [1.0, 0.0, 0.0],
            "film": [1.0, 0.0, 0.0],
            "postal_code": [0.0, 0.0, 1.0],
            "unrelated": [0.0, 0.0, 1.0],
        }
        return [vectors.get(sentence, [0.0, 0.0, 1.0]) for sentence in sentences]


@pytest.fixture
def sim() -> SemanticSimilarity:
    return SemanticSimilarity(enable_embedding_model=False)


# --------------------------------------------------------------------------- #
# _tokenize
# --------------------------------------------------------------------------- #


def test_tokenize_basic(sim: SemanticSimilarity):
    assert sim._tokenize("hello world") == {"hello", "world"}


def test_tokenize_underscores_to_spaces(sim: SemanticSimilarity):
    tokens = sim._tokenize("MARKET_DATA")
    assert "market" in tokens
    assert "data" in tokens


def test_tokenize_lowercases(sim: SemanticSimilarity):
    assert "Hello" not in sim._tokenize("Hello World")
    assert "hello" in sim._tokenize("Hello World")


def test_tokenize_empty(sim: SemanticSimilarity):
    assert sim._tokenize("") == set()


def test_tokenize_none_returns_empty(sim: SemanticSimilarity):
    assert sim._tokenize(None) == set()


# --------------------------------------------------------------------------- #
# Lexical overlap
# --------------------------------------------------------------------------- #


def test_jaccard_identical_strings(sim: SemanticSimilarity):
    assert sim._jaccard_similarity("credit score", "credit score") == 1.0


def test_jaccard_disjoint_strings(sim: SemanticSimilarity):
    assert sim._jaccard_similarity("apple banana", "cherry orange") == 0.0


def test_jaccard_partial_overlap(sim: SemanticSimilarity):
    score = sim._jaccard_similarity("credit score", "credit product")
    assert 0.0 < score < 1.0


def test_jaccard_empty_first(sim: SemanticSimilarity):
    assert sim._jaccard_similarity("", "hello") == 0.0


def test_jaccard_empty_second(sim: SemanticSimilarity):
    assert sim._jaccard_similarity("hello", "") == 0.0


def test_context_overlap_superset_returns_one(sim: SemanticSimilarity):
    score = sim._context_overlap("price", "volume price rate")
    assert score == 1.0


def test_context_overlap_disjoint_returns_zero(sim: SemanticSimilarity):
    assert sim._context_overlap("apple", "orange cherry") == 0.0


# --------------------------------------------------------------------------- #
# Semantic scoring
# --------------------------------------------------------------------------- #


def test_get_score_exact_match_is_one(sim: SemanticSimilarity):
    assert sim.get_score("customer_id", "customer_id") == 1.0


def test_get_score_case_insensitive_match_is_one(sim: SemanticSimilarity):
    assert sim.get_score("  Customer_ID  ", "customer_id") == 1.0


def test_get_score_empty_strings_returns_zero(sim: SemanticSimilarity):
    assert sim.get_score("", "anything") == 0.0
    assert sim.get_score("anything", "") == 0.0


def test_get_score_similar_names_higher_than_random(sim: SemanticSimilarity):
    similar = sim.get_score("loan_amount", "total_loan_amount")
    unrelated = sim.get_score("loan_amount", "zip_code")
    assert similar > unrelated


def test_get_score_underscore_tables(sim: SemanticSimilarity):
    score = sim.get_score("MARKET_DATA", "market_data")
    assert score == 1.0


def test_injected_embedding_model_can_match_without_token_overlap():
    sim_with_embeddings = SemanticSimilarity(
        embedding_model=FakeEmbeddingModel(),
        enable_embedding_model=True,
    )
    semantic = sim_with_embeddings.get_score("customer", "client")
    unrelated = sim_with_embeddings.get_score("customer", "postal_code")
    assert semantic == 1.0
    assert semantic > unrelated


def test_injected_embedding_model_supports_domain_terms():
    sim_with_embeddings = SemanticSimilarity(
        embedding_model=FakeEmbeddingModel(),
        enable_embedding_model=True,
    )
    semantic = sim_with_embeddings.get_score("physician", "doctor")
    unrelated = sim_with_embeddings.get_score("physician", "unrelated")
    assert semantic == 1.0
    assert semantic > unrelated


# --------------------------------------------------------------------------- #
# get_matrix_scores
# --------------------------------------------------------------------------- #


def test_get_matrix_scores_shape(sim: SemanticSimilarity):
    matrix = sim.get_matrix_scores(["a", "b", "c"], ["x", "y"])
    assert len(matrix) == 3
    assert all(len(row) == 2 for row in matrix)


def test_get_matrix_scores_diagonal_is_one_for_identical(sim: SemanticSimilarity):
    names = ["customer_id", "order_date", "amount"]
    matrix = sim.get_matrix_scores(names, names)
    for i in range(len(names)):
        assert matrix[i][i] == 1.0


def test_get_matrix_scores_empty_list_a_returns_empty(sim: SemanticSimilarity):
    assert sim.get_matrix_scores([], ["a", "b"]) == []


def test_get_matrix_scores_empty_list_b_returns_empty(sim: SemanticSimilarity):
    assert sim.get_matrix_scores(["a"], []) == []


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #


def test_module_level_get_similarity_score_exact():
    assert get_similarity_score("user_id", "user_id") == 1.0


def test_module_level_get_similarity_matrix_shape():
    configure_default_similarity(SemanticSimilarity(enable_embedding_model=False))
    try:
        matrix = get_similarity_matrix(["a", "b"], ["x", "y", "z"])
        assert len(matrix) == 2
        assert all(len(row) == 3 for row in matrix)
    finally:
        configure_default_similarity(None)


def test_module_level_get_similarity_score_is_semantic():
    configure_default_similarity(
        SemanticSimilarity(embedding_model=FakeEmbeddingModel(), enable_embedding_model=True)
    )
    try:
        assert get_similarity_score("customer", "client") == 1.0
    finally:
        configure_default_similarity(None)

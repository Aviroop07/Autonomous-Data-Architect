"""Unit tests for src.util.matching.gale_shapley_matching.

The matcher implements row-proposing Gale-Shapley deferred acceptance over a
similarity score matrix. Preferences are derived from descending score, scores
below threshold are unacceptable, and ties break by lower index.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest

from src.util.matching import gale_shapley_matching


def _row_prefers(
    matrix: List[List[float]],
    row: int,
    candidate_col: int,
    current_col: Optional[int],
    threshold: float,
) -> bool:
    candidate_score = matrix[row][candidate_col]
    if candidate_score < threshold:
        return False
    if current_col is None:
        return True
    current_score = matrix[row][current_col]
    return candidate_score > current_score or (
        candidate_score == current_score and candidate_col < current_col
    )


def _column_prefers(
    matrix: List[List[float]],
    col: int,
    candidate_row: int,
    current_row: Optional[int],
    threshold: float,
) -> bool:
    candidate_score = matrix[candidate_row][col]
    if candidate_score < threshold:
        return False
    if current_row is None:
        return True
    current_score = matrix[current_row][col]
    return candidate_score > current_score or (
        candidate_score == current_score and candidate_row < current_row
    )


def _assert_no_blocking_pairs(
    matrix: List[List[float]],
    matches: List[Tuple[int, int]],
    threshold: float,
) -> None:
    row_to_col = {row: col for row, col in matches}
    col_to_row = {col: row for row, col in matches}
    for row, scores in enumerate(matrix):
        for col, score in enumerate(scores):
            if score < threshold or row_to_col.get(row) == col:
                continue
            if _row_prefers(matrix, row, col, row_to_col.get(row), threshold) and _column_prefers(
                matrix, col, row, col_to_row.get(col), threshold
            ):
                pytest.fail(f"Blocking pair found: row {row}, column {col}")


def test_empty_matrix_returns_empty_list():
    assert gale_shapley_matching([], 0.0) == []
    assert gale_shapley_matching([[]], 0.0) == []
    assert gale_shapley_matching([[], []], 0.0) == []


def test_single_pair_above_threshold():
    assert gale_shapley_matching([[0.9]], 0.5) == [(0, 0)]


def test_single_pair_below_threshold_is_skipped():
    assert gale_shapley_matching([[0.4]], 0.5) == []


def test_threshold_is_inclusive():
    assert gale_shapley_matching([[0.5]], 0.5) == [(0, 0)]


def test_perfect_diagonal():
    matrix = [
        [0.9, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.7],
    ]
    result = gale_shapley_matching(matrix, 0.5)
    assert result == [(0, 0), (1, 1), (2, 2)]
    _assert_no_blocking_pairs(matrix, result, 0.5)


def test_threshold_filters_low_scores():
    matrix = [
        [0.9, 0.2],
        [0.3, 0.6],
    ]
    result = gale_shapley_matching(matrix, 0.55)
    assert result == [(0, 0), (1, 1)]
    _assert_no_blocking_pairs(matrix, result, 0.55)


def test_deferred_acceptance_rejects_and_requeues_rows():
    matrix = [
        [0.80, 0.70],
        [0.90, 0.10],
    ]
    result = gale_shapley_matching(matrix, 0.5)
    assert result == [(0, 1), (1, 0)]
    _assert_no_blocking_pairs(matrix, result, 0.5)


def test_competition_loser_left_unmatched_when_no_fallback():
    matrix = [
        [0.80, 0.10],
        [0.95, 0.10],
    ]
    result = gale_shapley_matching(matrix, 0.5)
    assert result == [(1, 0)]
    _assert_no_blocking_pairs(matrix, result, 0.5)


def test_one_to_one_each_index_used_at_most_once():
    matrix = [
        [0.9, 0.85],
        [0.8, 0.95],
    ]
    result = gale_shapley_matching(matrix, 0.5)
    rows_used = [row for row, _ in result]
    cols_used = [col for _, col in result]
    assert len(rows_used) == len(set(rows_used))
    assert len(cols_used) == len(set(cols_used))
    _assert_no_blocking_pairs(matrix, result, 0.5)


def test_stable_matching_has_no_blocking_pairs():
    matrix = [
        [0.80, 0.70, 0.20],
        [0.90, 0.60, 0.50],
        [0.40, 0.30, 0.35],
    ]
    result = gale_shapley_matching(matrix, 0.30)
    assert result == [(0, 1), (1, 0), (2, 2)]
    _assert_no_blocking_pairs(matrix, result, 0.30)


def test_ties_are_deterministic():
    matrix = [
        [0.70, 0.70],
        [0.70, 0.70],
    ]
    expected = [(0, 0), (1, 1)]
    assert gale_shapley_matching(matrix, 0.5) == expected
    assert gale_shapley_matching(matrix, 0.5) == expected


def test_rectangular_more_rows_than_cols():
    matrix = [
        [0.80, 0.70],
        [0.90, 0.60],
        [0.40, 0.65],
    ]
    result = gale_shapley_matching(matrix, 0.5)
    assert result == [(0, 1), (1, 0)]
    _assert_no_blocking_pairs(matrix, result, 0.5)


def test_rectangular_more_cols_than_rows():
    matrix = [
        [0.80, 0.70, 0.10],
        [0.90, 0.60, 0.85],
    ]
    result = gale_shapley_matching(matrix, 0.5)
    assert result == [(0, 1), (1, 0)]
    _assert_no_blocking_pairs(matrix, result, 0.5)


def test_all_below_threshold_returns_empty():
    matrix = [
        [0.1, 0.2],
        [0.3, 0.05],
    ]
    assert gale_shapley_matching(matrix, 0.9) == []


def test_ragged_matrix_raises_value_error():
    with pytest.raises(ValueError, match="rectangular"):
        gale_shapley_matching([[0.9], [0.8, 0.7]], 0.5)

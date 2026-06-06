from collections import deque
from typing import Dict, List, Optional, Tuple


def gale_shapley_matching(score_matrix: List[List[float]], threshold: float) -> List[Tuple[int, int]]:
    """
    Gale-Shapley deferred-acceptance matching over a score matrix.

    Rows propose to columns. Both row and column preferences are derived from
    descending similarity score, with lower indices used as deterministic
    tie-breakers. Scores below ``threshold`` are treated as unacceptable.

    Args:
        score_matrix: score_matrix[i][j] is the similarity score between row i
            from set A and column j from set B.
        threshold: Inclusive minimum score for an acceptable pair.

    Returns:
        Stable one-to-one matches as ``(row_index, column_index)`` tuples,
        sorted by row index for deterministic downstream behavior.
    """
    rows = len(score_matrix)
    if rows == 0:
        return []

    cols = len(score_matrix[0])
    if any(len(row) != cols for row in score_matrix):
        raise ValueError("score_matrix must be rectangular")
    if cols == 0:
        return []

    row_preferences: List[List[int]] = []
    for i in range(rows):
        acceptable_cols = [j for j in range(cols) if score_matrix[i][j] >= threshold]
        acceptable_cols.sort(key=lambda j: (-score_matrix[i][j], j))
        row_preferences.append(acceptable_cols)

    column_ranks: List[Dict[int, int]] = []
    for j in range(cols):
        acceptable_rows = [i for i in range(rows) if score_matrix[i][j] >= threshold]
        acceptable_rows.sort(key=lambda i: (-score_matrix[i][j], i))
        column_ranks.append({row_idx: rank for rank, row_idx in enumerate(acceptable_rows)})

    next_proposal = [0] * rows
    free_rows = deque(i for i, preferences in enumerate(row_preferences) if preferences)
    accepted_by_column: List[Optional[int]] = [None] * cols

    while free_rows:
        row = free_rows.popleft()
        preferences = row_preferences[row]
        if next_proposal[row] >= len(preferences):
            continue

        col = preferences[next_proposal[row]]
        next_proposal[row] += 1

        incumbent = accepted_by_column[col]
        if incumbent is None:
            accepted_by_column[col] = row
            continue

        if column_ranks[col][row] < column_ranks[col][incumbent]:
            accepted_by_column[col] = row
            if next_proposal[incumbent] < len(row_preferences[incumbent]):
                free_rows.append(incumbent)
        elif next_proposal[row] < len(preferences):
            free_rows.append(row)

    matches = [
        (row, col)
        for col, row in enumerate(accepted_by_column)
        if row is not None
    ]
    matches.sort(key=lambda pair: pair[0])
    return matches

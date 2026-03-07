from typing import List, Tuple

def gale_shapley_matching(score_matrix: List[List[float]], threshold: float) -> List[Tuple[int, int]]:
    """
    Implements a greedy 1-1 matching algorithm (often referred to as greedy Gale-Shapley in this context).
    
    Args:
        score_matrix: A 2D list where score_matrix[i][j] is the similarity score between element i from set A 
                     and element j from set B.
        threshold: Scores below this threshold are ignored.
        
    Returns:
        A list of tuples (index_a, index_b) representing the best 1-1 mappings.
    """
    rows = len(score_matrix)
    if rows == 0:
        return []
    cols = len(score_matrix[0])
    
    # Create a list of all potential matches above the threshold
    candidates = []
    for i in range(rows):
        for j in range(cols):
            score = score_matrix[i][j]
            if score >= threshold:
                candidates.append((i, j, score))
    
    # Sort candidates by score in descending order
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    matched_a = [False] * rows
    matched_b = [False] * cols
    matches = []
    
    for i, j, score in candidates:
        if not matched_a[i] and not matched_b[j]:
            matches.append((i, j))
            matched_a[i] = True
            matched_b[j] = True
            
    return matches

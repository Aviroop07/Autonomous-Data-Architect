from typing import List, Optional, Set
import re

class TokenSimilarity:
    """
    Token-overlap similarity engine using Jaccard distance and context overlap.
    Purely lexical (no embeddings); replaces heavy Transformer-based bi-encoders
    for reliable pipeline performance.
    """
    def __init__(self, model_name: Optional[str] = None) -> None:
        # We ignore model_name as we are standardizing on Jaccard/Keyword
        self._threshold = 0.5
        pass

    def _tokenize(self, text: str) -> Set[str]:
        """Cleans and tokenizes text into lowercase words."""
        if not text:
            return set()
        # Remove punctuation, split by whitespace and underscores (e.g. MARKET_DATA -> market, data)
        text = text.lower().replace("_", " ")
        return set(re.findall(r'\w+', text))

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Standard Jaccard similarity: intersection / union."""
        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)

    def _context_overlap(self, s1: str, s2: str) -> float:
        """Calculates overlap relative to the smaller set (favors partial matches)."""
        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        return len(intersection) / min(len(tokens1), len(tokens2))

    def get_score(self, s1: str, s2: str) -> float:
        """
        Calculates a hybrid similarity score.
        Biases towards context overlap to handle technical rephrasings and schema matches.
        """
        if not s1 or not s2:
            return 0.0

        if s1.strip().lower() == s2.strip().lower():
            return 1.0

        jaccard = self._jaccard_similarity(s1, s2)
        overlap = self._context_overlap(s1, s2)

        # Weighted average: we favor the overlap for ADA applications
        # (e.g. 'Price' vs 'Volume Price' should have a high score)
        return (0.3 * jaccard) + (0.7 * overlap)

    def get_matrix_scores(self, list_a: List[str], list_b: List[str]) -> List[List[float]]:
        """
        Calculates a matrix of similarity scores between two lists of strings.
        """
        if not list_a or not list_b:
            return []

        matrix = []
        for val_a in list_a:
            row = []
            for val_b in list_b:
                row.append(self.get_score(val_a, val_b))
            matrix.append(row)

        return matrix

# Singleton instance for convenience
_default_similarity = None

def get_similarity_score(s1: str, s2: str) -> float:
    global _default_similarity
    if _default_similarity is None:
        _default_similarity = TokenSimilarity()
    return _default_similarity.get_score(s1, s2)

def get_similarity_matrix(list_a: List[str], list_b: List[str]) -> List[List[float]]:
    global _default_similarity
    if _default_similarity is None:
        _default_similarity = TokenSimilarity()
    return _default_similarity.get_matrix_scores(list_a, list_b)

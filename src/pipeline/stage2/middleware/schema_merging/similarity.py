import math
import re
from typing import List, Optional, Protocol, Set


class EmbeddingModel(Protocol):
    def encode(self, sentences: List[str], **kwargs: object) -> object: ...


class SemanticSimilarity:
    """
    Embedding-backed semantic similarity engine for schema names.

    Stage 2 uses this for table and column alignment. The primary signal is
    cosine similarity over sentence-transformer embeddings. A lexical overlap
    score is retained as a deterministic fallback for exact/near-exact names and
    for environments where the embedding model cannot be loaded.
    """

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        enable_embedding_model: bool = True,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self._embedding_model = embedding_model
        self._enable_embedding_model = enable_embedding_model
        self._embedding_load_attempted = embedding_model is not None

    def _tokenize(self, text: Optional[str]) -> Set[str]:
        """Cleans and tokenizes text into lowercase words."""
        if not text:
            return set()
        normalized = text.lower().replace("_", " ")
        return set(re.findall(r"\w+", normalized))

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Standard lexical Jaccard similarity: intersection / union."""
        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)

    def _context_overlap(self, s1: str, s2: str) -> float:
        """Lexical overlap relative to the smaller set."""
        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        return len(intersection) / min(len(tokens1), len(tokens2))

    def _lexical_score(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if s1.strip().lower() == s2.strip().lower():
            return 1.0
        jaccard = self._jaccard_similarity(s1, s2)
        overlap = self._context_overlap(s1, s2)
        return (0.3 * jaccard) + (0.7 * overlap)

    def _get_embedding_model(self) -> Optional[EmbeddingModel]:
        if self._embedding_model is not None:
            return self._embedding_model  # type: ignore[return-value]
        if not self._enable_embedding_model or self._embedding_load_attempted:
            return None

        self._embedding_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.model_name)  # type: ignore[assignment]
        except Exception:
            self._embedding_model = None
        return self._embedding_model  # type: ignore[return-value]

    def _embedding_score(self, s1: str, s2: str) -> float:
        model = self._get_embedding_model()
        if model is None:
            return 0.0

        import warnings

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Token indices sequence length"
                )
                embeddings = model.encode(
                    [s1, s2],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except TypeError:
            embeddings = model.encode([s1, s2])
        except Exception:
            return 0.0

        try:
            return self._cosine_similarity(embeddings[0], embeddings[1])  # type: ignore[index]
        except Exception:
            return 0.0

    def _cosine_similarity(self, left_vector: object, right_vector: object) -> float:
        left = self._to_float_list(left_vector)
        right = self._to_float_list(right_vector)
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(min(dot / (left_norm * right_norm), 1.0), 0.0)

    def _to_float_list(self, vector: object) -> List[float]:
        if hasattr(vector, "tolist"):
            raw_values = vector.tolist()  # type: ignore[attr-defined]
        else:
            raw_values = list(vector)  # type: ignore[arg-type]
        return [float(value) for value in raw_values]

    def get_score(self, s1: str, s2: str) -> float:
        """
        Calculates semantic schema-name similarity.

        Exact/near-exact lexical matches and embedding similarity are both valid
        evidence of equivalence; the score uses the strongest available signal.
        """
        if not s1 or not s2:
            return 0.0
        if s1.strip().lower() == s2.strip().lower():
            return 1.0
        lexical = self._lexical_score(s1, s2)
        embedding = self._embedding_score(s1, s2)
        return max(lexical, embedding)

    def get_matrix_scores(
        self, list_a: List[str], list_b: List[str]
    ) -> List[List[float]]:
        """Calculates a matrix of semantic similarity scores between two string lists."""
        if not list_a or not list_b:
            return []

        matrix = []
        for val_a in list_a:
            row = []
            for val_b in list_b:
                row.append(self.get_score(val_a, val_b))
            matrix.append(row)
        return matrix


_default_similarity: Optional[SemanticSimilarity] = None


def configure_default_similarity(similarity: Optional[SemanticSimilarity]) -> None:
    global _default_similarity
    _default_similarity = similarity


def get_similarity_score(s1: str, s2: str) -> float:
    global _default_similarity
    if _default_similarity is None:
        _default_similarity = SemanticSimilarity()
    return _default_similarity.get_score(s1, s2)


def get_similarity_matrix(list_a: List[str], list_b: List[str]) -> List[List[float]]:
    global _default_similarity
    if _default_similarity is None:
        _default_similarity = SemanticSimilarity()
    return _default_similarity.get_matrix_scores(list_a, list_b)

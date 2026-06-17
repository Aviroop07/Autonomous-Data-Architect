from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.util.algorithms.span_index import TokenSpanIndex, SearchResult


@dataclass(frozen=True)
class MatchResult:
    best_span: str
    score: float
    match_type: str  # "verbatim", "sentence", "high", "medium", "low", "none"
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.match_type in ("verbatim", "sentence", "high", "medium")


class FactOriginMatcher:
    """
    Semantic origin matching for fact verification using pre-computed token-span index.

    Uses NLTK sentence boundary detection as a pre-processing step:
    if the claimed origin or fact maps to a whole source sentence, that match
    is preferred over token-window semantic search.

    Match categories (ordinal):
        "verbatim"  — exact substring match (score=1.0)
        "sentence"  — NLTK sentence boundary match (score=1.0)
        "high"      — semantic score >= 0.75
        "medium"    — semantic score >= 0.50
        "low"       — semantic score >= 0.25
        "none"      — semantic score < 0.25 or no match
    """

    HIGH_THRESHOLD = 0.75
    MEDIUM_THRESHOLD = 0.50
    LOW_THRESHOLD = 0.25

    def __init__(
        self,
        source_text: str,
        **span_index_kwargs,
    ):
        self.source_text = source_text
        self.span_index = TokenSpanIndex(source_text, **span_index_kwargs)
        self._sentences: List[str] = self._split_sentences(source_text)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)

    def _sentence_match(self, claimed_origin: str) -> Optional[str]:
        """Check if claimed_origin matches a whole sentence (exact or via substring)."""
        if not claimed_origin:
            return None
        lower_origin = claimed_origin.lower().strip()
        for sent in self._sentences:
            if sent == claimed_origin:
                return sent
            if sent.lower() == lower_origin:
                return sent
            if lower_origin in sent.lower() or sent.lower() in lower_origin:
                continue
        return None

    @staticmethod
    def _categorize_score(score: float) -> str:
        if score >= FactOriginMatcher.HIGH_THRESHOLD:
            return "high"
        if score >= FactOriginMatcher.MEDIUM_THRESHOLD:
            return "medium"
        if score >= FactOriginMatcher.LOW_THRESHOLD:
            return "low"
        return "none"

    def verify_origin(self, fact_text: str, claimed_origin: str) -> MatchResult:
        """
        Verify if claimed_origin semantically matches fact_text against source_text.
        Returns MatchResult with best matching source span and ordinal match category.
        """
        # 1. Exact substring check (fast path)
        if claimed_origin and claimed_origin in self.source_text:
            return MatchResult(
                best_span=claimed_origin,
                score=1.0,
                match_type="verbatim",
            )

        # 2. Sentence-level match (NLTK-based boundary detection)
        sentence_match = self._sentence_match(claimed_origin)
        if sentence_match:
            return MatchResult(
                best_span=sentence_match,
                score=1.0,
                match_type="sentence",
            )

        # 3. Semantic search via FAISS token-span index
        result = self.span_index.get_best_match(fact_text)
        if result is None:
            return MatchResult(
                best_span="",
                score=0.0,
                match_type="none",
                warning="No valid spans found in source",
            )

        category = self._categorize_score(result.score)
        warning = None
        if category == "none":
            warning = f"Semantic score {result.score:.2f} below low threshold {self.LOW_THRESHOLD}"

        return MatchResult(
            best_span=result.span.text,
            score=result.score,
            match_type=category,
            warning=warning,
        )

    def find_best_source_span(self, fact_text: str) -> Optional[SearchResult]:
        """Find the semantically closest source span for a fact."""
        return self.span_index.get_best_match(fact_text)

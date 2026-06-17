from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


@dataclass(frozen=True)
class Span:
    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    window_size: int


@dataclass(frozen=True)
class SearchResult:
    span: Span
    score: float


def _word_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Return (token, char_start, char_end) for every non-whitespace run."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _terms(text: str) -> List[str]:
    """Lowercase alphabetic terms for TF-IDF (strips punctuation)."""
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


class _TfidfIndex:
    """
    Minimal TF-IDF vectorizer + FAISS IndexFlatIP.
    Pure numpy — no torch, no sklearn, no sentence_transformers.
    """

    def __init__(self, corpus: List[str]) -> None:
        n = len(corpus)
        df: Dict[str, int] = {}
        for doc in corpus:
            for t in set(_terms(doc)):
                df[t] = df.get(t, 0) + 1

        self._vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(df))}
        V = len(self._vocab)
        self._idf: np.ndarray = np.array(
            [np.log((1.0 + n) / (1.0 + df[t])) + 1.0 for t in sorted(df)],
            dtype="float32",
        )

        mat = np.zeros((n, V), dtype="float32")
        for i, doc in enumerate(corpus):
            counts = Counter(_terms(doc))
            for term, cnt in counts.items():
                if term in self._vocab:
                    j = self._vocab[term]
                    mat[i, j] = (1.0 + np.log(float(cnt))) * self._idf[j]
            norm = np.linalg.norm(mat[i])
            if norm > 1e-9:
                mat[i] /= norm

        self._index = faiss.IndexFlatIP(V)
        self._index.add(mat)

    def query(self, text: str, k: int) -> List[Tuple[int, float]]:
        V = len(self._vocab)
        vec = np.zeros((1, V), dtype="float32")
        counts = Counter(_terms(text))
        for term, cnt in counts.items():
            if term in self._vocab:
                j = self._vocab[term]
                vec[0, j] = (1.0 + np.log(float(cnt))) * self._idf[j]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        k_capped = min(k, self._index.ntotal)
        if k_capped == 0:
            return []
        scores, indices = self._index.search(vec, k_capped)
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


class TokenSpanIndex:
    """
    Sliding-window span index over source text with TF-IDF + FAISS exact search.

    Builds all word-window spans once, then serves semantic similarity queries
    in sub-millisecond time via normalized inner-product (cosine) search.
    """

    DEFAULT_WINDOW_SIZES = [12, 24, 40, 64, 96]
    DEFAULT_MIN_SPAN_CHARS = 20

    def __init__(
        self,
        source_text: str,
        window_sizes: Optional[List[int]] = None,
        min_span_chars: int = DEFAULT_MIN_SPAN_CHARS,
        **_ignored,
    ):
        self.source_text = source_text
        self.window_sizes = window_sizes or self.DEFAULT_WINDOW_SIZES
        self.min_span_chars = min_span_chars

        self._offsets = _word_offsets(source_text)
        self.spans = self._generate_spans()
        self._tfidf: Optional[_TfidfIndex] = (
            _TfidfIndex([s.text for s in self.spans]) if self.spans else None
        )

    def _generate_spans(self) -> List[Span]:
        spans: List[Span] = []
        offsets = self._offsets
        n = len(offsets)
        for w in self.window_sizes:
            if w > n:
                continue
            for i in range(n - w + 1):
                char_start = offsets[i][1]
                char_end = offsets[i + w - 1][2]
                if char_end - char_start < self.min_span_chars:
                    continue
                spans.append(
                    Span(
                        text=self.source_text[char_start:char_end],
                        token_start=i,
                        token_end=i + w,
                        char_start=char_start,
                        char_end=char_end,
                        window_size=w,
                    )
                )
        return spans

    def search(self, fact_text: str, k: int = 3) -> List[SearchResult]:
        if self._tfidf is None or not self.spans:
            return []
        return [
            SearchResult(span=self.spans[idx], score=score)
            for idx, score in self._tfidf.query(fact_text, k=k)
        ]

    def get_best_match(self, fact_text: str) -> Optional[SearchResult]:
        results = self.search(fact_text, k=1)
        return results[0] if results else None


def compute_source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]

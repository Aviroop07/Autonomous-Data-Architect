from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


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
    """Minimal TF-IDF vectorizer + exact cosine search over a SPARSE matrix.

    Rows are L2-normalized, so an inner product IS the cosine similarity and an
    exact top-k needs nothing more than one sparse matmul.

    This previously materialized a DENSE (n_spans x vocab) float32 array and
    handed it to faiss.IndexFlatIP. Both halves of that were a problem:

      Memory. Span counts grow as (windows x tokens) and vocabulary grows with
      the document, so the dense array is quadratic in document size while being
      ~99% zeros -- roughly 600 MB on a 10k-token specification, and the only
      O(n^2)-MEMORY defect in the codebase. A TF-IDF matrix is intrinsically
      sparse; storing the zeros was pure waste.

      Dependency weight. faiss-cpu was a required dependency of this
      reproducibility artifact for this single call site. IndexFlatIP performs a
      brute-force exact inner product -- precisely what a sparse matmul does --
      so there was no approximate-search capability being used to justify it.

    Results are unchanged in weights and scores; ordering is strictly MORE
    determined than before, since ties now break by ascending index rather than
    by whatever order faiss produced.
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

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for i, doc in enumerate(corpus):
            entries = self._weights(doc)
            for j, w in entries:
                rows.append(i)
                cols.append(j)
                vals.append(w)
        self._matrix = sparse.csr_matrix(
            (
                np.array(vals, dtype="float32"),
                (np.array(rows, dtype="int64"), np.array(cols, dtype="int64")),
            ),
            shape=(n, V),
            dtype="float32",
        )
        self._n = n
        self._vocab_size = V

    def _weights(self, text: str) -> List[Tuple[int, float]]:
        """L2-normalized TF-IDF weights for one document, as (column, value).

        Normalizing over just the present terms is identical to normalizing the
        full dense row -- the absent terms contribute zero to the norm.
        """
        counts = Counter(_terms(text))
        entries = [
            (self._vocab[t], (1.0 + np.log(float(c))) * self._idf[self._vocab[t]])
            for t, c in counts.items()
            if t in self._vocab
        ]
        if not entries:
            return []
        norm = float(np.linalg.norm(np.array([w for _, w in entries], dtype="float32")))
        if norm <= 1e-9:
            return []
        return [(j, float(w) / norm) for j, w in entries]

    def query(self, text: str, k: int) -> List[Tuple[int, float]]:
        k_capped = min(k, self._n)
        if k_capped <= 0 or self._vocab_size == 0:
            return []

        entries = self._weights(text)
        if not entries:
            return []
        query_vec = sparse.csr_matrix(
            (
                np.array([w for _, w in entries], dtype="float32"),
                (
                    np.zeros(len(entries), dtype="int64"),
                    np.array([j for j, _ in entries], dtype="int64"),
                ),
            ),
            shape=(1, self._vocab_size),
            dtype="float32",
        )

        scores = np.asarray((self._matrix @ query_vec.T).todense()).ravel()
        # lexsort on (index, -score) orders by descending score and breaks ties
        # by ascending index -- fully determined by the data, which faiss never
        # guaranteed and which this project needs for reproducibility.
        #
        # Deliberately a full sort rather than argpartition + a partial sort.
        # argpartition permutes arbitrarily within the selected block, so a
        # later stable sort preserves THAT arbitrary order, not index order --
        # equal-scoring spans came back in an order that depended on
        # partitioning internals. The sparse matmul above dominates cost
        # regardless, so O(n log n) here buys determinism for nothing.
        order = np.lexsort((np.arange(scores.size), -scores))[:k_capped]
        return [(int(i), float(scores[i])) for i in order]


class TokenSpanIndex:
    """
    Sliding-window span index over source text with exact TF-IDF cosine search.

    Builds all word-window spans once, then serves similarity queries via a
    sparse normalized inner product -- see _TfidfIndex for why this is sparse and
    dependency-free rather than dense-plus-faiss.
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

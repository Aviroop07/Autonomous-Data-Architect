"""The sparse TF-IDF index must match a dense reference exactly, and must not
allocate a dense matrix.

_TfidfIndex used to build a DENSE (n_spans x vocab) float32 array and hand it to
faiss.IndexFlatIP. Span count grows with (windows x tokens) and vocabulary grows
with the document, so that array is quadratic in document size and ~99% zeros --
about 600 MB on a 10k-token spec, the only O(n^2)-MEMORY defect in the codebase.

Correctness is pinned against an independent dense implementation of the same
maths rather than against the old code, so the test states what cosine TF-IDF
retrieval SHOULD return instead of merely freezing current behaviour.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy import sparse

from src.util.algorithms.span_index import TokenSpanIndex, _terms, _TfidfIndex


def _dense_reference(corpus: list[str], query: str, k: int):
    """Textbook dense TF-IDF + cosine top-k. Deliberately written from the
    definition, not copied from the implementation under test."""
    n = len(corpus)
    df: dict[str, int] = {}
    for doc in corpus:
        for t in set(_terms(doc)):
            df[t] = df.get(t, 0) + 1
    vocab = {t: i for i, t in enumerate(sorted(df))}
    idf = np.array(
        [np.log((1.0 + n) / (1.0 + df[t])) + 1.0 for t in sorted(df)], dtype="float64"
    )

    def vec(text: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype="float64")
        for term, cnt in Counter(_terms(text)).items():
            if term in vocab:
                j = vocab[term]
                v[j] = (1.0 + np.log(float(cnt))) * idf[j]
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v

    mat = np.stack([vec(d) for d in corpus]) if corpus else np.zeros((0, len(vocab)))
    scores = mat @ vec(query)
    order = np.argsort(-scores, kind="stable")[: min(k, n)]
    return [(int(i), float(scores[i])) for i in order]


CORPUS = [
    "the patient was admitted to the ward",
    "a doctor reviewed the patient chart",
    "equipment maintenance is scheduled monthly",
    "the ward has limited equipment",
    "billing records are reconciled quarterly",
]


class TestMatchesDenseReference:
    @pytest.mark.parametrize(
        "query",
        [
            "patient admitted ward",
            "doctor chart",
            "equipment",
            "quarterly billing reconciliation",
            "nothing in common with the corpus zzz",
        ],
    )
    def test_same_ranking_and_scores(self, query):
        idx = _TfidfIndex(CORPUS)
        got = idx.query(query, k=5)
        want = _dense_reference(CORPUS, query, k=5)
        # A query sharing no vocabulary scores 0 everywhere; the sparse path
        # returns nothing rather than an arbitrary zero-score ordering, which is
        # the more useful answer. Only compare when there is signal.
        if not got:
            assert all(abs(s) < 1e-9 for _, s in want)
            return
        assert [i for i, _ in got] == [i for i, _ in want[: len(got)]]
        for (_, a), (_, b) in zip(got, want):
            assert a == pytest.approx(b, abs=1e-6)

    def test_best_match_is_the_obvious_document(self):
        idx = _TfidfIndex(CORPUS)
        top = idx.query("billing reconciled quarterly", k=1)
        assert top and top[0][0] == 4


class TestNoDenseAllocation:
    def test_matrix_is_sparse(self):
        idx = _TfidfIndex(CORPUS)
        assert sparse.issparse(idx._matrix)

    def test_stored_entries_track_terms_not_the_grid(self):
        """The property that makes this linear instead of quadratic: stored
        values scale with total term occurrences, not n_docs x vocab."""
        idx = _TfidfIndex(CORPUS)
        rows, cols = idx._matrix.shape
        dense_cells = rows * cols
        assert idx._matrix.nnz < dense_cells
        # On real text the grid is overwhelmingly empty.
        assert idx._matrix.nnz <= dense_cells * 0.5

    def test_scales_without_allocating_the_dense_grid(self):
        """A document large enough that the dense form would be many times
        bigger. The dense array is never built, so this must stay cheap -- and
        the assertion is on the RATIO, computed from shapes, rather than by
        allocating the dense matrix to compare against."""
        text = " ".join(f"token{i % 400} filler word here" for i in range(2000))
        idx = TokenSpanIndex(text, window_sizes=[12, 24], min_span_chars=20)
        assert idx._tfidf is not None
        rows, cols = idx._tfidf._matrix.shape
        assert rows > 1000, "expected many spans from a 8k-token document"
        density = idx._tfidf._matrix.nnz / max(1, rows * cols)
        assert density < 0.1, f"matrix is {density:.1%} dense; sparsity is the point"


class TestDegenerateInputs:
    def test_empty_corpus(self):
        assert _TfidfIndex([]).query("anything", k=3) == []

    def test_corpus_of_empty_strings(self):
        assert _TfidfIndex(["", "", ""]).query("anything", k=3) == []

    def test_query_with_no_shared_terms_returns_nothing(self):
        assert _TfidfIndex(CORPUS).query("zzz qqq", k=3) == []

    def test_k_larger_than_corpus_is_capped(self):
        assert len(_TfidfIndex(CORPUS).query("patient", k=99)) <= len(CORPUS)

    def test_k_zero(self):
        assert _TfidfIndex(CORPUS).query("patient", k=0) == []

    def test_punctuation_only_document(self):
        idx = _TfidfIndex(["...", "the patient"])
        assert idx.query("patient", k=2)[0][0] == 1


class TestTieBreaking:
    def test_equal_scores_come_back_in_index_order(self):
        """Stable sort, so identical documents rank deterministically -- faiss
        gave no such guarantee, and this project needs reproducibility."""
        idx = _TfidfIndex(["same text here", "same text here", "different entirely"])
        top = idx.query("same text here", k=2)
        assert [i for i, _ in top] == [0, 1]

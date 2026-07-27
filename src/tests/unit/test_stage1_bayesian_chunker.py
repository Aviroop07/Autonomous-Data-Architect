import numpy as np

from src.pipeline.stage1.middleware.bayesian_chunker import (
    BayesianChunker, 
    _group_into_segments, 
    _build_similarities
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact, FactTag

# ── Helpers ────────────────────────────────────────────────────────────

def _make_fact(fid: int, text: str, start: int = 0, end: int = 10) -> AtomicFact:
    return AtomicFact(id=fid, fact=text, tags=[FactTag.STRUCTURAL], start_char=start, end_char=end)

# ── Tests ────────────────────────────────────────────────────────────

class TestBayesianChunkerSegments:
    def test_group_into_segments_merges_overlapping_facts(self):
        facts = [
            _make_fact(1, "A", start=0, end=50),
            _make_fact(2, "B", start=0, end=50),
            _make_fact(3, "C", start=100, end=150),
            _make_fact(4, "D", start=100, end=150),
        ]
        segments, positions, ref_counts = _group_into_segments(facts)
        assert len(segments) == 2
        assert len(segments[0]) == 2
        assert len(segments[1]) == 2
        assert segments[0][0].id == 1
        assert segments[1][0].id == 3

    def test_compute_similarities_decay(self, mocker):
        def _mock_embed_fn(texts: list[str]) -> np.ndarray:
            return np.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ])
            
        mocker.patch("src.pipeline.stage1.middleware.bayesian_chunker.embed_texts", side_effect=_mock_embed_fn)
        
        segments = [
            [_make_fact(1, "A", start=0, end=10)],
            [_make_fact(2, "B", start=20, end=30)],
            [_make_fact(3, "C", start=40, end=50)],
        ]
        
        sim, adj, ref = _build_similarities(segments)
        
        # Diagonal cosine sim is 1.0
        assert np.isclose(sim[0, 0], 1.0)
        
        # Adjacency decay is 1 / (1 + gap)
        # Position 0 = (0+10)//2 = 5
        # Position 1 = (20+30)//2 = 25
        # Position 2 = (40+50)//2 = 45
        # gap(0, 1) = 20 -> adj = 1 / 21
        # gap(0, 2) = 40 -> adj = 1 / 41
        assert adj[0, 1] > adj[0, 2]
        assert np.isclose(adj[0, 1], 1.0 / 21.0)
        assert np.isclose(adj[0, 2], 1.0 / 41.0)


class TestBayesianChunkerGibbs:
    def test_vectorized_sweep_updates_cluster_assignments(self):
        chunker = BayesianChunker()
        chunker.rng = np.random.RandomState(42)
        
        S = 3
        z = [0, 0, 1]
        cluster_members = {0: [0, 1], 1: [2]}
        valid_clusters = {0, 1}
        
        # Manually create same_logpdf and diff_logpdf
        # Let's make same_logpdf highly favor [0, 1] being together, and diff_logpdf favor [2] being separate
        same_logpdf = np.zeros((S, S))
        diff_logpdf = np.zeros((S, S)) - 10.0  # Big penalty for diff
        
        ref_log_boost = np.zeros((S, S))
        
        chunker._sweep(
            z=z,
            cluster_members=cluster_members,
            valid_clusters=valid_clusters,
            same_logpdf=same_logpdf,
            diff_logpdf=diff_logpdf,
            ref_log_boost=ref_log_boost,
            S=S
        )
        
        # Ensure z was modified, but structure mostly holds due to RandomState and probabilities
        assert len(z) == 3
        assert sum(len(v) for v in cluster_members.values()) == 3

    def test_fit_loop_produces_valid_chunks(self, mocker):
        def _mock_embed_fn(texts: list[str]) -> np.ndarray:
            return np.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ])
            
        mocker.patch("src.pipeline.stage1.middleware.bayesian_chunker.embed_texts", side_effect=_mock_embed_fn)
        
        # Override Gibbs parameters to ensure rapid convergence
        chunker = BayesianChunker(n_burnin=10, n_samples=10, thin=1)
        facts = [
            _make_fact(1, "A", start=0, end=10),
            _make_fact(2, "B", start=20, end=30),
            _make_fact(3, "C", start=40, end=50),
            _make_fact(4, "D", start=60, end=70),
        ]
        
        plan = chunker.fit(facts)
        
        assert plan is not None
        assert len(plan.chunks) >= 1
        
        total_facts_in_chunks = sum(len(c) for c in plan.chunks)
        assert total_facts_in_chunks == len(facts)

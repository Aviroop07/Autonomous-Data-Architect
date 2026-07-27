import logging
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.stats import beta as beta_dist

from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.util.embeddings.encoder import embed_texts

logger = logging.getLogger(__name__)

# ── segment grouping ──────────────────────────────────────────────────────


def _group_into_segments(
    facts: List[AtomicFact],
) -> Tuple[List[List[AtomicFact]], np.ndarray, List[int]]:
    """
    Group facts by (start_char, end_char) into segments.
    Enrichment facts (start_char=-1) each form their own segment.

    Returns (segments, segment_positions, segment_ref_counts).
    """
    groups: Dict[Tuple[int, int], List[AtomicFact]] = defaultdict(list)
    for f in facts:
        key = (f.start_char, f.end_char)
        groups[key].append(f)

    segment_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    segments: List[List[AtomicFact]] = [groups[k] for k in segment_keys]
    positions: List[int] = []
    for k in segment_keys:
        pos = k[0] if k[0] >= 0 else -1
        positions.append(pos)

    ref_counts: List[int] = []
    for seg in segments:
        n_ref = sum(1 for f in seg if f.referenced_fact_ids)
        ref_counts.append(n_ref)

    return segments, np.array(positions, dtype=np.int64), ref_counts


# ── similarity matrices (segment-level) ───────────────────────────────────


def _segment_embeddings(segments: List[List[AtomicFact]]) -> np.ndarray:
    texts: List[str] = []
    for seg in segments:
        combined = " ".join(f.fact for f in seg)
        texts.append(combined)
    emb = embed_texts(texts)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return emb / norms


def _build_similarities(
    segments: List[List[AtomicFact]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (sim, adj, ref):
      sim[i,j] = cosine similarity of segment centroids in [0, 1]
      adj[i,j] = positional adjacency in [0, 1]
      ref[i,j] = binary cross-reference indicator
    """
    S = len(segments)
    centroid_emb = _segment_embeddings(segments)
    sim = np.clip(centroid_emb @ centroid_emb.T, 0.0, 1.0)

    # Position
    first_fact = [seg[0] for seg in segments]
    positions = np.array(
        [
            (f.start_char + f.end_char) // 2 if f.start_char >= 0 else -1
            for f in first_fact
        ],
        dtype=np.float64,
    )
    adj = np.ones((S, S))
    for i in range(S):
        for j in range(S):
            if positions[i] >= 0 and positions[j] >= 0:
                gap = abs(positions[i] - positions[j])
                adj[i, j] = 1.0 / (1.0 + gap)
            else:
                adj[i, j] = 0.5 if i != j else 1.0

    # Cross-references between segments
    seg_fact_ids = [set(f.id for f in seg) for seg in segments]
    ref = np.zeros((S, S), dtype=np.float64)
    for i in range(S):
        referenced = set()
        for f in segments[i]:
            referenced.update(f.referenced_fact_ids)
        for j in range(S):
            if i != j and referenced & seg_fact_ids[j]:
                ref[i, j] = 1.0
                ref[j, i] = 1.0

    return sim, adj, ref


# ── Beta MoM ──────────────────────────────────────────────────────────────


def _beta_mom(values: np.ndarray) -> Tuple[float, float]:
    if len(values) == 0:
        return 1.0, 1.0
    mu = float(np.mean(values))
    var = float(np.var(values, ddof=0))
    if var >= mu * (1.0 - mu) or var <= 0.0:
        var = mu * (1.0 - mu) * 0.999
    scale = mu * (1.0 - mu) / var - 1.0
    a = max(mu * scale, 1.0)
    b = max((1.0 - mu) * scale, 1.0)
    return a, b


# ── Bayesian Partition Sampler ────────────────────────────────────────────


class BayesianChunker:
    """
    Bayesian partition model for fact clustering.

    Each segment (NL fragment emitted by the LLM) is an atomic unit.
    Combines three signals:
      - embedding cosine similarity (sim)
      - positional adjacency (adj)
      - cross-references (ref)

    The model:
      w_ij = λ * adj_ij + (1-λ) * sim_ij
      w_ij | z_i = z_j  ~  Beta(a_same, b_same)
      w_ij | z_i ≠ z_j  ~  Beta(a_diff, b_diff)
      cross-references add log(γ) to same-cluster log-posterior

    Inference via collapsed Gibbs sampling with empirical Bayes
    re-estimation of parameters.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 5.0,
        n_burnin: int = 2000,
        n_samples: int = 2000,
        thin: int = 5,
        re_estimate_every: int = 100,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        self.thin = thin
        self.re_estimate_every = re_estimate_every
        self.rng = np.random.default_rng(random_state)

    def fit(self, facts: List[AtomicFact]) -> ChunkedPlan:
        if len(facts) <= 1:
            return ChunkedPlan(core_modeling_facts=facts, chunks=[facts])

        segments, positions, _ = _group_into_segments(facts)
        S = len(segments)
        logger.info(f"[BayesianChunker] {len(facts)} facts grouped into {S} segments.")

        sim, adj, ref = _build_similarities(segments)
        logger.info(f"[BayesianChunker] Similarities computed ({S}x{S} matrices).")

        # Initial λ: adjacency weight. Estimate from global separation.
        # Higher weight on the signal with more dynamic range.
        sim_range = float(np.ptp(sim))
        adj_range = float(np.ptp(adj))
        lam = adj_range / (adj_range + sim_range + 1e-9)
        lam = np.clip(lam, 0.1, 0.9)
        w = lam * adj + (1.0 - lam) * sim
        np.fill_diagonal(w, 1.0)

        # Initialise Beta parameters with weakly informative defaults
        a_same, b_same = 4.0, 2.0
        a_diff, b_diff = 1.5, 6.0
        gamma_log = np.log(self.gamma)

        # Gibbs state
        z = list(range(S))  # each segment starts in its own cluster
        cluster_members: Dict[int, List[int]] = {i: [i] for i in range(S)}
        valid_clusters = set(range(S))

        # Precompute cross-ref log-boost matrix
        ref_log_boost = ref * gamma_log

        total_sweeps = self.n_burnin + self.n_samples * self.thin
        coassoc = np.zeros((S, S), dtype=np.float64)
        n_recorded = 0

        self._log_state(sim, adj, ref, w, lam, a_same, b_same, a_diff, b_diff)

        w_clip = np.clip(w, 1e-9, 1.0 - 1e-9)
        same_logpdf = beta_dist.logpdf(w_clip, a_same, b_same)
        diff_logpdf = beta_dist.logpdf(w_clip, a_diff, b_diff)

        for sweep in range(total_sweeps):
            self._sweep(
                z,
                cluster_members,
                valid_clusters,
                same_logpdf,
                diff_logpdf,
                ref_log_boost,
                S,
            )

            # Re-estimate parameters
            if sweep > 0 and sweep % self.re_estimate_every == 0:
                lam, a_same, b_same, a_diff, b_diff = self._re_estimate(
                    z, valid_clusters, sim, adj, w, a_same, b_same, a_diff, b_diff
                )
                w = lam * adj + (1.0 - lam) * sim
                np.fill_diagonal(w, 1.0)

                w_clip = np.clip(w, 1e-9, 1.0 - 1e-9)
                same_logpdf = beta_dist.logpdf(w_clip, a_same, b_same)
                diff_logpdf = beta_dist.logpdf(w_clip, a_diff, b_diff)

            # Record posterior sample
            if sweep >= self.n_burnin and (sweep - self.n_burnin) % self.thin == 0:
                for i in range(S):
                    for j in range(i + 1, S):
                        if z[i] == z[j]:
                            coassoc[i, j] += 1.0
                            coassoc[j, i] += 1.0
                n_recorded += 1

        if n_recorded > 0:
            coassoc /= n_recorded

        # Extract final partition from co-association matrix
        final_clusters = self._extract_partition(coassoc, S)

        # Map back to facts
        final_chunks: List[List[AtomicFact]] = []
        for cluster_idx in final_clusters:
            chunk: List[AtomicFact] = []
            for seg_idx in cluster_idx:
                chunk.extend(segments[seg_idx])
            final_chunks.append(chunk)

        logger.info(
            f"[BayesianChunker] Extracted {len(final_chunks)} final chunks "
            f"(recorded {n_recorded} posterior samples)."
        )
        return ChunkedPlan(core_modeling_facts=facts, chunks=final_chunks)

    # ── helpers ──────────────────────────────────────────────────────────

    def _log_state(self, sim, adj, ref, w, lam, a_same, b_same, a_diff, b_diff):
        same_edges = w[(w >= 0.3) & (w < 1.0)]
        diff_edges = w[(w < 0.3) & (w > 0.0)]
        mean_same = f"{np.mean(same_edges):.3f}" if len(same_edges) else "N/A"
        mean_diff = f"{np.mean(diff_edges):.3f}" if len(diff_edges) else "N/A"
        logger.info(
            f"[BayesianChunker] λ={lam:.3f}, "
            f"same Beta({a_same:.2f},{b_same:.2f}) "
            f"(mean_same={mean_same}), "
            f"diff Beta({a_diff:.2f},{b_diff:.2f}) "
            f"(mean_diff={mean_diff}), "
            f"ref_edges={int(ref.sum()) // 2}"
        )

    def _sweep(
        self,
        z: List[int],
        cluster_members: Dict[int, List[int]],
        valid_clusters: set,
        same_logpdf: np.ndarray,
        diff_logpdf: np.ndarray,
        ref_log_boost: np.ndarray,
        S: int,
    ):
        order = self.rng.permutation(S)
        for s in order:
            current_k = z[s]
            cluster_members[current_k].remove(s)
            if not cluster_members[current_k]:
                del cluster_members[current_k]
                valid_clusters.discard(current_k)

            candidates = list(valid_clusters)
            log_probs = np.zeros(len(candidates) + 1)  # +1 for new cluster

            # Scores for existing clusters
            for idx, k in enumerate(candidates):
                n_k = len(cluster_members[k])
                score = np.log(n_k + 1e-300)
                # Same-cluster pairs
                mask = cluster_members[k]
                if mask:
                    score += np.sum(same_logpdf[s, mask])
                    score += np.sum(ref_log_boost[s, mask])
                log_probs[idx] = score

            # Score for new cluster
            new_score = np.log(self.alpha + 1e-300)
            new_score += np.sum(diff_logpdf[s, :]) - diff_logpdf[s, s]
            log_probs[-1] = new_score

            # Normalise and sample
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs) + 1e-300

            choice = self.rng.multinomial(1, probs).argmax()
            if choice == len(candidates):
                new_k = max(valid_clusters) + 1 if valid_clusters else 0
                valid_clusters.add(new_k)
                cluster_members[new_k] = []
                z[s] = new_k
            else:
                z[s] = candidates[choice]
            cluster_members[z[s]].append(s)

    def _re_estimate(
        self,
        z: List[int],
        valid_clusters: set,
        sim: np.ndarray,
        adj: np.ndarray,
        w: np.ndarray,
        a_same: float,
        b_same: float,
        a_diff: float,
        b_diff: float,
    ) -> Tuple[float, float, float, float, float]:
        S = len(z)
        within_vals: List[float] = []
        between_vals: List[float] = []
        within_adj: List[float] = []
        between_adj: List[float] = []
        within_sim: List[float] = []
        between_sim: List[float] = []

        for i in range(S):
            for j in range(i + 1, S):
                same = z[i] == z[j]
                if same:
                    within_vals.append(w[i, j])
                    within_adj.append(adj[i, j])
                    within_sim.append(sim[i, j])
                else:
                    between_vals.append(w[i, j])
                    between_adj.append(adj[i, j])
                    between_sim.append(sim[i, j])

        # Update λ: fraction of total separation from adjacency
        d_adj = (
            float(np.mean(within_adj)) - float(np.mean(between_adj))
            if within_adj and between_adj
            else 0.0
        )
        d_sim = (
            float(np.mean(within_sim)) - float(np.mean(between_sim))
            if within_sim and between_sim
            else 0.0
        )
        lam_new = d_adj / (d_adj + d_sim + 1e-9)
        lam_new = float(np.clip(lam_new, 0.1, 0.9))

        # Update Beta parameters via MoM
        a_same_new, b_same_new = (
            _beta_mom(np.array(within_vals))
            if len(within_vals) >= 2
            else (a_same, b_same)
        )
        a_diff_new, b_diff_new = (
            _beta_mom(np.array(between_vals))
            if len(between_vals) >= 2
            else (a_diff, b_diff)
        )

        return lam_new, a_same_new, b_same_new, a_diff_new, b_diff_new

    def _extract_partition(self, coassoc: np.ndarray, S: int) -> List[List[int]]:
        if S == 0:
            return []
        # Extract clusters by thresholding the co-association matrix directly.
        # A `dist = np.clip(1.0 - coassoc, 0, 1)` line used to sit here under a
        # "hierarchical clustering" comment; nothing consumed it -- the method
        # below is plain thresholding, not hierarchical clustering.
        # Use 0.5 as the threshold — pairs that co-occur more than half the time
        threshold = 0.5
        remaining = set(range(S))
        clusters: List[List[int]] = []
        while remaining:
            seed = next(iter(remaining))
            cluster = {seed}
            changed = True
            while changed:
                changed = False
                for i in list(remaining - cluster):
                    if any(coassoc[i, j] >= threshold for j in cluster):
                        cluster.add(i)
                        changed = True
            clusters.append(sorted(cluster))
            remaining -= cluster
        return clusters

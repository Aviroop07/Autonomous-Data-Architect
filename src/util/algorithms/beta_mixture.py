"""2-component Beta mixture ensemble over many parameter configurations.

Each cross-shard pair has cosine sim x in [0,1]. The population is a
mixture: same-entity pairs (high sim) and diff-entity pairs (lower sim).

ENSEMBLE APPROACH (model averaging):
  Instead of picking one hyperparameter set, we run many configurations
  (different variance floors, priors, thresholds) and average posteriors.
  Configurations that agree produce confident P ≈ 0 or P ≈ 1.
  Configurations that disagree produce P ≈ 0.5 (defer to LLM).

  This replaces 7+ ad-hoc heuristics with a single principled ensemble.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.special import gammaln, logsumexp
from scipy import stats


CLIP_EPS = 1e-4
MIN_OBS_FOR_MIXTURE = 4
PI_MIN = 0.05




def _build_adaptive_grid(x: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Build config grid adaptively based on data diagnostics.

    Returns list of (var_floor, prior_a, prior_b, threshold) configs.
    Each config is a statistically valid point in hyperparameter space.
    The grid expands with stronger separation evidence and contracts
    when data is scarce or unstructured.
    """
    n = len(x)
    print("\n[AdaptiveGrid] ====== Grid Diagnostics ======")
    print(f"[AdaptiveGrid] n (cross-shard pairs) = {n}")

    # ── Feasibility gating ──────────────────────────────────────────────
    if n < MIN_OBS_FOR_MIXTURE:
        print(f"[AdaptiveGrid] REJECT: n={n} < MIN_OBS_FOR_MIXTURE ({MIN_OBS_FOR_MIXTURE})")
        return []

    # ── Distribution diagnostics ────────────────────────────────────────

    # Skew: measures asymmetry.  ER data is right-skewed (many low, few high).
    # High positive skew = strong evidence of two components.
    skew_val = float(stats.skew(x))
    if not np.isfinite(skew_val):
        skew_val = 0.0
    skew_used = max(0.0, skew_val)  # only positive skew signals separation
    print(f"[AdaptiveGrid] skew (raw) = {skew_val:.4f}  (positive = right-skewed = two components)")

    # IQR: spread of the middle 50%.  Low IQR + high skew = tight diff cluster + tail of same pairs.
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr_val = q75 - q25
    iqr_safe = max(iqr_val, 1e-6)
    print(f"[AdaptiveGrid] IQR = {iqr_val:.4f}  (q25={q25:.4f}, q75={q75:.4f})")

    # Max gap: largest jump between consecutive sorted similarities.
    # A large gap is nature's hint at a Bayes-optimal decision boundary.
    sx = np.sort(x)
    gaps = np.diff(sx)
    max_gap = float(np.max(gaps)) if len(gaps) > 0 else 0.0
    gap_mid: Optional[float] = None
    if len(gaps) > 0:
        gi = int(np.argmax(gaps))
        gap_mid = float((sx[gi] + sx[gi + 1]) / 2)
    gap_str = f"at x~{gap_mid:.4f}" if gap_mid is not None else "(no gap)"
    print(f"[AdaptiveGrid] max_gap = {max_gap:.4f} {gap_str}")
    print(f"[AdaptiveGrid]   (expected max gap under null ~ log(n)/n = {np.log(max(n,2))/max(n,2):.4f})")

    # p_exact: fraction of pairs with sim > 0.99 (exact name matches).
    # This gives a hard lower bound on pi (mixing proportion).
    p_exact = float(np.mean(x > 0.99))
    print(f"[AdaptiveGrid] p_exact (sim > 0.99) = {p_exact:.4f}")

    # KS test against uniform: record but don't gate (too weak for n < 10).
    # `_run_config`'s per-config checks are the reliable filter.
    try:
        D_ks, p_ks = stats.kstest(x, 'uniform')
        print(f"[AdaptiveGrid] KS(uniform) D={D_ks:.4f} p={p_ks:.4f}")
    except Exception:
        p_ks = 0.0

    # ── Separability score ──────────────────────────────────────────────
    # Measures how clearly two components can be distinguished.
    # Accounts for: asymmetry (skew), natural split point (gap/IQR).
    # Does NOT need to know ground truth labels.
    sep_score = skew_used * (1.0 + max_gap / iqr_safe)
    print("[AdaptiveGrid] sep_score = skew_used * (1 + max_gap/IQR)")
    print(f"[AdaptiveGrid]   = {skew_used:.4f} * (1 + {max_gap:.4f}/{iqr_safe:.4f})")
    print(f"[AdaptiveGrid]   = {sep_score:.4f}")
    print("[AdaptiveGrid]   (higher = stronger two-component signal)")

    # ── Build grid dimension-by-dimension ───────────────────────────────

    # --- Priors ---
    # Weak prior (5,1): lets data speak, needs n>15 or strong separation
    # Strong prior (15,2): dominates when data is scarce or ambiguous
    if n < 10:
        priors = [(15.0, 2.0), (10.0, 1.5)]
        print("[AdaptiveGrid] n<10 -> strong priors only")
    elif sep_score > 3.0:
        priors = [(5.0, 1.0), (8.0, 2.0), (10.0, 1.5), (15.0, 2.0)]
    elif sep_score > 1.5:
        priors = [(8.0, 2.0), (10.0, 1.5), (15.0, 2.0)]
    else:
        priors = [(10.0, 1.5), (15.0, 2.0)]
    print(f"[AdaptiveGrid] priors (as Beta(a,b) for same component): {priors}")
    for pa, pb in priors:
        print(f"[AdaptiveGrid]   Beta({pa:.1f},{pb:.1f}) mean={pa/(pa+pb):.3f}, effective_n={pa+pb:.0f}")

    # --- Variance floors ---
    # VF controls how concentrated the "diff" component can be.
    # Low VF: precise estimate, but risky if data is tightly packed.
    # High VF: conservative, broader diff component.
    if sep_score > 4.0:
        var_floors = [0.005, 0.010, 0.020, 0.050]
        print("[AdaptiveGrid] sep_score>4 -> full VF range (including aggressive 0.005)")
    elif sep_score > 2.0:
        var_floors = [0.010, 0.020, 0.050]
        print("[AdaptiveGrid] sep_score>2 -> moderate VF range")
    else:
        var_floors = [0.020, 0.050]
        print("[AdaptiveGrid] sep_score<=2 -> conservative VF range")

    # --- Thresholds ---
    # Controls which pairs are labeled "high" (pseudo-truth same).
    # Too low: high group contaminated by diff pairs.
    # Too high: high group too small for MoM.
    thresholds = [0.80, 0.85, 0.90]
    print(f"[AdaptiveGrid] base thresholds: {thresholds}")

    if p_exact > 0.15:
        thresholds = [t + 0.05 for t in thresholds]
        print(f"[AdaptiveGrid] p_exact>0.15 -> shifted thresholds up: {thresholds}")
    elif p_exact < 0.03:
        thresholds.insert(0, 0.75)
        thresholds = thresholds[:4]
        print(f"[AdaptiveGrid] p_exact<0.03 -> added low threshold 0.75: {thresholds}")

    if max_gap > 0.15 and gap_mid is not None and 0.30 <= gap_mid <= 0.85:
        nearest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - gap_mid))
        old_th = thresholds[nearest_idx]
        new_th = round(gap_mid, 2)
        thresholds[nearest_idx] = new_th
        print(f"[AdaptiveGrid] max_gap>0.15 -> replaced threshold {old_th} with {new_th} (centered on gap at x~{gap_mid:.4f})")

    # ── Assemble grid ───────────────────────────────────────────────────
    grid = [(vf, pa, pb, th) for vf in var_floors for pa, pb in priors for th in thresholds]
    print(f"[AdaptiveGrid] total configs: {len(grid)}")
    for cfg in grid:
        print(f"[AdaptiveGrid]   VF={cfg[0]:.3f}  prior=({cfg[1]:.1f},{cfg[2]:.1f})  thresh={cfg[3]:.2f}")
    print("[AdaptiveGrid] ====== End Diagnostics ======\n")

    return grid


def _log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    log_B = gammaln(a) + gammaln(b) - gammaln(a + b)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - log_B


def _beta_mom(x: np.ndarray, var_floor: float) -> tuple:
    """MoM estimate of Beta(alpha, beta) with variance floor."""
    if len(x) < 2:
        return None, None
    mu = float(np.mean(x))
    var = max(float(np.var(x, ddof=0)), var_floor)
    v = mu * (1.0 - mu) / var - 1.0
    if v <= 0:
        return None, None
    return mu * v, (1.0 - mu) * v


def _run_config(
    x: np.ndarray,
    var_floor: float,
    prior_a: float, prior_b: float,
    high_thresh: float,
    n_high_override: float = -1,
) -> tuple:
    """Run one config: returns (is_informative, posteriors) or (False, None)."""
    n = len(x)
    if n < MIN_OBS_FOR_MIXTURE:
        return False, None

    # Degenerate case: all values nearly identical -> no two components exist
    if np.var(x) < 1e-15:
        return False, None

    sx = np.sort(x)
    mid = n // 2

    low_mask = x <= sx[mid - 1]
    # Use override if provided (ensemble mode uses the same threshold for all)
    high_mask = x >= high_thresh

    low_x = x[low_mask]
    high_x = x[high_mask]

    # Both components need >= 2 points for MoM estimation.
    # If a group is too small, the config is uninformative (prior alone
    # can't distinguish components without data support).
    if len(low_x) < 2 or len(high_x) < 2:
        return False, None

    a_diff, b_diff = _beta_mom(low_x, var_floor)
    if a_diff is None or a_diff + b_diff <= 2.0:
        a_diff, b_diff = 2.0, 2.0

    if len(high_x) >= 2:
        a_same_r, b_same_r = _beta_mom(high_x, var_floor)
        if a_same_r is not None:
            a_same = max(prior_a, a_same_r)
            b_same = max(prior_b, b_same_r)
        else:
            a_same, b_same = prior_a, prior_b
    else:
        a_same, b_same = prior_a, prior_b

    pi = max(PI_MIN, min(1.0 - PI_MIN, len(high_x) / n))

    mu_diff = a_diff / (a_diff + b_diff)
    mu_same = a_same / (a_same + b_same)
    is_informative = (
        a_diff + b_diff > 2.0 and a_same + b_same > 2.0
        and abs(mu_same - mu_diff) > 0.05
        and pi > 0 and pi < 1.0
    )

    if not is_informative:
        return False, None

    lp_same = np.log(pi) + _log_beta_pdf(x, a_same, b_same)
    lp_diff = np.log(1.0 - pi) + _log_beta_pdf(x, a_diff, b_diff)
    lpr = np.column_stack([lp_same, lp_diff])
    posteriors = np.exp(lp_same - logsumexp(lpr, axis=1))

    if np.any(np.isnan(posteriors)):
        return False, None

    # Check that posteriors are monotonically increasing with x
    # (monotonicity is a fundamental sanity check for 1D mixture)
    order = np.argsort(x)
    monotonic = np.all(np.diff(posteriors[order]) >= -1e-6)
    if not monotonic:
        return False, None

    return True, posteriors


def fit_beta_mixture_ensemble(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Run ensemble of configurations and return (n_valid, averaged_posteriors).

    The config grid is built *adaptively* from the data's distributional
    properties (skew, IQR, max gap, exact-match ratio, sample size).
    See `_build_adaptive_grid` for details.

    Returns (n_configs_ok, posteriors) where posteriors[i] is the
    ensemble-averaged P(same | x[i]). If no config is informative,
    all posteriors are 0.5.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), CLIP_EPS, 1.0 - CLIP_EPS)
    n = len(x)

    if n < MIN_OBS_FOR_MIXTURE:
        return 0, np.full(n, 0.5)

    configs = _build_adaptive_grid(x)
    if not configs:
        return 0, np.full(n, 0.5)

    all_posts = []

    for vf, prior_a, prior_b, thresh in configs:
        ok, posts = _run_config(x, vf, prior_a, prior_b, thresh)
        if ok:
            all_posts.append(posts)

    if not all_posts:
        print(f"[BetaEnsemble] WARNING: {len(configs)} configs tried, 0 informative. Returning 0.5.")
        return 0, np.full(n, 0.5)

    # Average posteriors across all informative configs
    ensemble = np.mean(all_posts, axis=0)
    print(f"[BetaEnsemble] {len(all_posts)}/{len(configs)} configs informative -> ensemble ready")
    return len(all_posts), ensemble


# ── Public API ─────────────────────────────────────────────────────────

def compute_merge_probability_matrix(
    X: np.ndarray,
    shard_map: List[int],
) -> np.ndarray:
    """Return NxN matrix P where P[i,j] = P(same entity | sim(i,j))."""
    N = X.shape[0]
    P_out = np.zeros((N, N))
    i_up, j_up = np.triu_indices(N, 1)

    valid_mask = X[i_up, j_up] > -100
    valid_sims = X[i_up[valid_mask], j_up[valid_mask]]
    valid_pairs = list(zip(i_up[valid_mask], j_up[valid_mask]))
    n_valid = len(valid_sims)

    if n_valid == 0:
        return P_out

    n_ok, posteriors = fit_beta_mixture_ensemble(valid_sims)

    for (i, j), p in zip(valid_pairs, posteriors):
        P_out[i, j] = p
        P_out[j, i] = p
    return P_out


def fit_beta_mixture(x: np.ndarray) -> tuple:
    """Single-config Beta mixture fit for backward compatibility.

    Uses VF=0.010, prior=Beta(15,2), thresh=0.85, no EM.
    Returns (pi, a_diff, b_diff, a_same, b_same).

    This mirrors the fitting logic in `_run_config` but returns the
    actual component parameters instead of posteriors.
    """
    xc = np.clip(np.asarray(x, dtype=np.float64), CLIP_EPS, 1.0 - CLIP_EPS)
    n = len(xc)

    if n < MIN_OBS_FOR_MIXTURE:
        return 0.5, 1.0, 1.0, 1.0, 1.0

    vf, pa, pb, th = 0.010, 15.0, 2.0, 0.85

    sx = np.sort(xc)
    mid = n // 2
    low_x = xc[xc <= sx[mid - 1]]
    high_x = xc[xc >= th]

    if len(low_x) < 2 or len(high_x) < 2:
        return 0.5, 1.0, 1.0, 1.0, 1.0

    a_diff, b_diff = _beta_mom(low_x, vf)
    if a_diff is None or a_diff + b_diff <= 2.0:
        a_diff, b_diff = 2.0, 2.0

    a_same_r, b_same_r = _beta_mom(high_x, vf)
    if a_same_r is not None:
        a_same = max(pa, a_same_r)
        b_same = max(pb, b_same_r)
    else:
        a_same, b_same = pa, pb

    pi = max(PI_MIN, min(1.0 - PI_MIN, len(high_x) / n))

    mu_diff = a_diff / (a_diff + b_diff)
    mu_same = a_same / (a_same + b_same)
    if abs(mu_same - mu_diff) <= 0.05:
        return 0.5, 1.0, 1.0, 1.0, 1.0

    return pi, float(a_diff), float(b_diff), float(a_same), float(b_same)


def compute_posterior_same(
    x: np.ndarray,
    pi1: float = 0.5,
    alpha0: float = 1.0, beta0: float = 1.0,
    alpha1: float = 1.0, beta1: float = 1.0,
) -> np.ndarray:
    """Compute P(same|x) given component parameters or via ensemble.

    If caller provides non-default params, compute posterior directly
    from the Beta mixture formula (no fitting needed).
    If using defaults, run the ensemble fit instead.
    """
    x = np.asarray(x, dtype=np.float64)

    # Detect if caller provided specific component params (vs defaults)
    has_custom_params = (
        pi1 != 0.5 or alpha0 != 1.0 or beta0 != 1.0
        or alpha1 != 1.0 or beta1 != 1.0
    )

    if has_custom_params:
        pi1 = max(1e-10, min(1 - 1e-10, pi1))
        lp_same = np.log(pi1) + _log_beta_pdf(x, alpha1, beta1)
        lp_diff = np.log(1.0 - pi1) + _log_beta_pdf(x, alpha0, beta0)
        lpr = np.column_stack([lp_same, lp_diff])
        return np.exp(lp_same - logsumexp(lpr, axis=1))

    _, posts = fit_beta_mixture_ensemble(x)
    return posts


def compute_flag_posteriors(similarities: List[float]) -> List[float]:
    """Run ensemble fit and return posteriors."""
    x = np.array(similarities, dtype=np.float64)
    _, posteriors = fit_beta_mixture_ensemble(x)
    return posteriors.tolist()

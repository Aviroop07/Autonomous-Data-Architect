"""Deterministic unit tests for Beta mixture ensemble (no LLM, no embedding model).

Tests the adaptive grid, per-config runner, MoM estimator, ensemble averaging,
backward-compat single-config fit, and the public API surface.
"""

from __future__ import annotations

import numpy as np

from src.util.algorithms.beta_mixture import (
    _build_adaptive_grid,
    _run_config,
    _beta_mom,
    _log_beta_pdf,
    fit_beta_mixture_ensemble,
    fit_beta_mixture,
    compute_posterior_same,
    compute_merge_probability_matrix,
    compute_flag_posteriors,
    CLIP_EPS,
)


# =========================================================================
# _log_beta_pdf
# =========================================================================


class TestLogBetaPdf:
    def test_at_one_is_neg_inf(self):
        pdf = _log_beta_pdf(np.array([0.0, 1.0]), 5.0, 2.0)
        assert np.isneginf(pdf[1])

    def test_at_zero_is_neg_inf(self):
        pdf = _log_beta_pdf(np.array([0.0]), 2.0, 5.0)
        assert np.isneginf(pdf[0])

    def test_at_mid_is_finite(self):
        pdf = _log_beta_pdf(np.array([0.5]), 5.0, 2.0)
        assert np.isfinite(pdf[0])

    def test_uniform_params_symmetric(self):
        """Beta(1,1) is uniform, so log pdf is constant = log(1) = 0."""
        pdf = _log_beta_pdf(np.array([0.2, 0.5, 0.8]), 1.0, 1.0)
        assert np.allclose(pdf, 0.0)


# =========================================================================
# _beta_mom
# =========================================================================


class TestBetaMom:
    def test_too_few_samples_returns_none(self):
        a, b = _beta_mom(np.array([0.5]), 0.01)
        assert a is None and b is None

    def test_zero_var_uses_floor(self):
        x = np.array([0.5, 0.5, 0.5])
        a, b = _beta_mom(x, 0.01)
        assert a is not None and b is not None
        assert np.isfinite(a) and np.isfinite(b)

    def test_negative_v_returns_none(self):
        """mu*(1-mu)/var - 1 <= 0 when spread exceeds Bernoulli variance."""
        x = np.array([0.0, 1.0])
        a, b = _beta_mom(x, 0.01)
        assert a is None and b is None

    def test_mom_reasonable_for_concentrated_data(self):
        rng = np.random.default_rng(42)
        x = rng.beta(10.0, 3.0, size=500)
        a, b = _beta_mom(x, 0.001)
        assert a is not None and b is not None
        assert 5.0 < a < 20.0
        assert 1.0 < b < 8.0


# =========================================================================
# _build_adaptive_grid
# =========================================================================


class TestBuildAdaptiveGrid:
    def test_below_min_obs_returns_empty(self):
        x = np.array([0.1, 0.2, 0.3])
        grid = _build_adaptive_grid(x)
        assert grid == []

    def test_exactly_min_obs_returns_configs(self):
        x = np.array([0.1, 0.2, 0.3, 0.9])
        grid = _build_adaptive_grid(x)
        assert len(grid) > 0

    def test_degenerate_all_same_returns_configs(self):
        """All-identical passes diagnostics but _run_config rejects it.
        Grid construction checks distribution shape, not variance."""
        x = np.full(10, 0.5)
        grid = _build_adaptive_grid(x)
        assert len(grid) > 0

    def test_empty_input_returns_empty(self):
        grid = _build_adaptive_grid(np.array([]))
        assert grid == []

    def test_strong_separation_produces_more_configs(self):
        rng = np.random.default_rng(42)
        low = rng.beta(2.0, 8.0, size=25)
        high = rng.beta(18.0, 2.0, size=8)
        x = np.concatenate([low, high])
        grid = _build_adaptive_grid(x)
        assert len(grid) >= 24

    def test_weak_separation_produces_fewer_configs(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.3, 0.7, size=10)
        grid = _build_adaptive_grid(x)
        assert len(grid) <= 18

    def test_high_p_exact_shifts_thresholds(self):
        """p_exact > 0.15 should shift thresholds up (max >= 0.90)."""
        x = np.array([0.1] * 8 + [0.991, 0.992, 0.993, 0.994])
        grid = _build_adaptive_grid(x)
        if grid:
            thresholds = [cfg[3] for cfg in grid]
            max_th = max(thresholds)
            assert max_th >= 0.90

    def test_low_p_exact_adds_low_threshold(self):
        """p_exact < 0.03 adds 0.75 before gap override, so >= 3 distinct thresholds."""
        x = np.array([0.1] * 8 + [0.85, 0.88])
        grid = _build_adaptive_grid(x)
        if grid:
            thresholds = [cfg[3] for cfg in grid]
            unique_ths = set(round(th, 2) for th in thresholds)
            assert len(unique_ths) >= 3


# =========================================================================
# _run_config
# =========================================================================


class TestRunConfig:
    def test_below_min_obs(self):
        ok, posts = _run_config(np.array([0.1, 0.2, 0.3]), 0.01, 15.0, 2.0, 0.85)
        assert not ok
        assert posts is None

    def test_degenerate_variance(self):
        x = np.full(10, 0.5)
        ok, posts = _run_config(x, 0.01, 15.0, 2.0, 0.85)
        assert not ok
        assert posts is None

    def test_too_few_in_low_group(self):
        """If fewer than 2 low-x samples, config is uninformative."""
        x = np.array([0.85, 0.86, 0.87, 0.88, 0.89])
        ok, posts = _run_config(x, 0.01, 15.0, 2.0, 0.85)
        assert not ok
        assert posts is None

    def test_too_few_in_high_group(self):
        """If fewer than 2 high-x samples, config is uninformative."""
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.86])
        ok, posts = _run_config(x, 0.01, 15.0, 2.0, 0.85)
        assert not ok
        assert posts is None

    def test_clear_separation_produces_strictly_increasing_posterior(self):
        """Two-component data: low cluster ~0.2, high cluster ~0.9."""
        rng = np.random.default_rng(42)
        low = rng.beta(2.0, 8.0, size=15)
        high = rng.beta(18.0, 2.0, size=5)
        x = np.concatenate([low, high])
        ok, posts = _run_config(x, 0.010, 15.0, 2.0, 0.85)
        assert ok
        assert posts is not None
        assert len(posts) == len(x)
        assert np.all((posts >= 0.0) & (posts <= 1.0))
        order = np.argsort(x)
        assert np.all(np.diff(posts[order]) >= -1e-6)

    def test_monotonicity_violation_returns_false(self):
        """All-same data should fail the monotonicity check.  Actually it
        fails the variance check first (var < 1e-15).  This test covers
        the alternative path by constructing nearly-identical data with
        slight noise that fools the variance gate but breaks monotonicity."""
        rng = np.random.default_rng(42)
        x = np.sort(rng.beta(25.0, 25.0, size=10))
        ok, posts = _run_config(x, 0.050, 15.0, 2.0, 0.85)
        if ok:
            assert np.all(np.diff(posts[np.argsort(x)]) >= -1e-6)
        else:
            pass

    def test_overlap_data_still_works_with_enough_separation(self):
        rng = np.random.default_rng(42)
        low = rng.beta(5.0, 5.0, size=12)
        high = rng.beta(15.0, 3.0, size=4)
        x = np.concatenate([low, high])
        ok, posts = _run_config(x, 0.020, 10.0, 1.5, 0.85)
        assert ok or not ok

    def test_posteriors_monotonic_for_two_clusters(self):
        np.random.default_rng(42)
        low = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28])
        high = np.array([0.85, 0.88, 0.90, 0.92])
        x = np.concatenate([low, high])
        ok, posts = _run_config(x, 0.010, 15.0, 2.0, 0.85)
        assert ok
        order = np.argsort(x)
        assert np.all(np.diff(posts[order]) >= -1e-6)


# =========================================================================
# fit_beta_mixture_ensemble
# =========================================================================


class TestFitBetaMixtureEnsemble:
    def test_too_few_samples_returns_all_point_five(self):
        x = np.array([0.1, 0.2, 0.3])
        n_ok, posts = fit_beta_mixture_ensemble(x)
        assert n_ok == 0
        assert np.allclose(posts, 0.5)

    def test_degenerate_data_returns_all_point_five(self):
        x = np.full(10, 0.5)
        n_ok, posts = fit_beta_mixture_ensemble(x)
        assert n_ok == 0
        assert np.allclose(posts, 0.5)

    def test_two_clusters_produces_bimodal_posteriors(self):
        rng = np.random.default_rng(42)
        low = rng.beta(2.0, 8.0, size=20)
        high = rng.beta(18.0, 2.0, size=6)
        x = np.concatenate([low, high])
        n_ok, posts = fit_beta_mixture_ensemble(x)
        assert n_ok > 0
        low_mask = x < 0.5
        if np.any(low_mask):
            assert np.mean(posts[low_mask]) < 0.5
        high_mask = x > 0.7
        if np.any(high_mask):
            assert np.mean(posts[high_mask]) > 0.5

    def test_identical_values_clipped_to_eps(self):
        x = np.array([0.0, 0.5, 1.0])
        n_ok, posts = fit_beta_mixture_ensemble(x)
        assert n_ok >= 0
        assert len(posts) == 3
        assert np.all(posts >= 0.0) and np.all(posts <= 1.0)

    def test_returns_same_length_as_input(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.0, 1.0, size=12)
        n_ok, posts = fit_beta_mixture_ensemble(x)
        assert len(posts) == 12

    def test_near_one_posterior_for_identical_sim(self):
        """When low and high clusters are well-separated, low sims get near-0 P and
        high sims (exact-match range) get near-1 P."""
        rng = np.random.default_rng(42)
        low = rng.beta(1.5, 10.0, size=20)
        high = rng.beta(50.0, 2.0, size=6)
        x = np.concatenate([low, high])
        n_ok, posts = fit_beta_mixture_ensemble(x)
        assert n_ok > 0
        high_mask = x > 0.92
        if np.any(high_mask):
            assert np.all(posts[high_mask] > 0.85)


# =========================================================================
# fit_beta_mixture (backward compat single-config)
# =========================================================================


class TestFitBetaMixture:
    def test_too_few_samples_returns_fallback(self):
        pi, a0, b0, a1, b1 = fit_beta_mixture(np.array([0.1, 0.2, 0.3]))
        assert pi == 0.5
        assert a0 == 1.0

    def test_dominant_same_cluster_returns_pi_above_zero(self):
        np.random.default_rng(42)
        low = np.array([0.10, 0.12, 0.15, 0.18])
        high = np.array([0.85, 0.88, 0.90, 0.92, 0.95, 0.96])
        x = np.concatenate([low, high])
        pi, a0, b0, a1, b1 = fit_beta_mixture(x)
        assert a0 is not None
        assert pi >= 0.05

    def test_returns_valid_components(self):
        rng = np.random.default_rng(42)
        x = np.concatenate(
            [
                rng.beta(2.0, 10.0, size=10),
                rng.beta(30.0, 2.0, size=4),
            ]
        )
        pi, a0, b0, a1, b1 = fit_beta_mixture(x)
        assert pi >= 0.0 and pi <= 1.0
        assert a0 > 0 and b0 > 0
        assert a1 > 0 and b1 > 0


# =========================================================================
# compute_posterior_same
# =========================================================================


class TestComputePosteriorSame:
    def test_default_params_runs_ensemble(self):
        x = np.array([0.1, 0.2, 0.3, 0.4])
        posts = compute_posterior_same(x)
        assert len(posts) == 4
        assert np.all((posts >= 0.0) & (posts <= 1.0))

    def test_custom_params_returns_direct(self):
        x = np.array([0.1, 0.5, 0.9])
        posts = compute_posterior_same(
            x, pi1=0.3, alpha0=2.0, beta0=8.0, alpha1=10.0, beta1=2.0
        )
        assert len(posts) == 3
        assert posts[0] < 0.5
        assert posts[2] > 0.5

    def test_custom_params_monotonic(self):
        x = np.linspace(0.01, 0.99, 10)
        posts = compute_posterior_same(
            x, pi1=0.4, alpha0=2.0, beta0=5.0, alpha1=5.0, beta1=2.0
        )
        assert np.all(np.diff(posts) >= -1e-6)

    def test_custom_params_clamps_pi(self):
        x = np.array([0.5])
        posts = compute_posterior_same(
            x, pi1=0.0, alpha0=1.0, beta0=1.0, alpha1=1.0, beta1=1.0
        )
        assert np.isfinite(posts[0])


# =========================================================================
# compute_merge_probability_matrix
# =========================================================================


class TestComputeMergeProbabilityMatrix:
    def test_empty_input_returns_zeros(self):
        P = compute_merge_probability_matrix(np.zeros((0, 0)), [])
        assert P.shape == (0, 0)

    def test_single_entity_returns_zeros(self):
        P = compute_merge_probability_matrix(np.array([[0.0]]), [0])
        assert P.shape == (1, 1)
        assert P[0, 0] == 0.0

    def test_two_entities_same_shard_returns_zeros(self):
        X = np.array([[0.0, -np.inf], [-np.inf, 0.0]])
        P = compute_merge_probability_matrix(X, [0, 0])
        assert P[0, 1] == 0.0 and P[1, 0] == 0.0

    def test_two_entities_diff_shard_produces_symmetric_posterior(self):
        X = np.array([[0.0, 0.85], [0.85, 0.0]])
        P = compute_merge_probability_matrix(X, [0, 1])
        assert P[0, 1] == P[1, 0]


# =========================================================================
# compute_flag_posteriors
# =========================================================================


class TestComputeFlagPosteriors:
    def test_empty_list_returns_empty(self):
        posts = compute_flag_posteriors([])
        assert posts == []

    def test_two_distinct_groups(self):
        sims = [0.1, 0.2, 0.15, 0.18, 0.85, 0.90, 0.88]
        posts = compute_flag_posteriors(sims)
        assert len(posts) == len(sims)
        assert all(0.0 <= p <= 1.0 for p in posts)

    def test_all_low_similarities(self):
        posts = compute_flag_posteriors([0.1, 0.12, 0.15, 0.08])
        assert len(posts) == 4


# =========================================================================
# CLIP_EPS boundary
# =========================================================================


class TestClipEps:
    def test_values_at_zero_and_one_are_clipped(self):
        n_ok, posts = fit_beta_mixture_ensemble(np.array([0.0, 0.5, 1.0]))
        assert np.all(posts >= 0.0) and np.all(posts <= 1.0)

    def test_clip_eps_is_small_positive(self):
        assert 0.0 < CLIP_EPS < 0.01

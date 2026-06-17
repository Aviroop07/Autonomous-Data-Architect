"""Unit tests for src.util.dist_miner.

Deterministic, offline. All randomness is seeded with np.random.seed so the
synthetic samples are reproducible. We use generous tolerances: the point is
correct *family selection / ranking behavior*, not exact parameter recovery.

Note (observed behavior): for clean Normal data the miner often ranks Gamma
marginally above Normal by AIC (a large-shape Gamma approximates a Normal and
KS does not reject it). So for the normal case we assert Normal is *near the
top*, not strictly best -- matching the documented "best or near top" intent.
"""

import numpy as np
import pytest

from src.util.analysis.dist_miner import (
    DistributionFit,
    MiningResult,
    _clean,
    _is_discrete,
    mine_column_distribution,
    MIN_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_clean_drops_nan_and_inf():
    arr = _clean(np.array([1.0, np.nan, np.inf, -np.inf, 2.0, 3.0]))
    assert arr.tolist() == [1.0, 2.0, 3.0]


def test_clean_flattens_to_1d():
    arr = _clean(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert arr.ndim == 1
    assert arr.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_is_discrete_true_for_integer_data_with_few_uniques():
    np.random.seed(5)
    data = np.random.randint(1, 6, 300).astype(float)  # 5 unique integer values
    assert _is_discrete(data) is True


def test_is_discrete_false_for_continuous_float_data():
    np.random.seed(11)
    data = np.random.normal(0.0, 1.0, 300)
    assert _is_discrete(data) is False


def test_is_discrete_false_for_empty_array():
    assert _is_discrete(np.array([])) is False


def test_is_discrete_false_when_too_many_uniques():
    # 1000 distinct integers -> exceeds DISCRETE_MAX_UNIQUE (500)
    data = np.arange(1000, dtype=float)
    assert _is_discrete(data) is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_raises_on_too_few_samples():
    with pytest.raises(ValueError):
        mine_column_distribution([1.0, 2.0, 3.0])  # fewer than MIN_SAMPLES


def test_min_samples_constant_is_eight():
    assert MIN_SAMPLES == 8


# ---------------------------------------------------------------------------
# Family selection
# ---------------------------------------------------------------------------

def test_normal_data_ranks_normal_near_top():
    np.random.seed(42)
    data = np.random.normal(50.0, 5.0, 500)
    result = mine_column_distribution(data)
    top_families = [c.family for c in result.candidates[:3]]
    assert "normal" in top_families
    assert not result.is_discrete


def test_uniform_data_selects_uniform():
    np.random.seed(7)
    data = np.random.uniform(0.0, 100.0, 500)
    result = mine_column_distribution(data)
    assert result.best.family == "uniform"


def test_poisson_data_selects_poisson_and_is_discrete():
    np.random.seed(1)
    data = np.random.poisson(4.0, 600).astype(float)
    result = mine_column_distribution(data)
    assert result.is_discrete is True
    assert result.best.family == "poisson"


def test_zipf_data_selects_zipf_and_is_discrete():
    np.random.seed(3)
    data = np.random.zipf(2.0, 600).astype(float)
    data = data[data < 1000]  # trim extreme tail so unique count stays bounded
    result = mine_column_distribution(data)
    assert result.is_discrete is True
    assert result.best.family == "zipf"


def test_discrete_integer_few_uniques_is_flagged_discrete():
    np.random.seed(5)
    data = np.random.randint(1, 6, 300).astype(float)
    result = mine_column_distribution(data)
    assert result.is_discrete is True


def test_force_discrete_override_is_respected():
    np.random.seed(11)
    data = np.random.normal(100.0, 10.0, 300)  # naturally continuous
    result = mine_column_distribution(data, force_discrete=True)
    assert result.is_discrete is True


# ---------------------------------------------------------------------------
# AIC ordering & result structure
# ---------------------------------------------------------------------------

def test_valid_candidates_sorted_by_aic_ascending():
    np.random.seed(42)
    data = np.random.normal(50.0, 5.0, 500)
    result = mine_column_distribution(data)
    valid = [c for c in result.candidates if c.valid]
    aics = [c.aic for c in valid]
    assert aics == sorted(aics)


def test_best_is_first_valid_candidate():
    np.random.seed(42)
    data = np.random.normal(50.0, 5.0, 500)
    result = mine_column_distribution(data)
    valid = [c for c in result.candidates if c.valid]
    assert valid, "expected at least one valid candidate"
    assert result.best is valid[0]


def test_max_candidates_limits_list_length():
    np.random.seed(7)
    data = np.random.uniform(0.0, 100.0, 500)
    result = mine_column_distribution(data, max_candidates=2)
    assert len(result.candidates) <= 2


def test_result_and_fit_field_types():
    np.random.seed(7)
    data = np.random.uniform(0.0, 100.0, 500)
    result = mine_column_distribution(data)

    assert isinstance(result, MiningResult)
    assert isinstance(result.best, DistributionFit)
    assert isinstance(result.candidates, list)
    assert all(isinstance(c, DistributionFit) for c in result.candidates)
    assert result.n_samples == 500
    assert isinstance(result.is_discrete, bool)

    fit = result.best
    assert isinstance(fit.family, str)
    assert isinstance(fit.params, dict)
    assert isinstance(fit.n_params, int)
    assert isinstance(fit.log_likelihood, float)
    assert isinstance(fit.aic, float)
    assert isinstance(fit.bic, float)
    assert isinstance(fit.ks_statistic, float)
    assert isinstance(fit.ks_pvalue, float)
    assert isinstance(fit.valid, bool)


def test_as_ground_truth_spec_shape():
    np.random.seed(7)
    data = np.random.uniform(0.0, 100.0, 500)
    result = mine_column_distribution(data)
    spec = result.best.as_ground_truth_spec()
    assert set(spec.keys()) == {"family", "params"}
    assert spec["family"] == result.best.family
    assert spec["params"] == dict(result.best.params)


def test_summary_table_is_string_listing_families():
    np.random.seed(7)
    data = np.random.uniform(0.0, 100.0, 500)
    result = mine_column_distribution(data)
    table = result.summary_table()
    assert isinstance(table, str)
    assert result.best.family in table


def test_nan_inf_dropped_before_n_samples_count():
    np.random.seed(7)
    clean = np.random.uniform(0.0, 100.0, 100)
    polluted = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    result = mine_column_distribution(polluted)
    assert result.n_samples == 100  # the 3 non-finite values were dropped

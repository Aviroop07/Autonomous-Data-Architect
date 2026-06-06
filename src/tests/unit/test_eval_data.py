"""Unit tests for data-level evaluation helpers.

Deterministic, offline. No live API. Tests _mre, _ks, _parse_gt_dist,
estimate_params, and log_pdf from distributions.py.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.data_level.data_eval import (
    _mre,
    _ks,
    _parse_gt_dist,
    evaluate_column,
    WORST_CASE,
)
from src.evaluation.data_level.distributions import (
    estimate_params,
    log_pdf,
    max_density_point,
)


# --------------------------------------------------------------------------- #
# _mre
# --------------------------------------------------------------------------- #

def test_mre_perfect_params():
    assert _mre({"mean": 5.0, "std": 2.0}, {"mean": 5.0, "std": 2.0}) == 0.0


def test_mre_missing_pred_key_skipped():
    # Only "mean" is in pred; "std" is absent -> only mean contributes
    pred = {"mean": 5.0}
    gt = {"mean": 5.0, "std": 2.0}
    assert _mre(pred, gt) == 0.0


def test_mre_no_common_keys_returns_one():
    assert _mre({"alpha": 1.0}, {"mean": 5.0}) == 1.0


def test_mre_gt_param_zero_uses_one_as_denom():
    pred = {"mean": 0.5}
    gt = {"mean": 0.0}
    # denom = 1.0 (zero guard), error = 0.5
    assert _mre(pred, gt) == pytest.approx(0.5)


def test_mre_symmetric_error():
    pred = {"mean": 6.0}
    gt = {"mean": 5.0}
    error = _mre(pred, gt)
    assert error == pytest.approx(0.2)  # |6-5|/5 = 0.2


def test_mre_multiple_params_averaged():
    pred = {"mean": 6.0, "std": 2.0}
    gt = {"mean": 5.0, "std": 2.0}
    # error for mean: 0.2, error for std: 0.0 -> mean = 0.1
    assert _mre(pred, gt) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# _ks
# --------------------------------------------------------------------------- #

def test_ks_normal_data_vs_correct_gt_is_low():
    np.random.seed(0)
    data = np.random.normal(0.0, 1.0, 1000)
    ks = _ks(data, "normal", {"mean": 0.0, "std": 1.0})
    assert ks < 0.05  # well-fit -> low KS


def test_ks_normal_data_vs_wrong_gt_is_high():
    np.random.seed(0)
    data = np.random.normal(0.0, 1.0, 500)
    ks = _ks(data, "normal", {"mean": 1000.0, "std": 1.0})  # very wrong mean
    assert ks > 0.3


def test_ks_too_short_array_returns_one():
    assert _ks(np.array([1.0]), "normal", {"mean": 0.0, "std": 1.0}) == 1.0


def test_ks_invalid_family_returns_one():
    data = np.random.normal(0.0, 1.0, 100)
    assert _ks(data, "fantasy_distribution", {}) == 1.0


# --------------------------------------------------------------------------- #
# _parse_gt_dist
# --------------------------------------------------------------------------- #

def test_parse_normal_spec():
    result = _parse_gt_dist({"family": "normal", "params": {"mean": 7.0, "std": 1.3}})
    assert result is not None
    family, params = result
    assert family == "normal"
    assert params["mean"] == 7.0
    assert params["std"] == 1.3


def test_parse_lognormal_mean_variance_converts_to_mu_sigma():
    result = _parse_gt_dist({"family": "lognormal", "params": {"mean": 3.5, "variance": 1.44}})
    assert result is not None
    family, params = result
    assert family == "lognormal"
    assert "mu" in params
    assert "sigma" in params
    assert params["mu"] == pytest.approx(3.5)
    assert params["sigma"] == pytest.approx(math.sqrt(1.44))


def test_parse_lognormal_mu_sigma_passthrough():
    result = _parse_gt_dist({"family": "lognormal", "params": {"mu": 3.5, "sigma": 1.2}})
    assert result is not None
    _, params = result
    assert params["mu"] == pytest.approx(3.5)
    assert params["sigma"] == pytest.approx(1.2)


def test_parse_exponential_lambda_aliased_to_rate():
    result = _parse_gt_dist({"family": "exponential", "params": {"lambda": 0.5}})
    assert result is not None
    _, params = result
    assert "rate" in params
    assert params["rate"] == pytest.approx(0.5)
    assert "lambda" not in params


def test_parse_uniform_spec():
    result = _parse_gt_dist({"family": "uniform", "params": {"low": 0.0, "high": 100.0}})
    assert result is not None
    family, params = result
    assert family == "uniform"
    assert params["low"] == 0.0
    assert params["high"] == 100.0


def test_parse_malformed_returns_none():
    assert _parse_gt_dist({}) is None
    assert _parse_gt_dist({"family": 123}) is None


def test_parse_case_insensitive_family():
    result = _parse_gt_dist({"family": "Normal", "params": {"mean": 0.0, "std": 1.0}})
    assert result is not None
    assert result[0] == "normal"


# --------------------------------------------------------------------------- #
# estimate_params from distributions.py
# --------------------------------------------------------------------------- #

def test_estimate_params_normal():
    np.random.seed(42)
    data = np.random.normal(50.0, 5.0, 1000)
    params = estimate_params(data, "normal")
    assert "mean" in params and "std" in params
    assert abs(params["mean"] - 50.0) < 1.0
    assert abs(params["std"] - 5.0) < 1.0


def test_estimate_params_lognormal():
    np.random.seed(7)
    data = np.random.lognormal(mean=2.0, sigma=0.5, size=500)
    params = estimate_params(data, "lognormal")
    assert "mu" in params and "sigma" in params
    assert abs(params["mu"] - 2.0) < 0.2
    assert abs(params["sigma"] - 0.5) < 0.2


def test_estimate_params_exponential():
    np.random.seed(1)
    data = np.random.exponential(scale=3.0, size=1000)
    params = estimate_params(data, "exponential")
    assert "rate" in params
    assert abs(params["rate"] - 1.0 / 3.0) < 0.1


def test_estimate_params_uniform():
    np.random.seed(2)
    data = np.random.uniform(10.0, 50.0, 500)
    params = estimate_params(data, "uniform")
    assert "low" in params and "high" in params
    assert params["low"] >= 9.5
    assert params["high"] <= 50.5


def test_estimate_params_raises_for_empty():
    with pytest.raises(ValueError):
        estimate_params(np.array([]), "normal")


def test_estimate_params_raises_for_unknown_family():
    with pytest.raises(ValueError):
        estimate_params(np.array([1.0, 2.0, 3.0]), "fantasy_dist")


# --------------------------------------------------------------------------- #
# log_pdf
# --------------------------------------------------------------------------- #

def test_log_pdf_normal_at_mean_is_maximum():
    params = {"mean": 0.0, "std": 1.0}
    at_mean = log_pdf(np.array([0.0]), "normal", params)[0]
    at_tail = log_pdf(np.array([3.0]), "normal", params)[0]
    assert at_mean > at_tail


def test_log_pdf_uniform_outside_support_is_neg_inf():
    params = {"low": 0.0, "high": 1.0}
    # x=2.0 is outside [0,1], should be -inf
    val = log_pdf(np.array([2.0]), "uniform", params)[0]
    assert not math.isfinite(val)


def test_log_pdf_returns_array_same_shape():
    params = {"mean": 0.0, "std": 1.0}
    x = np.array([1.0, 2.0, 3.0])
    result = log_pdf(x, "normal", params)
    assert result.shape == x.shape


# --------------------------------------------------------------------------- #
# max_density_point
# --------------------------------------------------------------------------- #

def test_max_density_normal_is_mean():
    assert max_density_point("normal", {"mean": 7.5, "std": 1.0}) == pytest.approx(7.5)


def test_max_density_exponential_is_zero():
    assert max_density_point("exponential", {"rate": 2.0}) == 0.0


def test_max_density_zipf_is_one():
    assert max_density_point("zipf", {"a": 2.0}) == 1.0


# --------------------------------------------------------------------------- #
# evaluate_column (end-to-end for a single column)
# --------------------------------------------------------------------------- #

def test_evaluate_column_near_perfect_fit_has_low_ks():
    np.random.seed(5)
    data = np.random.normal(50.0, 5.0, 500)
    result = evaluate_column(data, {"family": "normal", "params": {"mean": 50.0, "std": 5.0}})
    assert result["ks"] < 0.1
    assert result["fa"] > 0.9


def test_evaluate_column_bad_fit_has_high_ks():
    np.random.seed(5)
    data = np.random.normal(50.0, 5.0, 500)
    result = evaluate_column(data, {"family": "normal", "params": {"mean": 1000.0, "std": 1.0}})
    assert result["ks"] > 0.3


def test_evaluate_column_malformed_spec_returns_worst_case():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = evaluate_column(data, {})
    assert result == WORST_CASE


def test_evaluate_column_too_short_returns_worst_case():
    result = evaluate_column(np.array([1.0]), {"family": "normal", "params": {"mean": 1.0, "std": 1.0}})
    assert result == WORST_CASE

"""
Data-level evaluation metrics for ScribbleDB.

Metrics (per column, averaged over all GT columns with schema-recall penalty):
  MRE  -- mean relative error of MLE-estimated distribution parameters
  NLL  -- normalised negative log-likelihood (exp scale so higher = worse)
  KS   -- Kolmogorov-Smirnov statistic against the fitted GT distribution
  FA   -- fraction of agreement (1 - KS), convenience complement

Missing-column penalty: columns in GT that are absent in the generated data
receive worst-case scores (MRE=1.0, NLL=0, KS=1.0, FA=0.0).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.data_level.distributions import (
    cdf_func,
    estimate_params,
    log_pdf,
    max_density_point,
)


# ---------------------------------------------------------------------------
# Single-column metrics
# ---------------------------------------------------------------------------

def _mre(pred_params: Dict[str, float], gt_params: Dict[str, float]) -> float:
    """
    Mean Relative Error between predicted and ground-truth distribution params.

    Only numeric (float-valued) parameters that exist in both dicts are compared.
    Returns 1.0 (worst case) when no common parameters exist.
    """
    errors: List[float] = []
    for key, gt_val in gt_params.items():
        if key not in pred_params:
            continue
        pred_val = pred_params[key]
        denom = abs(gt_val) if gt_val != 0.0 else 1.0
        errors.append(abs(pred_val - gt_val) / denom)
    return float(np.mean(errors)) if errors else 1.0


def _nll(data: np.ndarray, family: str, gt_params: Dict[str, float]) -> float:
    """
    Normalised NLL = exp((1/n) * sum(log f(x_i) - log f(x*)))

    x* is the mode / maximum-density point of the GT distribution.
    Lower is better (1.0 = perfect agreement with GT density).
    Returns 0.0 on failure (worst-case penalised).
    """
    try:
        arr = data[np.isfinite(data)]
        if len(arr) == 0:
            return 0.0
        x_star = max_density_point(family, gt_params)
        log_fi = log_pdf(arr, family, gt_params)
        log_f_star_val = log_pdf(np.array([x_star]), family, gt_params)[0]
        if not np.isfinite(log_f_star_val):
            return 0.0
        mean_diff = float(np.mean(log_fi[np.isfinite(log_fi)] - log_f_star_val))
        return float(np.exp(mean_diff))
    except Exception:
        return 0.0


def _ks(data: np.ndarray, family: str, gt_params: Dict[str, float]) -> float:
    """
    KS statistic: scipy.stats.kstest against the GT theoretical CDF.
    Returns 1.0 (worst) on failure.
    """
    try:
        arr = data[np.isfinite(data)]
        if len(arr) < 2:
            return 1.0
        cdf = cdf_func(family, gt_params)
        stat, _ = stats.kstest(arr, cdf)
        return float(stat)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# GT distribution spec parsing
# ---------------------------------------------------------------------------

def _parse_gt_dist(spec: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, float]]]:
    """
    Parse a ground-truth distribution spec dict into (family, params).

    The spec format used in cases.jsonl:
      {"family": "normal", "params": {"mean": 7.0, "std": 1.3}}
    or with lognormal variance shorthand:
      {"family": "lognormal", "params": {"mean": 3.5, "variance": 1.2}}

    Returns None if the spec is malformed.
    """
    try:
        family: str = spec["family"].lower()
        raw_params: Dict[str, Any] = spec.get("params", {})
        params: Dict[str, float] = {}

        # Normalise common aliases
        for k, v in raw_params.items():
            params[k] = float(v)

        # lognormal: convert mean/variance -> mu/sigma if needed
        if family == "lognormal" and "mean" in params and "mu" not in params:
            # Treat ground-truth "mean" as log-space mu, "variance"/"std" as sigma
            params["mu"] = params.pop("mean")
            if "variance" in params:
                params["sigma"] = float(np.sqrt(params.pop("variance")))
            elif "std" in params:
                params["sigma"] = params.pop("std")
            else:
                params["sigma"] = 1.0

        # exponential: alias lambda -> rate
        if family == "exponential" and "lambda" in params and "rate" not in params:
            params["rate"] = params.pop("lambda")

        return family, params
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Column-level evaluator
# ---------------------------------------------------------------------------

def evaluate_column(
    data: np.ndarray,
    gt_spec: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute MRE, NLL, KS, FA for a single column's generated data vs its GT spec.

    Parameters
    ----------
    data    : 1-D array of generated values (may contain NaN/Inf which are dropped)
    gt_spec : ground-truth distribution spec dict (family + params)

    Returns
    -------
    dict with keys: mre, nll, ks, fa
    Worst-case values on any failure: mre=1.0, nll=0.0, ks=1.0, fa=0.0
    """
    parsed = _parse_gt_dist(gt_spec)
    if parsed is None:
        return {"mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0}

    family, gt_params = parsed
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        return {"mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0}

    try:
        pred_params = estimate_params(arr, family)
    except Exception:
        return {"mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0}

    mre = _mre(pred_params, gt_params)
    nll = _nll(arr, family, gt_params)
    ks = _ks(arr, family, gt_params)

    return {
        "mre": min(mre, 1.0),
        "nll": max(nll, 0.0),
        "ks": min(ks, 1.0),
        "fa": max(1.0 - ks, 0.0),
    }


# ---------------------------------------------------------------------------
# Case-level evaluator (across all GT columns)
# ---------------------------------------------------------------------------

WORST_CASE = {"mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0}


def evaluate_data(
    dataframes: Dict[str, pd.DataFrame],
    gt_distributions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate all columns specified in gt_distributions against generated DataFrames.

    Parameters
    ----------
    dataframes       : {table_name: DataFrame} produced by the pipeline
    gt_distributions : {TABLE.column: dist_spec} from the ground-truth JSONL

    Returns
    -------
    dict with:
      column_scores  -- per-column breakdown
      mre, nll, ks, fa  -- macro averages (with missing-column penalty)
      n_evaluated    -- number of GT columns found in data
      n_missing      -- number of GT columns absent in generated data
    """
    column_scores: Dict[str, Dict[str, float]] = {}

    for col_key, gt_spec in gt_distributions.items():
        # col_key may be "TABLE.column" or "TABLE.column (label)" for filtered specs
        base_key = col_key.split(" (")[0]  # strip "(label)" suffixes
        parts = base_key.split(".", 1)
        if len(parts) != 2:
            column_scores[col_key] = dict(WORST_CASE)
            continue

        table_name, col_name = parts[0], parts[1]
        df = dataframes.get(table_name) or dataframes.get(table_name.lower())

        if df is None or col_name not in df.columns:
            column_scores[col_key] = dict(WORST_CASE)
            continue

        data = df[col_name].dropna().to_numpy(dtype=float, na_value=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            column_scores[col_key] = evaluate_column(data, gt_spec)

    if not column_scores:
        return {
            "column_scores": {},
            "mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0,
            "n_evaluated": 0, "n_missing": 0,
        }

    scores_list = list(column_scores.values())
    n_missing = sum(1 for s in scores_list if s["ks"] == 1.0 and s["mre"] == 1.0 and s["nll"] == 0.0)
    n_evaluated = len(scores_list) - n_missing

    return {
        "column_scores": column_scores,
        "mre": float(np.mean([s["mre"] for s in scores_list])),
        "nll": float(np.mean([s["nll"] for s in scores_list])),
        "ks": float(np.mean([s["ks"] for s in scores_list])),
        "fa": float(np.mean([s["fa"] for s in scores_list])),
        "n_evaluated": n_evaluated,
        "n_missing": n_missing,
    }

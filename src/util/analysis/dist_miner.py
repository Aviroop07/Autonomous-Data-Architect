"""
Distribution miner for ScribbleDB.

Given a 1-D array of observed values, tries a battery of parametric families,
fits each via MLE (or MOM where MLE is unavailable), and ranks them by AIC.

Usage
-----
    from src.util.analysis.dist_miner import mine_column_distribution

    result = mine_column_distribution(data)
    print(result.best.family, result.best.params)
    print(result.summary_table())

Design
------
* AIC (Akaike Information Criterion) is the primary ranking metric.
  AIC = 2k - 2*ln(L), where k = number of free parameters.
  Lower AIC = better trade-off between fit and complexity.

* BIC (Bayesian IC) is reported as a secondary metric.
  BIC = k*ln(n) - 2*ln(L).

* KS p-value is used as a reject filter: families with p < REJECT_THRESHOLD
  are marked invalid (poor fit) even if their AIC is low.

* Discrete vs continuous is auto-detected:
  - >= 95% integer-valued AND <= 500 unique values -> discrete
  - Override with force_discrete=True/False

* For bounded columns (e.g. 0.0-1.0), Beta is tried only when
  all values fall in (0, 1).

* Missing / infinite values are silently dropped before fitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class DistributionFit:
    """Result of fitting one distribution family to data."""
    family: str
    params: Dict[str, float]
    n_params: int
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    valid: bool = True          # False if fit failed or KS rejects at 5%

    def as_ground_truth_spec(self) -> Dict:
        """Return the spec dict used in ground_truth_distributions JSONL."""
        return {"family": self.family, "params": dict(self.params)}


@dataclass
class MiningResult:
    """Full output of mine_column_distribution."""
    best: DistributionFit
    candidates: List[DistributionFit]   # all families tried, sorted by AIC
    n_samples: int
    is_discrete: bool

    def summary_table(self) -> str:
        """Return a human-readable ranking table (for inspection)."""
        header = f"{'Family':<16} {'AIC':>10} {'BIC':>10} {'KS':>8} {'KS-p':>8} {'Valid':>6}"
        sep = "-" * 60
        rows = [header, sep]
        for f in self.candidates:
            mark = "*" if f is self.best else " "
            rows.append(
                f"{mark}{f.family:<15} {f.aic:>10.2f} {f.bic:>10.2f} "
                f"{f.ks_statistic:>8.4f} {f.ks_pvalue:>8.4f} {'YES' if f.valid else 'NO':>6}"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REJECT_KS_PVALUE = 0.05     # reject families with KS p-value below this
MIN_SAMPLES = 8             # minimum samples to attempt fitting
DISCRETE_INTEGER_FRAC = 0.95
DISCRETE_MAX_UNIQUE = 500


# ---------------------------------------------------------------------------
# Family catalogue
# ---------------------------------------------------------------------------

# Each entry: (name, n_free_params, needs_positive, needs_unit_interval, discrete)
_CONTINUOUS_FAMILIES: List[Tuple[str, int, bool, bool]] = [
    ("normal",      2, False, False),
    ("uniform",     2, False, False),
    ("lognormal",   2, True,  False),
    ("exponential", 1, True,  False),
    ("gamma",       2, True,  False),
    ("beta",        2, False, True),   # unit-interval only
]

_DISCRETE_FAMILIES: List[Tuple[str, int]] = [
    ("poisson", 1),
    ("zipf",    1),
]


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _clean(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _is_discrete(arr: np.ndarray) -> bool:
    if len(arr) == 0:
        return False
    integer_frac = np.mean(arr == np.round(arr))
    return bool(integer_frac >= DISCRETE_INTEGER_FRAC and
                len(np.unique(arr)) <= DISCRETE_MAX_UNIQUE)


def _log_likelihood(arr: np.ndarray, family: str, params: Dict[str, float]) -> float:
    """
    Compute sum of log-pdf/log-pmf for each point in arr.
    Returns -inf if any point has zero probability under the model.
    """
    from src.evaluation.data_level.distributions import log_pdf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll_arr = log_pdf(arr, family, params)
    finite_mask = np.isfinite(ll_arr)
    if not np.all(finite_mask):
        # Points outside support: assign a very small log-prob rather than
        # -inf so that a mostly-good fit is not completely discarded.
        # Penalty: log(1e-10) per out-of-support point.
        n_bad = np.sum(~finite_mask)
        ll = float(np.sum(ll_arr[finite_mask])) + n_bad * np.log(1e-10)
    else:
        ll = float(np.sum(ll_arr))
    return ll if np.isfinite(ll) else -1e18


def _ks_test(arr: np.ndarray, family: str, params: Dict[str, float]) -> Tuple[float, float]:
    """Return (statistic, pvalue) from the KS test."""
    from src.evaluation.data_level.distributions import cdf_func
    try:
        cdf = cdf_func(family, params)
        stat, pvalue = stats.kstest(arr, cdf)
        return float(stat), float(pvalue)
    except Exception:
        return 1.0, 0.0


# ---------------------------------------------------------------------------
# MLE / MOM parameter estimators
# ---------------------------------------------------------------------------

def _fit_family(arr: np.ndarray, family: str) -> Optional[Dict[str, float]]:
    """
    Return MLE / MOM parameter estimates for the given family.
    Returns None if the family is inapplicable to this data.
    """
    from src.evaluation.data_level.distributions import estimate_params
    try:
        params = estimate_params(arr, family)
        # Sanity checks
        if family == "beta":
            a, b = params.get("alpha", 0), params.get("beta", 0)
            if a <= 0 or b <= 0:
                return None
        if family in ("lognormal", "exponential", "gamma", "zipf", "poisson"):
            if np.any(arr <= 0) and family in ("lognormal", "zipf"):
                return None
        return params
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mine_column_distribution(
    data,
    max_candidates: int = 5,
    force_discrete: Optional[bool] = None,
    reject_threshold: float = REJECT_KS_PVALUE,
) -> MiningResult:
    """
    Find the best-fitting univariate distribution for a column of data.

    Parameters
    ----------
    data             : array-like of numeric values (NaN/Inf are dropped)
    max_candidates   : how many top candidates to keep in the result
    force_discrete   : override automatic discrete detection
    reject_threshold : KS p-value below which a fit is marked invalid

    Returns
    -------
    MiningResult with .best (DistributionFit) and .candidates (list, sorted by AIC)

    Raises
    ------
    ValueError if fewer than MIN_SAMPLES finite values are available
    """
    arr = _clean(np.asarray(data, dtype=float))
    n = len(arr)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"mine_column_distribution: need >= {MIN_SAMPLES} finite samples, got {n}"
        )

    is_discrete = force_discrete if force_discrete is not None else _is_discrete(arr)

    all_positive = bool(np.all(arr > 0))
    all_unit = bool(np.all((arr > 0) & (arr < 1)))

    candidates: List[DistributionFit] = []

    # ---- Continuous families ----
    for fam, k, needs_pos, needs_unit in _CONTINUOUS_FAMILIES:
        if needs_pos and not all_positive:
            continue
        if needs_unit and not all_unit:
            continue

        params = _fit_family(arr, fam)
        if params is None:
            continue

        ll = _log_likelihood(arr, fam, params)
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        ks_stat, ks_p = _ks_test(arr, fam, params)
        valid = ks_p >= reject_threshold

        candidates.append(DistributionFit(
            family=fam,
            params=params,
            n_params=k,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            valid=valid,
        ))

    # ---- Discrete families (only when data looks discrete) ----
    if is_discrete:
        int_arr = np.round(arr).astype(int)
        pos_arr = int_arr[int_arr >= 1].astype(float)

        for fam, k in _DISCRETE_FAMILIES:
            target = pos_arr if fam == "zipf" else arr
            if len(target) < MIN_SAMPLES:
                continue

            params = _fit_family(target, fam)
            if params is None:
                continue

            ll = _log_likelihood(target, fam, params)
            aic = 2 * k - 2 * ll
            bic = k * np.log(len(target)) - 2 * ll
            ks_stat, ks_p = _ks_test(target, fam, params)
            valid = ks_p >= reject_threshold

            candidates.append(DistributionFit(
                family=fam,
                params=params,
                n_params=k,
                log_likelihood=ll,
                aic=aic,
                bic=bic,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                valid=valid,
            ))

    if not candidates:
        raise ValueError("mine_column_distribution: no family could be fitted to this data.")

    # Sort: valid first, then by AIC ascending
    candidates.sort(key=lambda f: (not f.valid, f.aic))

    # Pick best: first valid candidate; fall back to lowest AIC if all invalid
    best = next((f for f in candidates if f.valid), candidates[0])

    return MiningResult(
        best=best,
        candidates=candidates[:max_candidates],
        n_samples=n,
        is_discrete=is_discrete,
    )


# ---------------------------------------------------------------------------
# Batch mining over a DataFrame
# ---------------------------------------------------------------------------

def mine_dataframe(
    df,
    numeric_only: bool = True,
    min_non_null_frac: float = 0.2,
    **kwargs,
) -> Dict[str, MiningResult]:
    """
    Mine best-fit distributions for every applicable column in a DataFrame.

    Parameters
    ----------
    df                : pandas DataFrame
    numeric_only      : skip non-numeric columns
    min_non_null_frac : skip columns where non-null fraction < this threshold
    **kwargs          : forwarded to mine_column_distribution

    Returns
    -------
    dict mapping column_name -> MiningResult (only successful columns included)
    """
    import pandas as pd

    results: Dict[str, MiningResult] = {}
    for col in df.columns:
        series = df[col]
        if numeric_only and not pd.api.types.is_numeric_dtype(series):
            continue
        data = series.dropna().to_numpy(dtype=float, na_value=np.nan)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
        non_null_frac = len(data) / max(len(series), 1)
        if non_null_frac < min_non_null_frac:
            continue
        try:
            results[col] = mine_column_distribution(data, **kwargs)
        except ValueError:
            pass
    return results


def mine_schema_distributions(
    dataframes: Dict[str, Any],
    **kwargs,
) -> Dict[str, Dict[str, MiningResult]]:
    """
    Mine distributions for all tables in a dict of DataFrames.

    Returns
    -------
    dict mapping table_name -> {column_name -> MiningResult}
    """
    return {
        table: mine_dataframe(df, **kwargs)
        for table, df in dataframes.items()
    }


def to_ground_truth_spec(
    mining_results: Dict[str, Dict[str, MiningResult]],
) -> Dict[str, Dict]:
    """
    Convert nested mining results to ground_truth_distributions format:
    {"TABLE.column": {"family": ..., "params": {...}}, ...}
    """
    spec: Dict[str, Dict] = {}
    for table, cols in mining_results.items():
        for col, result in cols.items():
            spec[f"{table}.{col}"] = result.best.as_ground_truth_spec()
    return spec

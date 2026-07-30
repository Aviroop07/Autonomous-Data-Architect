"""
Data-level evaluation metrics for ScribbleDB.

Metrics (per column, averaged over all GT columns with schema-recall penalty):
  MRE  -- mean relative error of MLE-estimated distribution parameters
  NLL  -- normalised negative log-likelihood (exp scale so higher = worse)
  DISTANCE -- statistical distance from the stated distribution, reported with
              the KIND used: Kolmogorov-Smirnov for continuous and ordered
              discrete families, total variation for categoricals, where a
              cumulative distribution would need an ordering nominal labels
              do not have. Both are in [0, 1] and 0 is a perfect match.

Missing-column penalty: columns in GT that are absent in the generated data
receive worst-case scores (MRE=1.0, NLL=0, DISTANCE=1.0).

FA was removed: it was defined as 1 - KS, so it carried no information KS did
not already carry, and reporting both invited reading one number as two.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.data_level.distributions import (
    canon_label,
    categorical_log_pmf,
    categorical_pmf,
    cdf_func,
    estimate_params,
    log_pdf,
    max_density_point,
    total_variation_distance,
)

# KS as scipy computes it assumes a CONTINUOUS reference CDF. For a discrete
# family the empirical CDF only steps at the support points, so kstest's
# statistic is inflated -- measured: data drawn from zipf(a=1.5) scored against
# zipf(a=1.5) returned exactly 1.0, the worst possible value, on a perfect
# sample. These families get the discrete analogue of the statistic instead.
_DISCRETE_FAMILIES = frozenset({"poisson", "zipf", "categorical"})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-column metrics
# ---------------------------------------------------------------------------


def _source_param(family: str, key: str) -> str:
    """Undo _parse_gt_dist's renaming, so a tier can be looked up.

    Tiers are recorded against the vocabulary cases.jsonl is authored in
    (mean/variance/lambda/min/max), while scoring happens in the vocabulary
    distributions.py computes with (mu/sigma/rate/low/high). Without this the
    tier lookup silently misses and every parameter falls back to `point`,
    which is exactly the behaviour the tiers exist to stop.
    """
    if key.startswith("p_"):
        return "weights"
    if family == "lognormal":
        return {"mu": "mean", "sigma": "variance"}.get(key, key)
    if family == "exponential":
        return {"rate": "lambda"}.get(key, key)
    if family == "uniform":
        return {"low": "min", "high": "max"}.get(key, key)
    return key


def param_tiers(spec: Dict[str, Any]) -> Dict[str, str]:
    """How each parameter of this ground-truth spec may legitimately be scored.

    A spec with no `scoring` block is treated as entirely `point`, so datasets
    predating the annotation score exactly as before.
    """
    scoring = spec.get("scoring") or {}
    out: Dict[str, str] = {}
    for name, entry in scoring.items():
        if isinstance(entry, dict) and isinstance(entry.get("tier"), str):
            out[name] = entry["tier"]
    return out


def _mre(
    pred_params: Dict[str, float],
    gt_params: Dict[str, float],
    scoreable: Optional[Callable[[str], bool]] = None,
) -> Optional[float]:
    """
    Mean Relative Error between predicted and ground-truth distribution params.

    Only numeric parameters present in both dicts are compared, and only those
    the specification actually PINS. A parameter the prose never states is a
    free variable -- comparing it to a point target marks the pipeline wrong
    for a choice it was entitled to make.

    Returns None, not 1.0, when nothing is comparable. A vacuous perfect or
    vacuous worst score is indistinguishable from a real one once averaged,
    which is the same trap the IC-Recall and CSR guards exist to close.
    """
    errors: List[float] = []
    for key, gt_val in gt_params.items():
        if key not in pred_params:
            continue
        if scoreable is not None and not scoreable(key):
            continue
        pred_val = pred_params[key]
        denom = abs(gt_val) if gt_val != 0.0 else 1.0
        errors.append(abs(pred_val - gt_val) / denom)
    return float(np.mean(errors)) if errors else None


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
    Supremum distance between the empirical and ground-truth CDFs.

    Continuous families go through scipy.stats.kstest. Discrete ones must not:
    kstest's statistic assumes a continuous reference CDF, and on a discrete
    support it is inflated to the point of being useless -- a sample drawn from
    zipf(a=1.5) scored against zipf(a=1.5) returned exactly 1.0, the worst
    possible value. For those families the statistic is computed directly as
    max|F_emp(x) - F_gt(x)| over the observed support, which is the right
    discrete analogue: still a supremum CDF distance, still 0 for a perfect fit,
    but evaluated only where the step function actually steps.

    Returns 1.0 (worst) on failure.
    """
    try:
        arr = data[np.isfinite(data)]
        if len(arr) < 2:
            return 1.0
        cdf = cdf_func(family, gt_params)

        if family.lower() in _DISCRETE_FAMILIES:
            ordered = np.sort(arr)
            support = np.unique(ordered)
            # Empirical CDF at each support point, evaluated from the right so
            # the point's own mass is included -- matching a step CDF.
            emp = np.searchsorted(ordered, support, side="right") / float(len(ordered))
            # cdf_func is typed as scalar-in/scalar-out but every implementation
            # is a vectorised scipy CDF or an array-aware closure.
            theo = np.asarray(cdf(support), dtype=float)  # type: ignore[arg-type]
            return float(np.max(np.abs(emp - theo)))

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

    Translates the vocabulary cases.jsonl is authored in into the one
    distributions.py computes with. The two had drifted apart, and because
    every failure here becomes a worst-case score rather than an error, the
    drift was invisible: this function returned None for EVERY entry in
    cases.jsonl, so evaluate_column reported mre=1.0/distance=1.0 even on data
    drawn exactly from the ground-truth distribution. Two causes, both fixed
    here -- the family lived under "distribution", not "family", and uniform's
    min/max and categorical's weights were never translated to the low/high and
    p_<label> keys the density and CDF functions read.

    Accepted spec shapes:
      {"distribution": "normal",      "params": {"mean": 7.0, "std": 1.3}}
      {"family":       "normal",      "params": {"mean": 7.0, "std": 1.3}}
      {"distribution": "lognormal",   "params": {"mean": 3.5, "variance": 1.2}}
      {"distribution": "uniform",     "params": {"min": 0, "max": 10}}
      {"distribution": "categorical", "params": {"weights": {"A": 0.7, "B": 0.3}}}

    Returns None if the spec is malformed, having logged why -- silence here is
    indistinguishable from a genuinely terrible score.
    """
    try:
        raw_family = spec.get("distribution", spec.get("family"))
        if not isinstance(raw_family, str):
            raise KeyError(
                "spec names no distribution family under 'distribution' or 'family'"
            )
        family: str = raw_family.lower()
        raw_params: Dict[str, Any] = spec.get("params", {}) or {}
        params: Dict[str, float] = {}

        # categorical carries a nested label->probability mapping rather than
        # scalars, so it is translated before the scalar coercion below.
        if family == "categorical":
            weights = raw_params.get("weights")
            if isinstance(weights, dict):
                for label, weight in weights.items():
                    params[f"p_{canon_label(label)}"] = float(weight)
            else:
                for k, v in raw_params.items():
                    params[str(k)] = float(v)
            if not params:
                raise ValueError("categorical spec carries no weights")
            return family, params

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

        # uniform: alias min/max -> low/high
        if family == "uniform":
            if "min" in params and "low" not in params:
                params["low"] = params.pop("min")
            if "max" in params and "high" not in params:
                params["high"] = params.pop("max")

        return family, params
    except Exception as exc:
        logger.warning(
            "[data_eval] unparseable ground-truth distribution spec %r: %s: %s "
            "-- this column will score worst-case (mre=1.0, distance=1.0).",
            spec,
            type(exc).__name__,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Column-level evaluator
# ---------------------------------------------------------------------------


def evaluate_column(
    data: np.ndarray,
    gt_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute MRE, NLL, KS, FA for a single column's generated data vs its GT spec.

    Parameters
    ----------
    data    : 1-D array of generated values (may contain NaN/Inf which are dropped)
    gt_spec : ground-truth distribution spec dict (family + params)

    Returns
    -------
    dict with keys: mre, nll, ks
    Worst-case values on any failure: mre=1.0, nll=0.0, distance=1.0
    """
    parsed = _parse_gt_dist(gt_spec)
    if parsed is None:
        return dict(WORST_CASE)

    family, gt_params = parsed

    # A categorical is scored by LABEL, never coerced through float(). That
    # coercion is what made nominal categories -- BRONZE/SILVER/GOLD, LOW/HIGH,
    # roughly three quarters of the categorical ground truth -- impossible to
    # score at all, since float("BRONZE") raises and the failure became a
    # worst-case number. It also routes to total variation distance rather than
    # KS, because KS is a supremum over a CUMULATIVE distribution and nominal
    # labels have no ordering to accumulate along.
    tiers = param_tiers(gt_spec)

    def _is_point(key: str) -> bool:
        return tiers.get(_source_param(family, key), "point") == "point"

    # KS and NLL compare the data against the WHOLE ground-truth distribution,
    # so they are point tests on every parameter at once. If any parameter of
    # this column is free, the distribution being tested against is partly one
    # the specification never asked for, and the distance is not a fidelity
    # measure -- it is reported as unscoreable rather than as a bad score.
    all_pinned = all(_is_point(k) for k in gt_params)

    if family == "categorical":
        return _evaluate_categorical(data, gt_params, _is_point, all_pinned)

    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        return dict(WORST_CASE)

    try:
        pred_params = estimate_params(arr, family)
    except Exception:
        return dict(WORST_CASE)

    mre = _mre(pred_params, gt_params, _is_point)
    out: Dict[str, Any] = {
        "mre": None if mre is None else min(mre, 1.0),
        "n_pinned": sum(1 for k in gt_params if _is_point(k)),
        "n_free": sum(1 for k in gt_params if not _is_point(k)),
    }
    if all_pinned:
        out["nll"] = max(_nll(arr, family, gt_params), 0.0)
        out["distance"] = min(_ks(arr, family, gt_params), 1.0)
        out["distance_kind"] = "ks"
    else:
        out["nll"] = None
        out["distance"] = None
        out["distance_kind"] = "unscoreable_free_params"
    out.update(_score_bands(pred_params, family, gt_spec))
    return out


def _score_bands(
    pred_params: Dict[str, float], family: str, gt_spec: Dict[str, Any]
) -> Dict[str, Any]:
    """Plausibility, not fidelity: did an estimated free parameter land in band?

    Reported with its own denominator. Folded into the fidelity average, a
    case would score well merely for being under-specified, which rewards
    vague specifications -- the opposite of what this benchmark measures.
    """
    bands = gt_spec.get("free_params") or {}
    if not bands:
        return {}
    applicable = satisfied = 0
    for key, value in pred_params.items():
        band = bands.get(_source_param(family, key))
        if not isinstance(band, dict):
            continue
        lo, hi = band.get("min"), band.get("max")
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            continue
        # A lognormal's spread is stored as a variance but estimated as sigma.
        estimated = value * value if (family == "lognormal" and key == "sigma") else value
        applicable += 1
        satisfied += 1 if lo <= estimated <= hi else 0
    if not applicable:
        return {}
    return {"band_applicable": applicable, "band_satisfied": satisfied}


def _evaluate_categorical(
    data: Any,
    gt_params: Dict[str, float],
    is_point: Optional[Callable[[str], bool]] = None,
    all_pinned: bool = True,
) -> Dict[str, Any]:
    """MRE, NLL and TVD for a categorical of any label type."""
    observed = categorical_pmf(data)
    if not observed:
        return dict(WORST_CASE)

    expected = {k[2:]: v for k, v in gt_params.items() if k.startswith("p_")}
    if not expected:
        return dict(WORST_CASE)

    # MRE over the probabilities themselves, matched by label. Comparing the
    # estimated pmf to the stated pmf IS the parameter comparison here -- a
    # categorical has no parameters other than its category probabilities.
    pred_params = {f"p_{k}": v for k, v in observed.items()}
    mre = _mre(pred_params, {f"p_{k}": v for k, v in expected.items()}, is_point)

    # NLL by label, normalised against the most probable category, mirroring the
    # continuous case where it is normalised against the mode.
    log_p = categorical_log_pmf(data, gt_params)
    finite = log_p[np.isfinite(log_p)]
    if len(finite) == 0:
        nll = 0.0
    else:
        best = max(expected.values())
        nll = (
            float(np.exp(float(np.mean(finite)) - math.log(best))) if best > 0 else 0.0
        )

    return {
        "mre": None if mre is None else min(mre, 1.0),
        "nll": max(nll, 0.0) if all_pinned else None,
        "distance": total_variation_distance(observed, expected)
        if all_pinned
        else None,
        "distance_kind": "tvd" if all_pinned else "unscoreable_free_params",
    }


# ---------------------------------------------------------------------------
# Case-level evaluator (across all GT columns)
# ---------------------------------------------------------------------------

WORST_CASE: Dict[str, Any] = {
    "mre": 1.0,
    "nll": 0.0,
    "distance": 1.0,
    "distance_kind": "none",
}


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
      mre, nll, distance  -- macro averages (with missing-column penalty)
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
            "mre": 1.0,
            "nll": 0.0,
            "distance": 1.0,
            "n_evaluated": 0,
            "n_missing": 0,
        }

    scores_list = list(column_scores.values())
    n_missing = sum(
        1
        for s in scores_list
        if s["distance"] == 1.0 and s["mre"] == 1.0 and s["nll"] == 0.0
    )
    n_evaluated = len(scores_list) - n_missing

    def _avg(key: str) -> Optional[float]:
        """Average over the columns this metric can legitimately score.

        Columns whose parameters the specification never states carry None,
        and they are EXCLUDED rather than defaulted. Substituting a value --
        1.0 or 0.0 -- would let a benchmark's score move with how vague its
        specifications are, in whichever direction the default happened to
        favour. The denominator is reported alongside so the coverage is
        visible rather than implied.
        """
        vals = [s[key] for s in scores_list if isinstance(s.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None

    band_applicable = sum(int(s.get("band_applicable", 0) or 0) for s in scores_list)
    band_satisfied = sum(int(s.get("band_satisfied", 0) or 0) for s in scores_list)

    return {
        "column_scores": column_scores,
        "mre": _avg("mre"),
        "nll": _avg("nll"),
        "distance": _avg("distance"),
        # Fidelity's own denominator: how many columns the specification pinned
        # tightly enough to score at all.
        "n_scoreable": sum(
            1 for s in scores_list if isinstance(s.get("distance"), (int, float))
        ),
        "n_unscoreable": sum(
            1
            for s in scores_list
            if s.get("distance_kind") == "unscoreable_free_params"
        ),
        # Plausibility is a SEPARATE tier with a separate denominator. Folded
        # into fidelity, a case would score well for being under-specified.
        "plausibility_rate": (
            band_satisfied / band_applicable if band_applicable else None
        ),
        "n_band_applicable": band_applicable,
        # Which distance each column used, so a mean over mixed kinds is never
        # reported as if it were homogeneous.
        "distance_kinds": {
            kind: sum(1 for s in scores_list if s.get("distance_kind") == kind)
            for kind in sorted({s.get("distance_kind", "none") for s in scores_list})
        },
        "n_evaluated": n_evaluated,
        "n_missing": n_missing,
    }

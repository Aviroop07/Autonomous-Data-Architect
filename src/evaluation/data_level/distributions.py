"""
Distribution parameter estimation and evaluation utilities.

Supported families:
  normal, lognormal, poisson, zipf, categorical,
  exponential, uniform, beta, gamma

All public functions are stateless and operate on numpy arrays.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(data: np.ndarray) -> np.ndarray:
    """Remove NaN / Inf values and return a 1-D float64 array."""
    arr = np.asarray(data, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _require_positive(data: np.ndarray, name: str) -> np.ndarray:
    """Return only strictly positive elements; warn if any were dropped."""
    pos = data[data > 0]
    if len(pos) < len(data):
        warnings.warn(
            f"[distributions] {name}: {len(data) - len(pos)} non-positive value(s) "
            "removed before fitting.",
            stacklevel=3,
        )
    return pos


def _require_unit_interval(data: np.ndarray, name: str) -> np.ndarray:
    """Return elements in (0, 1); warn if any were clipped."""
    unit = data[(data > 0) & (data < 1)]
    if len(unit) < len(data):
        warnings.warn(
            f"[distributions] {name}: {len(data) - len(unit)} out-of-(0,1) value(s) "
            "removed before fitting.",
            stacklevel=3,
        )
    return unit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_params(data: np.ndarray, family: str) -> Dict[str, float]:
    """
    Estimate distribution parameters from *data* via MLE (or MOM where MLE
    is not available in scipy).

    Parameters
    ----------
    data   : array-like of observations (NaN/Inf are silently dropped)
    family : one of the supported family names (case-insensitive)

    Returns
    -------
    dict of parameter names -> float values

    Raises
    ------
    ValueError  if family is unrecognised or data is empty after cleaning
    """
    arr = _clean(data)
    fam = family.lower()

    if len(arr) == 0:
        raise ValueError(f"estimate_params: no finite data for family '{family}'")

    if fam == "normal":
        loc, scale = stats.norm.fit(arr)
        return {"mean": float(loc), "std": float(max(scale, 1e-9))}

    if fam == "lognormal":
        pos = _require_positive(arr, "lognormal")
        if len(pos) == 0:
            return {"mu": 0.0, "sigma": 1.0}
        log_pos = np.log(pos)
        mu = float(np.mean(log_pos))
        sigma = float(max(np.std(log_pos, ddof=1) if len(log_pos) > 1 else 1.0, 1e-9))
        return {"mu": mu, "sigma": sigma}

    if fam == "poisson":
        # Poisson MLE: lambda = sample mean (rounded to non-negative)
        lam = float(max(np.mean(arr), 1e-9))
        return {"lambda": lam}

    if fam == "zipf":
        # scipy.zipf_gen does not implement .fit(); use MOM estimate instead.
        # For Zipf(a), E[X] = zeta(a-1)/zeta(a).  We approximate by noting
        # that for integer data with mean m, a simple bound is a ~ 1 + 1/log(m)
        # which is tractable and always > 1.
        pos = _require_positive(arr, "zipf")
        pos_int = np.round(pos).astype(int)
        pos_int = pos_int[pos_int >= 1]
        if len(pos_int) == 0:
            return {"a": 2.0}
        mean_val = float(np.mean(pos_int))
        if mean_val <= 1.0:
            a_est = 2.0
        else:
            a_est = float(max(1.0 + 1.0 / np.log(mean_val), 1.01))
        return {"a": a_est}

    if fam == "categorical":
        # Return empirical probability mass function
        unique, counts = np.unique(arr, return_counts=True)
        total = counts.sum()
        pmf = {str(v): float(c / total) for v, c in zip(unique, counts)}
        # Flatten into a serialisable dict; keys become "p_<value>"
        result: Dict[str, float] = {f"p_{k}": v for k, v in pmf.items()}
        result["n_categories"] = float(len(unique))
        return result

    if fam == "exponential":
        pos = _require_positive(arr, "exponential")
        if len(pos) == 0:
            return {"rate": 1.0}
        # MLE for exponential: rate = 1 / mean
        rate = float(1.0 / max(np.mean(pos), 1e-9))
        return {"rate": rate}

    if fam == "uniform":
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            hi = lo + 1.0
        return {"low": lo, "high": hi}

    if fam == "beta":
        unit = _require_unit_interval(arr, "beta")
        if len(unit) < 2:
            return {"alpha": 1.0, "beta": 1.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _beta_a, _beta_b, *_ = stats.beta.fit(unit, floc=0, fscale=1)
        a_val = float(_beta_a)  # type: ignore[arg-type]
        b_val = float(_beta_b)  # type: ignore[arg-type]
        return {"alpha": max(a_val, 1e-3), "beta": max(b_val, 1e-3)}

    if fam == "gamma":
        pos = _require_positive(arr, "gamma")
        if len(pos) < 2:
            return {"shape": 1.0, "rate": 1.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, loc, scale = stats.gamma.fit(pos, floc=0)
        rate = float(1.0 / max(scale, 1e-9))
        return {"shape": float(max(shape, 1e-3)), "rate": rate}

    raise ValueError(f"estimate_params: unsupported family '{family}'")


def max_density_point(family: str, params: Dict[str, float]) -> float:
    """
    Return x* = argmax f(x) for the specified distribution.

    For discrete distributions (poisson, zipf, categorical) the mode is
    returned as a float.
    """
    fam = family.lower()

    if fam == "normal":
        return float(params["mean"])

    if fam == "lognormal":
        # Mode of LogNormal = exp(mu - sigma^2)
        return float(np.exp(params["mu"] - params["sigma"] ** 2))

    if fam == "poisson":
        # Mode = floor(lambda) for lambda > 0
        lam = params["lambda"]
        return float(max(int(lam), 0))

    if fam == "zipf":
        return 1.0  # Zipf mode is always 1 (minimum value)

    if fam == "categorical":
        # Return category with highest probability
        cat_probs = {k: v for k, v in params.items() if k.startswith("p_")}
        if not cat_probs:
            return 0.0
        best = max(cat_probs, key=lambda k: cat_probs[k])
        try:
            return float(best[2:])  # strip "p_" prefix
        except ValueError:
            return 0.0

    if fam == "exponential":
        return 0.0  # Exponential mode = 0

    if fam == "uniform":
        # Uniform is flat; return midpoint as representative
        return float((params["low"] + params["high"]) / 2.0)

    if fam == "beta":
        a, b = params["alpha"], params["beta"]
        if a > 1 and b > 1:
            return float((a - 1) / (a + b - 2))
        if a <= 1 and b > 1:
            return 0.0
        if a > 1 and b <= 1:
            return 1.0
        return 0.5  # a <= 1 and b <= 1: bimodal, return midpoint

    if fam == "gamma":
        shape = params["shape"]
        rate = params["rate"]
        if shape >= 1:
            return float((shape - 1) / rate)
        return 0.0

    raise ValueError(f"max_density_point: unsupported family '{family}'")


def log_pdf(x: np.ndarray, family: str, params: Dict[str, float]) -> np.ndarray:
    """
    Compute log f(x_i) for each point in x under the specified distribution.

    Returns an array of the same shape as x.  Points outside the support
    receive -inf (which corresponds to probability 0).
    """
    arr = np.asarray(x, dtype=float)
    fam = family.lower()

    if fam == "normal":
        return stats.norm.logpdf(arr, loc=params["mean"], scale=params["std"])

    if fam == "lognormal":
        # SciPy lognormal: shape=sigma, scale=exp(mu), loc=0
        s = params["sigma"]
        scale = np.exp(params["mu"])
        return stats.lognorm.logpdf(arr, s=s, scale=scale, loc=0)

    if fam == "poisson":
        return stats.poisson.logpmf(np.round(arr).astype(int), mu=params["lambda"])

    if fam == "zipf":
        a = params["a"]
        int_arr = np.round(arr).astype(int)
        return stats.zipf.logpmf(int_arr, a=a)

    if fam == "categorical":
        cat_probs = {k[2:]: v for k, v in params.items() if k.startswith("p_")}
        out = np.full(arr.shape, -np.inf)
        for i, xi in enumerate(arr):
            key = str(float(xi))
            if key in cat_probs and cat_probs[key] > 0:
                out[i] = np.log(cat_probs[key])
        return out

    if fam == "exponential":
        rate = params["rate"]
        return stats.expon.logpdf(arr, scale=1.0 / rate)

    if fam == "uniform":
        return stats.uniform.logpdf(arr, loc=params["low"], scale=params["high"] - params["low"])

    if fam == "beta":
        return stats.beta.logpdf(arr, a=params["alpha"], b=params["beta"], loc=0, scale=1)

    if fam == "gamma":
        shape = params["shape"]
        rate = params["rate"]
        return stats.gamma.logpdf(arr, a=shape, scale=1.0 / rate, loc=0)

    raise ValueError(f"log_pdf: unsupported family '{family}'")


def cdf_func(family: str, params: Dict[str, float]) -> Callable[[float], float]:
    """
    Return a CDF callable suitable for use with scipy.stats.kstest.

    The returned function accepts a scalar or array and returns CDF values.
    """
    fam = family.lower()

    if fam == "normal":
        dist = stats.norm(loc=params["mean"], scale=params["std"])
        return dist.cdf  # type: ignore[return-value]

    if fam == "lognormal":
        s = params["sigma"]
        scale = np.exp(params["mu"])
        dist = stats.lognorm(s=s, scale=scale, loc=0)
        return dist.cdf  # type: ignore[return-value]

    if fam == "poisson":
        dist = stats.poisson(mu=params["lambda"])
        return dist.cdf  # type: ignore[return-value]

    if fam == "zipf":
        dist = stats.zipf(a=params["a"])
        return dist.cdf  # type: ignore[return-value]

    if fam == "categorical":
        # Build empirical step CDF over sorted category values
        cat_probs = {float(k[2:]): v for k, v in params.items() if k.startswith("p_")}
        sorted_vals = sorted(cat_probs.keys())
        cumulative = []
        running = 0.0
        for v in sorted_vals:
            running += cat_probs[v]
            cumulative.append((v, running))

        def _categorical_cdf(x: Any) -> Any:
            x_arr = np.asarray(x, dtype=float)
            result = np.zeros_like(x_arr)
            for val, cum in cumulative:
                result[x_arr >= val] = cum
            return result

        return _categorical_cdf

    if fam == "exponential":
        dist = stats.expon(scale=1.0 / params["rate"])
        return dist.cdf  # type: ignore[return-value]

    if fam == "uniform":
        dist = stats.uniform(loc=params["low"], scale=params["high"] - params["low"])
        return dist.cdf  # type: ignore[return-value]

    if fam == "beta":
        dist = stats.beta(a=params["alpha"], b=params["beta"], loc=0, scale=1)
        return dist.cdf  # type: ignore[return-value]

    if fam == "gamma":
        dist = stats.gamma(a=params["shape"], scale=1.0 / params["rate"], loc=0)
        return dist.cdf  # type: ignore[return-value]

    raise ValueError(f"cdf_func: unsupported family '{family}'")

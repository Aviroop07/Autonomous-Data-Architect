"""The benchmark's ground truth must be readable by the evaluator that scores it.

These two vocabularies had silently drifted apart. cases.jsonl names the family
under "distribution"; _parse_gt_dist read spec["family"]. It therefore returned
None for EVERY entry, and because a parse failure becomes a worst-case score
rather than an error, evaluate_column reported mre=1.0/ks=1.0/fa=0.0 on data
drawn exactly from the ground-truth distribution. Nothing surfaced it, because
the data-level metrics cannot run until Stage 4 exists -- so the numbers would
have looked computed the moment Stage 4 landed, while being floor values.

The self-consistency property below is the check that would have caught it, and
is the one worth keeping: sample from a ground-truth distribution, score that
sample against its own ground truth, and the result must be good. Any future
drift between the authored vocabulary and the computed one breaks it.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.evaluation.data_level.data_eval import _parse_gt_dist, evaluate_column

CASES = pathlib.Path("dataset/handcrafted/cases.jsonl")

# Families whose ground-truth parameters name a sampleable numeric distribution.
# Nominal categorical (string labels) is deliberately excluded -- see
# test_nominal_categorical_is_a_known_unsupported_shape.
_SAMPLEABLE = {
    "normal",
    "lognormal",
    "uniform",
    "poisson",
    "exponential",
    "zipf",
    "categorical",
}


def _load_cases() -> List[Dict[str, Any]]:
    if not CASES.exists():
        pytest.skip(f"{CASES} not present")
    return [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gt_dists() -> List[tuple[int, str, Dict[str, Any]]]:
    out: List[tuple[int, str, Dict[str, Any]]] = []
    for case in _load_cases():
        for column, spec in (case.get("ground_truth_distributions") or {}).items():
            out.append((case["id"], column, spec))
    return out


def _is_nominal_categorical(spec: Dict[str, Any]) -> bool:
    """True for a categorical whose labels are not numeric."""
    if spec.get("distribution", spec.get("family")) != "categorical":
        return False
    weights = (spec.get("params") or {}).get("weights") or {}
    for label in weights:
        try:
            float(label)
        except TypeError, ValueError:
            return True
    return False


def _sample(spec: Dict[str, Any], n: int = 8000) -> Optional[np.ndarray]:
    """Draw from the ground truth itself, in the authored parameterisation."""
    rng = np.random.default_rng(12345)
    family = spec.get("distribution", spec.get("family"))
    p = spec.get("params") or {}
    if family == "normal":
        return rng.normal(p["mean"], p["std"], n)
    if family == "lognormal":
        return rng.lognormal(p["mean"], float(np.sqrt(p["variance"])), n)
    if family == "uniform":
        return rng.uniform(p["min"], p["max"], n)
    if family == "poisson":
        return rng.poisson(p["lambda"], n).astype(float)
    if family == "exponential":
        return rng.exponential(1.0 / p["lambda"], n)
    if family == "zipf":
        return rng.zipf(p["a"], n).astype(float)
    if family == "categorical":
        weights = p["weights"]
        labels = list(weights)
        probs = np.array([float(weights[k]) for k in labels], dtype=float)
        probs /= probs.sum()
        drawn = rng.choice(labels, size=n, p=probs)
        return np.array([float(x) for x in drawn])
    return None


def test_the_dataset_is_present_and_non_trivial() -> None:
    """Guards the parametrisation -- an empty dataset would pass vacuously."""
    assert len(_gt_dists()) > 50


@pytest.mark.parametrize(
    "case_id,column,spec",
    _gt_dists(),
    ids=lambda v: str(v) if isinstance(v, int) else None,
)
def test_every_ground_truth_spec_parses(
    case_id: int, column: str, spec: Dict[str, Any]
) -> None:
    """A spec the evaluator cannot parse scores worst-case, silently."""
    parsed = _parse_gt_dist(spec)
    assert parsed is not None, f"case {case_id} column {column}: unparseable {spec}"
    family, params = parsed
    assert family
    assert params, "parsed to an empty parameter set"


@pytest.mark.parametrize(
    "case_id,column,spec",
    [t for t in _gt_dists() if not _is_nominal_categorical(t[2])],
    ids=lambda v: str(v) if isinstance(v, int) else None,
)
def test_perfect_data_does_not_score_worst_case(
    case_id: int, column: str, spec: Dict[str, Any]
) -> None:
    """Self-consistency: data drawn from the GT, scored against that same GT.

    Deliberately a floor check rather than an accuracy check -- the point is to
    catch a broken contract, not to pin estimator precision, and asserting tight
    bounds here would make the test fail for statistical reasons instead.
    """
    family = spec.get("distribution", spec.get("family"))
    if family not in _SAMPLEABLE:
        pytest.skip(f"no sampler for family {family!r}")
    data = _sample(spec)
    assert data is not None

    result = evaluate_column(data, spec)
    assert not (result["ks"] >= 0.99 and result["mre"] >= 0.99), (
        f"case {case_id} column {column} ({family}) scores worst-case on data drawn "
        f"from its own ground truth: {result}"
    )
    assert result["ks"] < 0.5, f"case {case_id} {column}: ks={result['ks']:.3f}"


def test_nominal_categorical_is_a_known_unsupported_shape() -> None:
    """Records a real gap rather than hiding it.

    distributions.py's categorical path does float(k[2:]) on category keys and
    str(float(xi)) on the data, so it can only score categories that are
    numbers. Five ground-truth columns are nominal (loyalty tiers, severities,
    plan names), and those cannot be scored at the data level as things stand --
    KS over a nominal variable is not meaningful anyway, so this needs a decision
    about the metric, not a coercion.
    """
    nominal = [t for t in _gt_dists() if _is_nominal_categorical(t[2])]
    assert nominal, "expected the known nominal-categorical columns to still exist"
    # They must at least PARSE, so the gap is confined to scoring.
    for case_id, column, spec in nominal:
        assert _parse_gt_dist(spec) is not None, f"case {case_id} {column}"

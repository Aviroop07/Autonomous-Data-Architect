"""Total variation distance, so nominal categoricals can be scored at all.

Before this, a categorical was scored by coercing both the data and the category
keys through float(). That works for a 1-5 rating and fails outright for
BRONZE/SILVER/GOLD -- and 132 of the 189 categorical ground truths in the
benchmark are nominal like that, roughly a quarter of every distribution in it.
The failure was invisible, because an exception inside evaluate_column becomes a
worst-case score rather than an error.

KS was also the wrong statistic for them, not merely an awkward one: it is a
supremum over a CUMULATIVE distribution, and accumulating requires an ordering
that nominal labels do not have. Deciding BRONZE sorts below SILVER is arbitrary
and changes the number. TVD needs no ordering.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from src.evaluation.data_level.data_eval import evaluate_column
from src.evaluation.data_level.distributions import (
    categorical_pmf,
    total_variation_distance,
)

TIERS: Dict[str, float] = {
    "BRONZE": 0.50,
    "SILVER": 0.30,
    "GOLD": 0.15,
    "PLATINUM": 0.05,
}


def _spec(weights: Dict[str, float]) -> Dict[str, Any]:
    return {"distribution": "categorical", "params": {"weights": dict(weights)}}


def _sample(weights: Dict[str, float], n: int = 6000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels: List[str] = list(weights)
    probs = np.array([weights[k] for k in labels], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(labels, size=n, p=probs)


# ---------------------------------------------------------------------------
# the distance itself
# ---------------------------------------------------------------------------


def test_tvd_is_zero_for_identical_distributions() -> None:
    assert total_variation_distance(TIERS, TIERS) == pytest.approx(0.0)


def test_tvd_is_one_for_disjoint_supports() -> None:
    """No overlap at all is the maximum possible disagreement."""
    other = {"COPPER": 0.6, "TIN": 0.4}
    assert total_variation_distance(TIERS, other) == pytest.approx(1.0)


def test_tvd_is_symmetric() -> None:
    other = {"BRONZE": 0.25, "SILVER": 0.25, "GOLD": 0.25, "PLATINUM": 0.25}
    assert total_variation_distance(TIERS, other) == pytest.approx(
        total_variation_distance(other, TIERS)
    )


def test_tvd_matches_the_hand_computed_value() -> None:
    """Half the summed absolute difference, checked by hand rather than trusted.

    |0.5-0.25| + |0.3-0.25| + |0.15-0.25| + |0.05-0.25|
      = 0.25 + 0.05 + 0.10 + 0.20 = 0.60, halved = 0.30
    """
    uniform = {k: 0.25 for k in TIERS}
    assert total_variation_distance(TIERS, uniform) == pytest.approx(0.30)


def test_tvd_needs_no_ordering_so_relabelling_cannot_change_it() -> None:
    """The property KS cannot have. Renaming categories consistently is a
    relabelling of a nominal variable and must leave the distance alone."""
    renamed_gt = {"AAA": 0.50, "BBB": 0.30, "CCC": 0.15, "DDD": 0.05}
    obs = {"BRONZE": 0.4, "SILVER": 0.4, "GOLD": 0.15, "PLATINUM": 0.05}
    renamed_obs = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.15, "DDD": 0.05}
    assert total_variation_distance(obs, TIERS) == pytest.approx(
        total_variation_distance(renamed_obs, renamed_gt)
    )


def test_tvd_stays_in_the_unit_interval_for_unnormalised_input() -> None:
    assert 0.0 <= total_variation_distance({"A": 5.0}, {"B": 5.0}) <= 1.0


# ---------------------------------------------------------------------------
# the empirical pmf
# ---------------------------------------------------------------------------


def test_categorical_pmf_handles_string_labels() -> None:
    pmf = categorical_pmf(np.array(["A", "A", "B"], dtype=object))
    assert pmf == pytest.approx({"A": 2 / 3, "B": 1 / 3})


def test_categorical_pmf_canonicalises_numeric_labels() -> None:
    """1, 1.0 and "1" are the same category, so they must not split."""
    pmf = categorical_pmf(np.array([1, 1.0, "1"], dtype=object))
    assert pmf == pytest.approx({"1": 1.0})


def test_categorical_pmf_ignores_missing_values() -> None:
    """NaN is absence, not a category -- and checking for it must not call
    float() on a string, which is what broke the numeric path."""
    pmf = categorical_pmf(np.array(["A", float("nan"), None, "B"], dtype=object))
    assert pmf == pytest.approx({"A": 0.5, "B": 0.5})


# ---------------------------------------------------------------------------
# end to end through evaluate_column
# ---------------------------------------------------------------------------


def test_nominal_data_drawn_from_its_own_ground_truth_scores_well() -> None:
    """The case that was previously unscorable: string labels, no numbers."""
    result = evaluate_column(_sample(TIERS), _spec(TIERS))
    assert result["distance_kind"] == "tvd"
    assert result["distance"] < 0.05, result
    assert result["mre"] < 0.2, result


def test_a_wrong_category_mix_is_penalised() -> None:
    uniform = {k: 0.25 for k in TIERS}
    result = evaluate_column(_sample(uniform), _spec(TIERS))
    assert result["distance"] > 0.2, result


def test_entirely_unseen_categories_score_worst_case_distance() -> None:
    data = np.array(["UNOBTAINIUM"] * 500, dtype=object)
    result = evaluate_column(data, _spec(TIERS))
    assert result["distance"] == pytest.approx(1.0)


def test_a_numeric_coded_categorical_also_uses_tvd() -> None:
    """A 1-5 rating is still a categorical; it should not silently take the
    continuous path just because its labels happen to parse as numbers."""
    weights = {"1": 0.5, "2": 0.3, "3": 0.2}
    rng = np.random.default_rng(3)
    data = rng.choice([1.0, 2.0, 3.0], size=4000, p=[0.5, 0.3, 0.2])
    result = evaluate_column(data, _spec(weights))
    assert result["distance_kind"] == "tvd"
    assert result["distance"] < 0.05, result


def test_a_continuous_family_still_reports_ks() -> None:
    """TVD is for categoricals only; nothing else should have changed."""
    rng = np.random.default_rng(11)
    data = rng.normal(50.0, 5.0, 2000)
    result = evaluate_column(
        data, {"distribution": "normal", "params": {"mean": 50.0, "std": 5.0}}
    )
    assert result["distance_kind"] == "ks"
    assert result["distance"] < 0.1, result


def test_every_nominal_categorical_in_the_benchmark_is_now_scorable() -> None:
    """The point of the exercise, measured against the real dataset.

    Each nominal categorical ground truth is sampled from itself and scored. None
    may come back at worst-case, which is what all of them did before.
    """
    import json
    import pathlib

    cases_path = pathlib.Path("dataset/handcrafted/cases.jsonl")
    if not cases_path.exists():
        pytest.skip(f"{cases_path} not present")

    nominal: List[Dict[str, Any]] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        for spec in (case.get("ground_truth_distributions") or {}).values():
            if spec.get("distribution") != "categorical":
                continue
            weights = (spec.get("params") or {}).get("weights") or {}
            if any(not _is_numeric(k) for k in weights):
                nominal.append(spec)

    assert len(nominal) > 50, (
        f"expected many nominal categoricals, found {len(nominal)}"
    )

    failures: List[str] = []
    for spec in nominal:
        weights = spec["params"]["weights"]
        result = evaluate_column(_sample(weights, n=3000), spec)
        if result["distance"] >= 0.5 or result["distance_kind"] != "tvd":
            failures.append(f"{sorted(weights)}: {result}")
    assert not failures, "nominal categoricals still unscorable:\n" + "\n".join(
        failures[:5]
    )


def _is_numeric(label: Any) -> bool:
    try:
        float(label)
    except TypeError, ValueError:
        return False
    return True

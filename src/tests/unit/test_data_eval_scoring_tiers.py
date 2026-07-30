"""A parameter the specification never states must not be scored as if it did.

19% of this benchmark's distribution parameters are not recoverable from the
prose in any form. Scoring those against a point target asserts a degree of
freedom the input never removed, so the pipeline is marked wrong for a choice
it was entitled to make. `scoring.tier` records what each parameter entitles a
metric to check; these tests pin that the evaluator honours it.

The controls matter as much as the assertions here. A tier system that quietly
did nothing would still let every existing test pass, so each case that MUST
change behaviour is paired with one that must NOT.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.data_level.data_eval import (
    _mre,
    _source_param,
    evaluate_column,
    param_tiers,
)


class TestSourceParam:
    """Tiers are authored in cases.jsonl's vocabulary, scored in another."""

    @pytest.mark.parametrize(
        "family,scored_key,authored_key",
        [
            ("lognormal", "mu", "mean"),
            ("lognormal", "sigma", "variance"),
            ("exponential", "rate", "lambda"),
            ("uniform", "low", "min"),
            ("uniform", "high", "max"),
            ("categorical", "p_GOLD", "weights"),
            ("normal", "mean", "mean"),
            ("normal", "std", "std"),
            ("poisson", "lambda", "lambda"),
            ("zipf", "a", "a"),
        ],
    )
    def test_round_trips_every_renamed_parameter(
        self, family: str, scored_key: str, authored_key: str
    ) -> None:
        # A miss here is silent: the tier lookup fails, the parameter falls back
        # to `point`, and the whole mechanism stops working while every test
        # still passes.
        assert _source_param(family, scored_key) == authored_key


class TestParamTiers:
    def test_absent_scoring_block_yields_no_tiers(self) -> None:
        # Backward compatibility: a dataset predating the annotation must score
        # exactly as it did before.
        assert param_tiers({"distribution": "normal", "params": {"mean": 1.0}}) == {}

    def test_reads_declared_tiers(self) -> None:
        spec = {
            "distribution": "normal",
            "params": {"mean": 1.0, "std": 2.0},
            "scoring": {"mean": {"tier": "point"}, "std": {"tier": "band"}},
        }
        assert param_tiers(spec) == {"mean": "point", "std": "band"}

    def test_ignores_malformed_entries(self) -> None:
        spec = {"scoring": {"mean": "point", "std": {"tier": "band"}, "a": {}}}
        assert param_tiers(spec) == {"std": "band"}


class TestMreHonoursTiers:
    def test_unfiltered_behaviour_is_unchanged(self) -> None:
        mre = _mre({"mean": 12.0}, {"mean": 10.0})
        assert mre == pytest.approx(0.2)

    def test_free_parameter_is_excluded_from_the_average(self) -> None:
        pred = {"mean": 10.0, "std": 99.0}
        gt = {"mean": 10.0, "std": 1.0}
        # With std free, a wildly wrong std must not drag the score down.
        assert _mre(pred, gt, lambda k: k == "mean") == pytest.approx(0.0)
        # Control: scored as a point target, the same data is penalised.
        unfiltered = _mre(pred, gt)
        assert unfiltered is not None and unfiltered > 1.0

    def test_returns_none_rather_than_a_vacuous_score(self) -> None:
        # Not 1.0. A vacuous worst score is indistinguishable from a real one
        # once averaged across columns.
        assert _mre({"mean": 1.0}, {"mean": 1.0}, lambda _k: False) is None
        assert _mre({}, {}) is None


class TestEvaluateColumnHonoursTiers:
    @staticmethod
    def _normal_sample() -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(10.0, 2.0, 4000)

    def test_all_pinned_scores_exactly_as_before(self) -> None:
        spec = {"distribution": "normal", "params": {"mean": 10.0, "std": 2.0}}
        out = evaluate_column(self._normal_sample(), spec)
        assert out["distance_kind"] == "ks"
        assert out["distance"] is not None and out["distance"] < 0.1
        assert out["mre"] is not None and out["mre"] < 0.1

    def test_a_free_parameter_makes_the_distance_unscoreable(self) -> None:
        # KS tests the data against the WHOLE distribution, so with a free
        # parameter it is partly testing a target the prose never set. That is
        # reported as unscoreable, never as a bad score.
        spec = {
            "distribution": "normal",
            "params": {"mean": 10.0, "std": 2.0},
            "scoring": {"mean": {"tier": "point"}, "std": {"tier": "band"}},
        }
        out = evaluate_column(self._normal_sample(), spec)
        assert out["distance"] is None
        assert out["nll"] is None
        assert out["distance_kind"] == "unscoreable_free_params"
        assert out["n_pinned"] == 1 and out["n_free"] == 1

    def test_a_wrong_free_parameter_does_not_hurt_the_pinned_score(self) -> None:
        spec = {
            "distribution": "normal",
            "params": {"mean": 10.0, "std": 0.01},
            "scoring": {"mean": {"tier": "point"}, "std": {"tier": "band"}},
        }
        out = evaluate_column(self._normal_sample(), spec)
        # std is off by 200x but free, so MRE stays clean on the stated mean.
        assert out["mre"] is not None and out["mre"] < 0.1

    def test_band_containment_is_reported_with_its_own_denominator(self) -> None:
        spec = {
            "distribution": "normal",
            "params": {"mean": 10.0, "std": 2.0},
            "scoring": {"mean": {"tier": "point"}, "std": {"tier": "band"}},
            "free_params": {"std": {"min": 1.0, "max": 3.0}},
        }
        out = evaluate_column(self._normal_sample(), spec)
        assert out["band_applicable"] == 1
        assert out["band_satisfied"] == 1

    def test_band_containment_can_fail(self) -> None:
        # The band must be able to reject, or it is decoration.
        spec = {
            "distribution": "normal",
            "params": {"mean": 10.0, "std": 2.0},
            "scoring": {"mean": {"tier": "point"}, "std": {"tier": "band"}},
            "free_params": {"std": {"min": 100.0, "max": 200.0}},
        }
        out = evaluate_column(self._normal_sample(), spec)
        assert out["band_applicable"] == 1
        assert out["band_satisfied"] == 0

    def test_lognormal_band_compares_variance_against_variance(self) -> None:
        # The spread is STORED as a variance and ESTIMATED as a sigma. Comparing
        # sigma to a variance band silently mis-scores every lognormal, which is
        # 127 of this benchmark's 691 distributions.
        rng = np.random.default_rng(1)
        data = rng.lognormal(1.0, 0.8, 4000)  # sigma 0.8 -> variance 0.64
        spec = {
            "distribution": "lognormal",
            "params": {"mean": 1.0, "variance": 0.64},
            "scoring": {"mean": {"tier": "point"}, "variance": {"tier": "band"}},
            "free_params": {"variance": {"min": 0.5, "max": 0.8}},
        }
        out = evaluate_column(data, spec)
        assert out["band_applicable"] == 1
        assert out["band_satisfied"] == 1, "sigma was compared without squaring"

    def test_fully_free_column_yields_no_fidelity_number(self) -> None:
        spec = {
            "distribution": "normal",
            "params": {"mean": 10.0, "std": 2.0},
            "scoring": {
                "mean": {"tier": "unscoreable"},
                "std": {"tier": "unscoreable"},
            },
        }
        out = evaluate_column(self._normal_sample(), spec)
        assert out["mre"] is None
        assert out["distance"] is None
        assert out["n_pinned"] == 0

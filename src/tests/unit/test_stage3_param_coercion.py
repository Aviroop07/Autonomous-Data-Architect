"""Malformed distribution parameters must fail as ValidationError, never
TypeError.

`parameters` is typed dict[str, float | list[str] | list[float]], so
{"mean": [10], "std_dev": [2]} satisfies the annotation and only failed later
at `v["std_dev"] <= 0`. That raises TypeError, which pydantic v2 does NOT wrap
into a ValidationError -- it propagated out of model construction, out of
get_response, and killed the whole shard's extraction rather than becoming
retryable feedback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pipeline.stage3.models.cross_shard import DistributionConstraint
from src.util.constraint_model.relation.nodes import BaseTable


def _make(family: str, parameters: dict) -> DistributionConstraint:
    return DistributionConstraint(
        fact_references=[1],
        on=BaseTable(name="T"),
        column="c",
        family=family,  # type: ignore[arg-type]
        parameters=parameters,
    )


class TestListValuedScalarsDoNotEscapePydantic:
    def test_single_element_lists_are_unwrapped(self):
        """An unambiguous serialisation slip. Recovering it here is free;
        rejecting it would spend a retry round on a value already in hand."""
        c = _make("GAUSSIAN", {"mean": [10], "std_dev": [2]})
        assert c.parameters["mean"] == 10.0
        assert c.parameters["std_dev"] == 2.0

    def test_multi_element_list_raises_validation_error_not_type_error(self):
        with pytest.raises(ValidationError):
            _make("GAUSSIAN", {"mean": [1, 2, 3], "std_dev": [2]})

    def test_string_parameter_raises_validation_error_not_type_error(self):
        with pytest.raises(ValidationError):
            _make("GAUSSIAN", {"mean": "not a number", "std_dev": 2})

    def test_numeric_string_is_accepted(self):
        c = _make("GAUSSIAN", {"mean": "10.5", "std_dev": "2"})
        assert c.parameters["mean"] == 10.5

    def test_as_number_rejects_a_boolean(self):
        """Reachable only by calling the helper directly: the field annotation
        (dict[str, float | ...]) makes pydantic coerce True -> 1.0 before the
        field_validator ever runs. The guard stays because _as_number is a
        general coercion helper, not solely a hook for this one field."""
        with pytest.raises(ValueError):
            DistributionConstraint._as_number(True, family="POISSON", key="lam")

    @pytest.mark.parametrize(
        "family,params",
        [
            ("LOG_NORMAL", {"mean": [1], "std_dev": [1]}),
            ("BETA", {"alpha": [2], "beta": [3]}),
            ("POISSON", {"lam": [4]}),
            ("UNIFORM", {"min_value": [0], "max_value": [9]}),
        ],
    )
    def test_every_scalar_family_is_covered(self, family, params):
        """Guards against a family being added to the Literal but omitted from
        _SCALAR_PARAMS, which would reopen the TypeError hole for it."""
        c = _make(family, params)
        assert all(isinstance(c.parameters[k], float) for k in params)


class TestCategoricalProbabilities:
    def test_non_list_probabilities_raise_validation_error(self):
        with pytest.raises(ValidationError):
            _make("CATEGORICAL", {"categories": ["a", "b"], "probabilities": 0.5})

    def test_non_numeric_probabilities_raise_validation_error(self):
        with pytest.raises(ValidationError):
            _make(
                "CATEGORICAL",
                {"categories": ["a", "b"], "probabilities": ["0.5", "0.5"]},
            )

    def test_valid_categorical_still_passes(self):
        c = _make(
            "CATEGORICAL", {"categories": ["a", "b"], "probabilities": [0.5, 0.5]}
        )
        assert c.parameters["categories"] == ["a", "b"]

    def test_probabilities_are_optional(self):
        c = _make("CATEGORICAL", {"categories": ["a"]})
        assert "probabilities" not in c.parameters


class TestExistingSemanticChecksStillHold:
    def test_non_positive_std_dev_still_rejected(self):
        with pytest.raises(ValidationError):
            _make("GAUSSIAN", {"mean": 10, "std_dev": 0})

    def test_uniform_bounds_still_ordered(self):
        with pytest.raises(ValidationError):
            _make("UNIFORM", {"min_value": 9, "max_value": 0})

    def test_missing_required_parameter_still_rejected(self):
        with pytest.raises(ValidationError):
            _make("GAUSSIAN", {"mean": 10})

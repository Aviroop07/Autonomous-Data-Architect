"""Tests for src/util/constraint_model/condition/cohesive.py."""

from __future__ import annotations

from src.util.constraint_model.condition.cohesive import (
    Correlated,
    Distributed,
    PairwiseCorrelation,
    StateSequence,
    StateTransition,
)


class TestDistributed:
    def test_valid_gaussian(self):
        d = Distributed(
            column="total", family="GAUSSIAN", parameters={"mean": 100, "std_dev": 10}
        )
        assert d._validate() == []

    def test_partial_parameters_are_valid(self):
        d = Distributed(column="total", family="GAUSSIAN", parameters={})
        assert d._validate() == []

    def test_negative_std_dev_rejected(self):
        d = Distributed(column="total", family="GAUSSIAN", parameters={"std_dev": -5})
        assert any("std_dev must be positive" in e for e in d._validate())

    def test_log_normal_zero_std_dev_rejected(self):
        d = Distributed(column="total", family="LOG_NORMAL", parameters={"std_dev": 0})
        assert any("std_dev must be positive" in e for e in d._validate())

    def test_beta_non_positive_alpha_rejected(self):
        d = Distributed(
            column="rate", family="BETA", parameters={"alpha": 0, "beta": 2}
        )
        assert any("alpha must be positive" in e for e in d._validate())

    def test_beta_valid(self):
        d = Distributed(
            column="rate", family="BETA", parameters={"alpha": 2, "beta": 3}
        )
        assert d._validate() == []

    def test_poisson_non_positive_lambda_rejected(self):
        d = Distributed(column="count", family="POISSON", parameters={"lam": -1})
        assert any("lam must be positive" in e for e in d._validate())

    def test_categorical_empty_categories_rejected(self):
        d = Distributed(
            column="status", family="CATEGORICAL", parameters={"categories": []}
        )
        assert any("non-empty list" in e for e in d._validate())

    def test_categorical_probabilities_length_mismatch_rejected(self):
        d = Distributed(
            column="status",
            family="CATEGORICAL",
            parameters={"categories": ["a", "b"], "probabilities": [1.0]},
        )
        assert any("length must match" in e for e in d._validate())

    def test_categorical_probabilities_not_summing_to_one_rejected(self):
        d = Distributed(
            column="status",
            family="CATEGORICAL",
            parameters={"categories": ["a", "b"], "probabilities": [0.5, 0.6]},
        )
        assert any("sum to 1.0" in e for e in d._validate())

    def test_categorical_negative_probability_rejected(self):
        d = Distributed(
            column="status",
            family="CATEGORICAL",
            parameters={"categories": ["a", "b"], "probabilities": [1.5, -0.5]},
        )
        assert any("cannot be negative" in e for e in d._validate())

    def test_categorical_valid(self):
        d = Distributed(
            column="status",
            family="CATEGORICAL",
            parameters={"categories": ["a", "b"], "probabilities": [0.4, 0.6]},
        )
        assert d._validate() == []

    def test_uniform_min_greater_than_max_rejected(self):
        d = Distributed(
            column="x", family="UNIFORM", parameters={"min_value": 10, "max_value": 1}
        )
        assert any("min_value must be <= max_value" in e for e in d._validate())

    def test_uniform_valid(self):
        d = Distributed(
            column="x", family="UNIFORM", parameters={"min_value": 1, "max_value": 10}
        )
        assert d._validate() == []

    def test_unknown_parameter_key_rejected(self):
        d = Distributed(
            column="total", family="GAUSSIAN", parameters={"nonsense_key": 1}
        )
        assert any("not recognized" in e for e in d._validate())

    def test_column_must_be_lower_snake(self):
        d = Distributed(column="Total", family="GAUSSIAN", parameters={})
        assert any("lower_snake_case" in e for e in d._validate())


class TestCorrelated:
    def test_valid_binary_correlation(self):
        c = Correlated(
            columns=["age", "income"],
            family="GAUSSIAN",
            pairwise=[PairwiseCorrelation(left="age", right="income", value=0.5)],
        )
        assert c._validate() == []

    def test_partial_pairwise_is_valid(self):
        c = Correlated(columns=["age", "income", "weight"], family="GAUSSIAN")
        assert c._validate() == []

    def test_duplicate_columns_rejected(self):
        c = Correlated(columns=["age", "age"], family="GAUSSIAN")
        assert any("duplicate" in e.lower() for e in c._validate())

    def test_pairwise_column_not_in_columns_rejected(self):
        c = Correlated(
            columns=["age", "income"],
            family="GAUSSIAN",
            pairwise=[PairwiseCorrelation(left="age", right="weight", value=0.5)],
        )
        assert any("not in Correlated.columns" in e for e in c._validate())

    def test_pairwise_value_out_of_range_rejected(self):
        c = Correlated(
            columns=["age", "income"],
            family="GAUSSIAN",
            pairwise=[PairwiseCorrelation(left="age", right="income", value=1.5)],
        )
        assert any("must be in [-1, 1]" in e for e in c._validate())

    def test_pairwise_self_correlation_rejected(self):
        c = Correlated(
            columns=["age", "income"],
            family="GAUSSIAN",
            pairwise=[PairwiseCorrelation(left="age", right="age", value=1.0)],
        )
        assert any("must be different columns" in e for e in c._validate())

    def test_shared_nu_for_student_t(self):
        c = Correlated(
            columns=["age", "income", "weight"],
            family="STUDENT_T",
            shared_parameters={"nu": 5.0},
        )
        assert c._validate() == []


class TestStateTransition:
    def test_self_loop_rejected(self):
        t = StateTransition(from_state="ready", to_state="ready")
        assert any("must differ" in e for e in t._validate())

    def test_valid_transition(self):
        t = StateTransition(from_state="ready", to_state="packed")
        assert t._validate() == []


class TestStateSequence:
    def test_valid_sequence(self):
        s = StateSequence(
            sequence_column="status",
            allowed_transitions=[
                StateTransition(from_state="ready", to_state="packed")
            ],
        )
        assert s._validate() == []

    def test_conflicting_allowed_and_forbidden_rejected(self):
        s = StateSequence(
            sequence_column="status",
            allowed_transitions=[
                StateTransition(from_state="ready", to_state="packed")
            ],
            forbidden_transitions=[
                StateTransition(from_state="ready", to_state="packed")
            ],
        )
        errors = s._validate()
        assert any("both allowed and forbidden" in e for e in errors)

    def test_sequence_column_must_be_lower_snake(self):
        s = StateSequence(sequence_column="Status")
        assert any(
            "sequence_column must be lower_snake_case" in e for e in s._validate()
        )

    def test_no_transitions_is_valid(self):
        s = StateSequence(sequence_column="status")
        assert s._validate() == []

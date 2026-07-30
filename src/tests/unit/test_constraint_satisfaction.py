"""Constraint Satisfaction Rate, the metric Stage 4's output cannot be scored
without.

Two kinds of test here. The hand-built cases pin each grammar form and each way
a check can be inconclusive. The self-consistency test at the end is the one that
would catch real drift: it builds rows that SATISFY a constraint drawn from the
actual benchmark grammar and demands CSR = 1.0, then violates a known share and
demands the rate fall by exactly that share. That is the same property that
caught the data-level evaluator silently scoring everything worst-case.

The recurring theme is that "violated" and "could not be checked" must never be
conflated -- a missing column, a null, or a dangling foreign key are all defects,
but they are not evidence that THIS rule was broken.
"""

from __future__ import annotations

from src.evaluation.data_level.constraint_satisfaction import (
    GeneratedData,
    constraint_satisfaction_rate,
    evaluate_constraint,
)


def _data(**tables) -> GeneratedData:
    return GeneratedData(tables)


class TestRange:
    def test_all_rows_within_range(self):
        c = {"type": "range", "table": "V", "column": "spo2", "min": 50, "max": 100}
        r = evaluate_constraint(c, _data(V=[{"spo2": 60}, {"spo2": 100}, {"spo2": 50}]))
        assert r.rate == 1.0
        assert r.applicable_rows == 3

    def test_bounds_are_inclusive(self):
        c = {"type": "range", "table": "V", "column": "x", "min": 1, "max": 3}
        r = evaluate_constraint(c, _data(V=[{"x": 1}, {"x": 3}]))
        assert r.violated_rows == 0

    def test_a_violation_is_counted(self):
        c = {"type": "range", "table": "V", "column": "x", "min": 0, "max": 10}
        r = evaluate_constraint(c, _data(V=[{"x": 5}, {"x": 11}, {"x": -1}, {"x": 0}]))
        assert r.violated_rows == 2
        assert r.rate == 0.5

    def test_an_open_bound_is_allowed(self):
        c = {"type": "range", "table": "V", "column": "x", "min": 0, "max": None}
        r = evaluate_constraint(c, _data(V=[{"x": 10**9}]))
        assert r.rate == 1.0


class TestComparisons:
    def test_column_against_literal(self):
        c = {"type": "gte", "table": "T", "column": "a", "value": 5}
        r = evaluate_constraint(c, _data(T=[{"a": 5}, {"a": 4}]))
        assert (r.satisfied_rows, r.violated_rows) == (1, 1)

    def test_column_against_another_column_same_table(self):
        """discharged_at >= admitted_at, the commonest cross-column shape."""
        c = {"type": "gte", "table": "E", "column": "out", "rhs_column": "in_"}
        r = evaluate_constraint(
            c, _data(E=[{"in_": 1, "out": 2}, {"in_": 5, "out": 3}])
        )
        assert (r.satisfied_rows, r.violated_rows) == (1, 1)

    def test_arithmetic_right_hand_side(self):
        """weal_size_mm >= saline_control_mm + 3."""
        c = {
            "type": "gte",
            "table": "T",
            "column": "weal",
            "rhs_expr": {"column": "saline", "op": "+", "value": 3},
        }
        r = evaluate_constraint(
            c, _data(T=[{"weal": 10, "saline": 5}, {"weal": 6, "saline": 5}])
        )
        assert (r.satisfied_rows, r.violated_rows) == (1, 1)

    def test_neq_and_lt(self):
        data = _data(T=[{"a": 1, "b": 1}, {"a": 1, "b": 2}])
        neq = evaluate_constraint(
            {"type": "neq", "table": "T", "column": "a", "rhs_column": "b"}, data
        )
        assert (neq.satisfied_rows, neq.violated_rows) == (1, 1)
        lt = evaluate_constraint(
            {"type": "lt", "table": "T", "column": "a", "value": 2}, data
        )
        assert lt.rate == 1.0


class TestJoins:
    def _joined(self):
        return _data(
            LINE=[
                {"code": "A", "charged": 10},
                {"code": "B", "charged": 99},
            ],
            CODE=[{"code": "A", "standard": 20}, {"code": "B", "standard": 20}],
        )

    def test_compare_against_a_joined_column(self):
        c = {
            "type": "lte",
            "table": "LINE",
            "column": "charged",
            "rhs_column": "standard",
            "rhs_table_ref": "CODE",
            "rhs_join": {"from": "LINE.code", "to": "CODE.code"},
        }
        r = evaluate_constraint(c, self._joined())
        assert (r.satisfied_rows, r.violated_rows) == (1, 1)

    def test_arithmetic_across_a_join(self):
        """charged <= standard * 2 -- both rows pass at 2x, one fails at 1x."""
        expr = {
            "column": "standard",
            "table_ref": "CODE",
            "join": {"from": "LINE.code", "to": "CODE.code"},
            "op": "*",
            "value": 2,
        }
        c = {"type": "lte", "table": "LINE", "column": "charged", "rhs_expr": expr}
        r = evaluate_constraint(c, self._joined())
        assert (r.satisfied_rows, r.violated_rows) == (1, 1)

    def test_a_dangling_foreign_key_is_unevaluable_not_violated(self):
        """Referential integrity is a different defect; blaming this rule for it
        would misattribute the failure."""
        data = _data(
            LINE=[{"code": "MISSING", "charged": 1}],
            CODE=[{"code": "A", "standard": 2}],
        )
        c = {
            "type": "lte",
            "table": "LINE",
            "column": "charged",
            "rhs_column": "standard",
            "rhs_table_ref": "CODE",
            "rhs_join": {"from": "LINE.code", "to": "CODE.code"},
        }
        r = evaluate_constraint(c, data)
        assert r.violated_rows == 0
        assert not r.is_evaluable


class TestIfThen:
    def _c(self):
        return {
            "type": "ifthen",
            "table": "ORD",
            "condition": {"type": "eq", "column": "controlled", "value": True},
            "result": {"type": "gte", "column": "grade", "value": 3},
        }

    def test_only_rows_matching_the_condition_are_tested(self):
        data = _data(
            ORD=[
                {"controlled": True, "grade": 5},  # applies, satisfied
                {"controlled": True, "grade": 1},  # applies, violated
                {"controlled": False, "grade": 1},  # vacuous
            ]
        )
        r = evaluate_constraint(self._c(), data)
        assert r.applicable_rows == 2
        assert r.vacuous_rows == 1
        assert r.rate == 0.5

    def test_a_never_triggered_constraint_has_no_rate(self):
        """The vacuity trap: 1.0 here would let a dataset that dodges every
        antecedent look perfect."""
        data = _data(ORD=[{"controlled": False, "grade": 0}] * 5)
        r = evaluate_constraint(self._c(), data)
        assert r.vacuous_rows == 5
        assert r.applicable_rows == 0
        assert r.rate is None
        assert r.is_evaluable

    def test_and_or_inside_a_condition(self):
        c = {
            "type": "ifthen",
            "table": "T",
            "condition": {
                "type": "or",
                "conditions": [
                    {"type": "lt", "column": "sbp", "value": 90},
                    {"type": "lt", "column": "spo2", "value": 92},
                ],
            },
            "result": {"type": "eq", "column": "escalated", "value": True},
        }
        data = _data(
            T=[
                {"sbp": 80, "spo2": 99, "escalated": True},
                {"sbp": 120, "spo2": 90, "escalated": False},
                {"sbp": 120, "spo2": 99, "escalated": False},
            ]
        )
        r = evaluate_constraint(c, data)
        assert (r.applicable_rows, r.violated_rows, r.vacuous_rows) == (2, 1, 1)


class TestUnevaluable:
    def test_a_missing_table_is_reported_not_scored(self):
        c = {"type": "range", "table": "ABSENT", "column": "x", "min": 0, "max": 1}
        r = evaluate_constraint(c, _data(OTHER=[{"x": 0}]))
        assert not r.is_evaluable
        assert "ABSENT" in (r.unevaluable_reason or "")

    def test_a_missing_column_is_not_a_violation(self):
        c = {"type": "range", "table": "T", "column": "ghost", "min": 0, "max": 1}
        r = evaluate_constraint(c, _data(T=[{"x": 0}]))
        assert r.violated_rows == 0
        assert not r.is_evaluable

    def test_nulls_are_skipped_rather_than_failed(self):
        """SQL semantics, and penalising a legitimately optional column would be
        the wrong default."""
        c = {"type": "gte", "table": "T", "column": "a", "value": 1}
        r = evaluate_constraint(c, _data(T=[{"a": None}, {"a": 5}]))
        assert r.applicable_rows == 1
        assert r.rate == 1.0

    def test_some_bad_rows_do_not_discard_the_checkable_ones(self):
        c = {"type": "gte", "table": "T", "column": "a", "value": 1}
        r = evaluate_constraint(c, _data(T=[{"a": None}, {"a": 5}, {"a": 0}]))
        assert r.applicable_rows == 2
        assert r.rate == 0.5


class TestAggregate:
    def test_csr_is_row_weighted_not_constraint_weighted(self):
        """A rule applying to 2 rows must not outvote one applying to 100."""
        data = _data(
            BIG=[{"x": 1}] * 100,  # all satisfy
            SMALL=[{"y": 9}, {"y": 9}],  # both violate
        )
        cs = [
            {"type": "lte", "table": "BIG", "column": "x", "value": 1},
            {"type": "lte", "table": "SMALL", "column": "y", "value": 1},
        ]
        rep = constraint_satisfaction_rate(cs, data)
        csr = rep.csr
        assert csr is not None
        assert csr == 100 / 102
        assert abs(csr - 0.5) > 0.4, "a mean of per-constraint rates would give 0.5"

    def test_vacuous_and_unevaluable_are_counted_separately(self):
        data = _data(T=[{"a": 1, "flag": False}])
        cs = [
            {"type": "range", "table": "GONE", "column": "a", "min": 0, "max": 1},
            {
                "type": "ifthen",
                "table": "T",
                "condition": {"type": "eq", "column": "flag", "value": True},
                "result": {"type": "gte", "column": "a", "value": 0},
            },
        ]
        rep = constraint_satisfaction_rate(cs, data)
        assert rep.n_unevaluable == 1
        assert rep.n_vacuous == 1
        assert rep.csr is None
        assert rep.as_dict()["n_constraints"] == 2

    def test_no_applicable_rows_yields_none_not_one(self):
        rep = constraint_satisfaction_rate([], _data(T=[{"a": 1}]))
        assert rep.csr is None


class TestSelfConsistency:
    """Build data that satisfies a constraint, then break a known share of it.

    The property that matters: the measured rate must equal the share actually
    left intact. A scorer that silently failed would pass every hand-built case
    above by returning 0.0 everywhere, and this is what catches that.
    """

    def _constraint(self):
        return {
            "type": "ifthen",
            "table": "ORDER_LINE",
            "condition": {"type": "gt", "column": "qty", "value": 0},
            "result": {
                "type": "lte",
                "column": "dispatched",
                "rhs_column": "qty",
            },
        }

    def test_data_built_to_satisfy_scores_one(self):
        rows = [{"qty": i, "dispatched": i} for i in range(1, 51)]
        rep = constraint_satisfaction_rate([self._constraint()], _data(ORDER_LINE=rows))
        assert rep.csr == 1.0
        assert rep.as_dict()["applicable_rows"] == 50

    def test_breaking_a_known_share_moves_the_rate_by_exactly_that_share(self):
        rows = [{"qty": i, "dispatched": i} for i in range(1, 51)]
        for row in rows[:10]:
            row["dispatched"] = row["qty"] + 1  # 10 of 50 now violate
        rep = constraint_satisfaction_rate([self._constraint()], _data(ORDER_LINE=rows))
        assert rep.csr == 0.8
        assert rep.as_dict()["violated_rows"] == 10

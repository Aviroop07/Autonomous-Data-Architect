"""Tests for src/util/constraint_model/conflicts/correlated.py."""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table
from src.util.constraint_model.condition.cohesive import Correlated, PairwiseCorrelation
from src.util.constraint_model.conflicts.correlated import check_correlated_conflicts
from src.util.constraint_model.conflicts.grouping import (
    annotate_populations,
    group_by_comparable_population,
)
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import BaseTable


def _schema(columns: list[str]) -> Schema:
    return Schema(
        tables=[
            Table(
                name="T",
                columns=[Column(name="id", data_type=DataType.INTEGER)]
                + [Column(name=c, data_type=DataType.FLOAT) for c in columns],
                primary_key=["id"],
            )
        ]
    )


def _corr(columns, family, fid, pairwise=None, shared_parameters=None) -> Constraint:
    return Constraint(
        relation=BaseTable(name="T"),
        condition=Correlated(
            columns=columns,
            family=family,
            pairwise=pairwise or [],
            shared_parameters=shared_parameters or {},
        ),
        fact_references=[fid],
    )


def _pw(left, right, value) -> PairwiseCorrelation:
    return PairwiseCorrelation(left=left, right=right, value=value)


def _cluster(constraints, schema):
    annotated, errors = annotate_populations(constraints, schema)
    assert errors == []
    clusters = group_by_comparable_population(annotated)
    assert len(clusters) == 1
    return clusters[0]


class TestValueMismatch:
    def test_disagreeing_pairwise_values(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.9)])
        c2 = _corr(["a", "b"], "GAUSSIAN", 2, pairwise=[_pw("a", "b", -0.5)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b"]))
        )
        assert unsupported == []
        assert len(conflicts) == 1
        assert conflicts[0].kind == "correlated_value_mismatch"

    def test_agreeing_pairwise_values_no_conflict(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.5)])
        c2 = _corr(
            ["a", "b"],
            "STUDENT_T",
            2,
            pairwise=[_pw("a", "b", 0.5)],
            shared_parameters={"nu": 10},
        )
        conflicts, _ = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b"]))
        )
        assert conflicts == []

    def test_left_right_order_does_not_matter(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.9)])
        c2 = _corr(["a", "b"], "GAUSSIAN", 2, pairwise=[_pw("b", "a", -0.5)])
        conflicts, _ = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b"]))
        )
        assert len(conflicts) == 1


class TestStudentTPrecondition:
    def test_nu_leq_2_with_overlapping_correlation_conflicts(self):
        c1 = _corr(["a", "b", "c"], "STUDENT_T", 1, shared_parameters={"nu": 1.5})
        c2 = _corr(["a", "b"], "GAUSSIAN", 2, pairwise=[_pw("a", "b", 0.5)])
        conflicts, _ = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b", "c"]))
        )
        assert any(c.kind == "correlated_precondition_violation" for c in conflicts)

    def test_nu_greater_than_2_no_precondition_conflict(self):
        c1 = _corr(["a", "b", "c"], "STUDENT_T", 1, shared_parameters={"nu": 5.0})
        c2 = _corr(["a", "b"], "GAUSSIAN", 2, pairwise=[_pw("a", "b", 0.5)])
        conflicts, _ = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b", "c"]))
        )
        assert not any(c.kind == "correlated_precondition_violation" for c in conflicts)

    def test_nu_leq_2_but_no_column_overlap_no_conflict(self):
        c1 = _corr(["a", "b"], "STUDENT_T", 1, shared_parameters={"nu": 1.0})
        c2 = _corr(["c", "d"], "GAUSSIAN", 2, pairwise=[_pw("c", "d", 0.5)])
        conflicts, _ = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b", "c", "d"]))
        )
        assert conflicts == []

    def test_nu_exactly_2_is_still_a_violation(self):
        # correlation is only defined for nu > 2 -- nu == 2 is the boundary, still invalid.
        c1 = _corr(["a", "b"], "STUDENT_T", 1, shared_parameters={"nu": 2.0})
        c2 = _corr(["a", "b"], "GAUSSIAN", 2, pairwise=[_pw("a", "b", 0.5)])
        conflicts, _ = check_correlated_conflicts(
            _cluster([c1, c2], _schema(["a", "b"]))
        )
        assert any(c.kind == "correlated_precondition_violation" for c in conflicts)


class TestChordalPdCompletion:
    def test_feasible_triangle_all_equal(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.5)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 0.5)])
        c3 = _corr(["a", "c"], "GAUSSIAN", 3, pairwise=[_pw("a", "c", 0.5)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3], _schema(["a", "b", "c"]))
        )
        assert conflicts == [] and unsupported == []

    def test_infeasible_triangle(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.9)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 0.9)])
        c3 = _corr(["a", "c"], "GAUSSIAN", 3, pairwise=[_pw("a", "c", -0.9)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3], _schema(["a", "b", "c"]))
        )
        assert unsupported == []
        assert len(conflicts) == 1
        assert conflicts[0].kind == "correlated_infeasible_matrix"
        assert set(conflicts[0].involved_fact_references) == {1, 2, 3}

    def test_extreme_values_near_boundary_still_pd(self):
        # a chain of 0.99 correlations across 4 columns -- individually
        # extreme but still jointly realizable (near-degenerate, not
        # actually infeasible).
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.99)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 0.99)])
        c3 = _corr(["a", "c"], "GAUSSIAN", 3, pairwise=[_pw("a", "c", 0.99)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3], _schema(["a", "b", "c"]))
        )
        assert conflicts == [] and unsupported == []

    def test_perfectly_degenerate_matrix_is_infeasible(self):
        # a=b=1.0, b=c=1.0, a=c=-1.0 is contradictory (a=b and b=c implies a=c=1, not -1)
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 1.0)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 1.0)])
        c3 = _corr(["a", "c"], "GAUSSIAN", 3, pairwise=[_pw("a", "c", -1.0)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3], _schema(["a", "b", "c"]))
        )
        assert len(conflicts) == 1
        assert conflicts[0].kind == "correlated_infeasible_matrix"

    def test_non_chordal_four_cycle_is_unsupported_not_a_false_verdict(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.5)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 0.5)])
        c3 = _corr(["c", "d"], "GAUSSIAN", 3, pairwise=[_pw("c", "d", 0.5)])
        c4 = _corr(["d", "a"], "GAUSSIAN", 4, pairwise=[_pw("d", "a", 0.5)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3, c4], _schema(["a", "b", "c", "d"]))
        )
        assert conflicts == []
        assert len(unsupported) == 1
        assert "not chordal" in unsupported[0]

    def test_non_chordal_becomes_chordal_and_checkable_once_a_chord_is_added(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.5)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 0.5)])
        c3 = _corr(["c", "d"], "GAUSSIAN", 3, pairwise=[_pw("c", "d", 0.5)])
        c4 = _corr(["d", "a"], "GAUSSIAN", 4, pairwise=[_pw("d", "a", 0.5)])
        c5 = _corr(
            ["a", "c"], "GAUSSIAN", 5, pairwise=[_pw("a", "c", 0.5)]
        )  # the chord
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3, c4, c5], _schema(["a", "b", "c", "d"]))
        )
        assert unsupported == []
        assert conflicts == []

    def test_five_column_chordal_network_with_one_infeasible_clique(self):
        # a-b-c form an infeasible triangle; d, e are independently, validly
        # correlated with a and b respectively but don't touch the broken
        # clique -- only ONE conflict should surface, isolated correctly.
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.9)])
        c2 = _corr(["b", "c"], "GAUSSIAN", 2, pairwise=[_pw("b", "c", 0.9)])
        c3 = _corr(["a", "c"], "GAUSSIAN", 3, pairwise=[_pw("a", "c", -0.9)])
        c4 = _corr(["a", "d"], "GAUSSIAN", 4, pairwise=[_pw("a", "d", 0.3)])
        c5 = _corr(["b", "e"], "GAUSSIAN", 5, pairwise=[_pw("b", "e", 0.3)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3, c4, c5], _schema(["a", "b", "c", "d", "e"]))
        )
        assert unsupported == []
        assert len(conflicts) == 1
        assert set(conflicts[0].involved_fact_references) == {1, 2, 3}

    def test_correlation_at_exactly_one_is_infeasible_with_any_third_column(self):
        # rho(a,b)=1.0 forces b to be an exact linear function of a; any
        # THIRD, independently-stated, different correlation for a and b
        # against the same c is then over-determined and must match exactly.
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 1.0)])
        c2 = _corr(["a", "c"], "GAUSSIAN", 2, pairwise=[_pw("a", "c", 0.5)])
        c3 = _corr(
            ["b", "c"], "GAUSSIAN", 3, pairwise=[_pw("b", "c", 0.9)]
        )  # must equal 0.5, not 0.9
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1, c2, c3], _schema(["a", "b", "c"]))
        )
        assert unsupported == []
        assert len(conflicts) == 1
        assert conflicts[0].kind == "correlated_infeasible_matrix"

    def test_correlated_families_outside_matrix_scope_are_ignored(self):
        # CLAYTON's scalar theta isn't part of the correlation-matrix
        # feasibility check at all -- must not error, crash, or falsely flag.
        c1 = _corr(["a", "b"], "CLAYTON", 1, shared_parameters={"theta": 2.0})
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1], _schema(["a", "b"]))
        )
        assert conflicts == [] and unsupported == []

    def test_single_pairwise_fact_alone_is_trivially_feasible(self):
        c1 = _corr(["a", "b"], "GAUSSIAN", 1, pairwise=[_pw("a", "b", 0.999)])
        conflicts, unsupported = check_correlated_conflicts(
            _cluster([c1], _schema(["a", "b"]))
        )
        assert conflicts == [] and unsupported == []

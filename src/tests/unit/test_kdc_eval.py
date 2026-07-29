"""Key and Dependency Correctness: the normalisation checks.

Each test is a textbook normal-form violation, because that is what the metric
exists to catch -- a partial dependency does not announce itself at review time,
it shows up later as an update anomaly in production.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.evaluation.schema_level.kdc_eval import evaluate_kdc
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table


class _FD:
    """Stands in for a conceptual FunctionalDependency."""

    def __init__(self, determinant: Sequence[str], dependent: Sequence[str]) -> None:
        self.determinant = list(determinant)
        self.dependent = list(dependent)


def _table(name: str, cols: Sequence[str], pk: Optional[List[str]]) -> Table:
    return Table(
        name=name,
        columns=[Column(name=c, data_type=DataType.VARCHAR) for c in cols],
        primary_key=list(pk) if pk else [],
    )


def _schema(*tables: Table) -> Schema:
    return Schema(tables=list(tables), relationships=[])


def test_a_dependency_on_the_key_is_enforced() -> None:
    schema = _schema(
        _table("STUDENT", ["student_id", "full_name", "email"], ["student_id"])
    )
    r = evaluate_kdc(schema, [_FD(["STUDENT.student_id"], ["STUDENT.full_name"])])
    assert r.kdc == 1.0
    assert r.n_violations == 0
    assert r.n_checked == 1


def test_partial_dependency_on_a_composite_key_is_a_2nf_violation() -> None:
    """The classic: ENROLMENT(student_id, course_id) with course_title depending
    on course_id alone. Editing a course title means touching every enrolment."""
    schema = _schema(
        _table(
            "ENROLMENT",
            ["student_id", "course_id", "course_title", "grade"],
            ["student_id", "course_id"],
        )
    )
    r = evaluate_kdc(schema, [_FD(["ENROLMENT.course_id"], ["ENROLMENT.course_title"])])
    assert r.partial_2nf, r.as_dict()
    assert r.kdc == 0.0
    assert "part of the key" in r.partial_2nf[0]


def test_transitive_dependency_is_a_3nf_violation() -> None:
    """EMPLOYEE(employee_id) with department_name depending on department_id --
    a non-key attribute determining another non-key attribute."""
    schema = _schema(
        _table(
            "EMPLOYEE",
            ["employee_id", "department_id", "department_name"],
            ["employee_id"],
        )
    )
    r = evaluate_kdc(
        schema, [_FD(["EMPLOYEE.department_id"], ["EMPLOYEE.department_name"])]
    )
    assert r.transitive_3nf, r.as_dict()
    assert "non-key determines non-key" in r.transitive_3nf[0]


def test_a_superkey_determinant_is_accepted() -> None:
    """More than the key still determines the row, so it is enforceable."""
    schema = _schema(
        _table("ORDER", ["order_id", "line_no", "qty"], ["order_id", "line_no"])
    )
    r = evaluate_kdc(
        schema,
        [_FD(["ORDER.order_id", "ORDER.line_no"], ["ORDER.qty"])],
    )
    assert r.n_violations == 0
    assert r.kdc == 1.0


def test_a_cross_table_dependency_is_reported_but_not_penalised() -> None:
    """After decomposition these are normal, so counting them as defects would
    punish the schema for being normalised."""
    schema = _schema(
        _table("ORDER", ["order_id", "customer_id"], ["order_id"]),
        _table("CUSTOMER", ["customer_id", "full_name"], ["customer_id"]),
    )
    r = evaluate_kdc(schema, [_FD(["ORDER.customer_id"], ["CUSTOMER.full_name"])])
    assert r.cross_table, r.as_dict()
    assert r.n_violations == 0
    assert r.kdc == 1.0


def test_determining_part_of_the_key_is_not_a_violation() -> None:
    """A dependency whose dependent is entirely key columns says nothing about
    normalisation."""
    schema = _schema(
        _table("ENROLMENT", ["student_id", "course_id"], ["student_id", "course_id"])
    )
    r = evaluate_kdc(schema, [_FD(["ENROLMENT.student_id"], ["ENROLMENT.course_id"])])
    assert r.n_violations == 0


def test_a_table_with_no_primary_key_is_reported() -> None:
    schema = _schema(_table("SCRATCH", ["a", "b"], None))
    r = evaluate_kdc(schema, [])
    assert r.tables_without_key == ["SCRATCH"]


def test_renaming_tables_and_columns_cannot_change_the_score() -> None:
    """Name-blind like the rest of the suite: the FD refs are renamed with the
    schema, and only the key STRUCTURE decides the outcome."""
    a = _schema(
        _table(
            "ENROLMENT",
            ["student_id", "course_id", "course_title"],
            ["student_id", "course_id"],
        )
    )
    fd_a = [_FD(["ENROLMENT.course_id"], ["ENROLMENT.course_title"])]
    b = _schema(
        _table(
            "SIGNUP",
            ["learner_ref", "class_ref", "class_label"],
            ["learner_ref", "class_ref"],
        )
    )
    fd_b = [_FD(["SIGNUP.class_ref"], ["SIGNUP.class_label"])]
    assert evaluate_kdc(a, fd_a).kdc == evaluate_kdc(b, fd_b).kdc


def test_a_dependency_naming_a_vanished_table_is_not_counted_here() -> None:
    """That is a capacity failure, which IC reports; counting it twice would
    double-penalise one defect."""
    schema = _schema(_table("KEPT", ["kept_id", "x"], ["kept_id"]))
    r = evaluate_kdc(schema, [_FD(["GONE.a"], ["GONE.b"])])
    assert r.n_checked == 0
    assert r.n_violations == 0


def test_no_dependencies_scores_vacuously_but_reports_it() -> None:
    """A 1.0 with nothing checked must be distinguishable from an earned 1.0."""
    schema = _schema(_table("A", ["a_id"], ["a_id"]))
    r = evaluate_kdc(schema, [])
    assert r.kdc == 1.0
    assert r.n_checked == 0

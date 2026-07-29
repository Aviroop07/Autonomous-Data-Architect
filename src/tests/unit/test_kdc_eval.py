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
    # "Not penalised" means no violation is counted. The SCORE is undefined
    # rather than 1.0, because the only dependency supplied was never checkable
    # -- reporting a perfect score off an empty check is what this module used
    # to do wrong.
    assert r.kdc is None
    assert r.n_checked == 0


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


def test_nothing_checked_is_undefined_not_a_perfect_score() -> None:
    """CORRECTS AN EARLIER VERSION OF THIS TEST, which asserted `kdc == 1.0`
    here. Its docstring already said a vacuous 1.0 "must be distinguishable from
    an earned 1.0" -- and then made them the same number, distinguishable only by
    reading n_checked separately.

    They were not distinguished in practice. A benchmark case whose reference
    dependencies include a real 3NF violation reported kdc 1.000 with
    n_checked 0, because the reference table names had been renamed by Stage 2
    and every dependency dropped out of the denominator. "1 - violations/0" has
    no value, so it now reports None."""
    schema = _schema(_table("A", ["a_id"], ["a_id"]))
    r = evaluate_kdc(schema, [])
    assert r.kdc is None
    assert r.n_checked == 0


def test_a_renamed_table_resolves_through_the_alignment() -> None:
    """The bug this module had. Every other metric in the suite is NAME-BLIND --
    tables are aligned by structural signature -- but KDC matched dependencies to
    tables by name, and renaming is exactly what Stage 2 does.

    Measured on benchmark case 104: the reference dependencies name FAULT_EVENT
    and SOLAR_FARM while the pipeline produced FAULT_RECORD and SITE, so all four
    dependencies silently left the denominator and the score came back 1.000
    having checked NOTHING -- including a genuine `fault_code -> fault_wording`
    third-normal-form violation that was sitting in the generated schema.
    """
    schema = _schema(
        _table("FAULT_RECORD", ["fault_id", "code", "code_description"], ["fault_id"])
    )
    fd = _FD(["FAULT_EVENT.code"], ["FAULT_EVENT.code_description"])

    # Without the alignment the dependency cannot be located at all.
    blind = evaluate_kdc(schema, [fd])
    assert blind.n_checked == 0
    assert blind.kdc is None
    assert blind.unresolved_tables, "an unlocatable dependency must be reported"

    # With it, the rename resolves and the violation is found.
    aligned = evaluate_kdc(schema, [fd], name_map={"FAULT_EVENT": "FAULT_RECORD"})
    assert aligned.n_checked == 1
    assert aligned.transitive_3nf, aligned.as_dict()
    assert aligned.kdc == 0.0


def test_the_alignment_is_only_a_fallback_for_direct_hits() -> None:
    """A table whose name already matches must not be re-routed by the map."""
    schema = _schema(
        _table("ORDER", ["order_id", "customer_ref", "customer_city"], ["order_id"]),
        _table("OTHER", ["other_id"], ["other_id"]),
    )
    fd = _FD(["ORDER.customer_ref"], ["ORDER.customer_city"])
    r = evaluate_kdc(schema, [fd], name_map={"ORDER": "OTHER"})
    assert r.n_checked == 1
    assert r.transitive_3nf, "resolved against ORDER itself, not the map target"


def test_an_unresolved_dependency_is_counted_and_surfaced() -> None:
    """Silence was the original failure. An unlocatable dependency has to appear
    somewhere a reader will see it."""
    schema = _schema(_table("A", ["a_id"], ["a_id"]))
    r = evaluate_kdc(schema, [_FD(["GHOST.x"], ["GHOST.y"])])
    assert r.n_checked == 0
    assert len(r.unresolved_tables) == 1
    assert r.as_dict()["kdc_unresolved_tables"] == 1.0

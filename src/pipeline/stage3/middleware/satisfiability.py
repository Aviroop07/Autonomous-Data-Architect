from __future__ import annotations

import math
import re
from typing import List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.middleware.expression_parser import classify_expression
from src.pipeline.stage3.models.sql_models import (
    BinaryOperand,
    CardinalityConstraint,
    FanoutConstraint,
    OperandValue,
    SQLGroundedConstraint,
    StructuralKnob,
)
from src.pipeline.stage3.models.validation import Stage3Issue
from src.pipeline.stage3.utils.sql_validator import validate_sql_against_schema


ColumnKind = Literal["numeric", "text", "boolean", "date", "unknown"]

NUMERIC_TYPES = {"INT", "INTEGER", "FLOAT", "DECIMAL", "NUMERIC", "DOUBLE", "REAL", "BIGINT", "SMALLINT", "TINYINT"}
TEXT_TYPES = {"VARCHAR", "TEXT", "UUID", "JSON"}
DATE_TYPES = {"DATE", "DATETIME", "TIMESTAMP"}
BOOLEAN_TYPES = {"BOOLEAN", "BOOL"}
ORDERED_OPERATORS = {"GT", "GTE", "LT", "LTE"}
EPSILON = 1e-6
DEFAULT_CARDINALITY_KNOB_VALUE = 100
DEFAULT_FANOUT_KNOB_VALUE = 3


class IndexedStateConstraint(BaseModel):
    index: int
    constraint: SQLGroundedConstraint


class StateConstraintGroup(BaseModel):
    normalized_query: str
    indexed_constraints: List[IndexedStateConstraint]


class TableCardinalityBound(BaseModel):
    table_name: str
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    fact_references: List[int] = Field(default_factory=list)


class RelationshipFanoutBound(BaseModel):
    parent_table: str
    child_table: str
    min_children_per_parent: Optional[int] = None
    max_children_per_parent: Optional[int] = None
    fact_references: List[int] = Field(default_factory=list)


class ColumnKindRecord(BaseModel):
    column_name: str
    kind: ColumnKind


class ColumnIndexRecord(BaseModel):
    column_name: str
    index: int


class LiteralRecord(BaseModel):
    column_name: str
    value: OperandValue


class NullRequirementRecord(BaseModel):
    column_name: str
    expected_null: bool


def check_state_constraint_satisfiability(
    constraints: List[SQLGroundedConstraint],
    schema: Schema,
) -> List[Stage3Issue]:
    issues: List[Stage3Issue] = []
    grouped = _group_constraints_by_state_query(constraints)
    for group in grouped:
        issues.extend(_check_state_query_group(group, schema))
    return issues


def check_structural_satisfiability(
    cardinality_constraints: List[CardinalityConstraint],
    fanout_constraints: List[FanoutConstraint],
    schema: Schema,
) -> Tuple[List[Stage3Issue], List[StructuralKnob]]:
    cardinality_bounds, cardinality_issues = _merge_cardinality_bounds(cardinality_constraints, schema)
    fanout_bounds, fanout_issues = _merge_fanout_bounds(fanout_constraints, schema)
    issues = [*cardinality_issues, *fanout_issues]
    knobs = _discover_structural_knobs(cardinality_bounds, fanout_bounds, schema)
    if issues:
        return issues, knobs

    issues.extend(_check_fk_zero_parent_conflicts(cardinality_bounds, schema))
    if issues:
        return issues, knobs

    issues.extend(_check_structural_milp(cardinality_bounds, fanout_bounds, schema))
    return issues, knobs


def format_satisfiability_issues_for_retry(issues: List[Stage3Issue]) -> str:
    if not issues:
        return "SOLVER FEASIBILITY CHECK PASSED"

    lines = ["SOLVER FEASIBILITY CHECK FAILED"]
    for issue in issues:
        lines.append(f"- [{issue.severity.upper()}] {issue.code} at {issue.target}: {issue.message}")
        if issue.suggested_action:
            lines.append(f"  Suggested action: {issue.suggested_action}")
        if issue.fact_references:
            lines.append(f"  Related facts: {issue.fact_references}")
    return "\n".join(lines)


def _merge_cardinality_bounds(
    constraints: List[CardinalityConstraint],
    schema: Schema,
) -> Tuple[List[TableCardinalityBound], List[Stage3Issue]]:
    table_names = {table.name.upper() for table in schema.tables}
    bounds: List[TableCardinalityBound] = [
        TableCardinalityBound(table_name=table.name.upper(), min_rows=0)
        for table in schema.tables
    ]
    issues: List[Stage3Issue] = []

    for idx, constraint in enumerate(constraints):
        validation_errors = constraint._validate(schema)
        if validation_errors:
            issues.append(_structural_issue(
                f"cardinality_constraints[{idx}]",
                "INVALID_CARDINALITY_CONSTRAINT",
                "critical",
                validation_errors[0],
                "Repair or remove the invalid table cardinality constraint.",
                constraint.fact_references,
            ))
            continue

        table_name = constraint.table_name.upper()
        if table_name not in table_names:
            continue

        bound = _get_table_cardinality_bound(bounds, table_name)
        lower = bound.min_rows
        upper = bound.max_rows
        fact_ids = bound.fact_references
        constraint_lower, constraint_upper = constraint.bounds()
        if constraint_lower is not None:
            lower = constraint_lower if lower is None else max(lower, constraint_lower)
        if constraint_upper is not None:
            upper = constraint_upper if upper is None else min(upper, constraint_upper)
        fact_ids = _merge_fact_references(fact_ids, constraint.fact_references)
        bound.min_rows = lower
        bound.max_rows = upper
        bound.fact_references = fact_ids

        if lower is not None and upper is not None and lower > upper:
            issues.append(_structural_issue(
                f"cardinality_constraints[{idx}]",
                "UNSAT_CARDINALITY_BOUNDS",
                "critical",
                f"Table '{constraint.table_name}' has inconsistent cardinality bounds: min {lower} > max {upper}.",
                "Relax the cardinality bounds or remove the contradictory table-size facts.",
                fact_ids,
            ))

    return bounds, issues


def _merge_fanout_bounds_with_issues(
    constraints: List[FanoutConstraint],
    schema: Schema,
) -> Tuple[List[RelationshipFanoutBound], List[Stage3Issue]]:
    bounds: List[RelationshipFanoutBound] = []
    issues: List[Stage3Issue] = []
    for idx, constraint in enumerate(constraints):
        validation_errors = constraint._validate(schema)
        if validation_errors:
            issues.append(_structural_issue(
                f"fanout_constraints[{idx}]",
                "INVALID_FANOUT_CONSTRAINT",
                "critical",
                validation_errors[0],
                "Repair or remove the invalid parent-child fanout constraint.",
                constraint.fact_references,
            ))
            continue

        parent = constraint.parent_table.upper()
        child = constraint.child_table.upper()
        bound = _get_or_create_fanout_bound(bounds, parent, child)
        lower = bound.min_children_per_parent
        upper = bound.max_children_per_parent
        fact_ids = bound.fact_references
        if constraint.min_children_per_parent is not None:
            lower = constraint.min_children_per_parent if lower is None else max(lower, constraint.min_children_per_parent)
        if constraint.max_children_per_parent is not None:
            upper = constraint.max_children_per_parent if upper is None else min(upper, constraint.max_children_per_parent)
        fact_ids = _merge_fact_references(fact_ids, constraint.fact_references)
        bound.min_children_per_parent = lower
        bound.max_children_per_parent = upper
        bound.fact_references = fact_ids

        if lower is not None and upper is not None and lower > upper:
            issues.append(_structural_issue(
                f"fanout_constraints[{idx}]",
                "UNSAT_FANOUT_BOUNDS",
                "critical",
                (
                    f"Relationship {constraint.parent_table}->{constraint.child_table} has inconsistent "
                    f"fanout bounds: min {lower} > max {upper}."
                ),
                "Relax the fanout bounds or remove the contradictory relationship facts.",
                fact_ids,
            ))

    return bounds, issues


def _merge_fanout_bounds(
    constraints: List[FanoutConstraint],
    schema: Schema,
) -> Tuple[List[RelationshipFanoutBound], List[Stage3Issue]]:
    return _merge_fanout_bounds_with_issues(constraints, schema)


def _discover_structural_knobs(
    cardinality_bounds: List[TableCardinalityBound],
    fanout_bounds: List[RelationshipFanoutBound],
    schema: Schema,
) -> List[StructuralKnob]:
    incoming_counts: List[Tuple[str, int]] = [(table.name.upper(), 0) for table in schema.tables]
    for rel in (schema.relationships or []):
        child = rel.referencing_table.upper()
        _increment_incoming_count(incoming_counts, child)

    knobs: List[StructuralKnob] = []
    for table in schema.tables:
        table_name = table.name.upper()
        cardinality_bound = _get_table_cardinality_bound(cardinality_bounds, table_name)
        lower = cardinality_bound.min_rows
        upper = cardinality_bound.max_rows
        has_fixed_cardinality = lower is not None and upper is not None and lower == upper
        has_explicit_cardinality = _has_explicit_cardinality_constraint(table_name, cardinality_bounds)
        is_root_or_parent = _get_incoming_count(incoming_counts, table_name) == 0 or _is_parent_table(schema, table_name)
        has_incoming_fanout_bound = _has_incoming_fanout_bound(fanout_bounds, table_name)
        is_single_parent_leaf = _get_incoming_count(incoming_counts, table_name) == 1 and not _is_parent_table(schema, table_name)
        should_emit_cardinality_knob = (
            not has_fixed_cardinality
            and (
                has_explicit_cardinality
                or is_root_or_parent
                or (not is_single_parent_leaf and not has_incoming_fanout_bound)
            )
        )
        if should_emit_cardinality_knob:
            knobs.append(StructuralKnob(
                kind="table_cardinality",
                name=f"{table_name.lower()}_row_count",
                table_name=table_name,
                min_value=lower,
                max_value=upper,
                default_value=_default_cardinality(lower, upper),
                source="bounded" if upper is not None or (lower is not None and lower > 0) else "unconstrained",
                reason=(
                    f"No exact row count is fixed for table {table_name}; this cardinality must be chosen "
                    "before materialization."
                ),
            ))

    for rel in (schema.relationships or []):
        parent = rel.referred_table.upper()
        child = rel.referencing_table.upper()
        fanout_bound = _find_fanout_bound(fanout_bounds, parent, child)
        lower = fanout_bound.min_children_per_parent if fanout_bound else None
        upper = fanout_bound.max_children_per_parent if fanout_bound else None
        has_fixed_fanout = lower is not None and upper is not None and lower == upper
        child_has_explicit_cardinality = _has_explicit_cardinality_constraint(child, cardinality_bounds)
        has_explicit_fanout_bound = fanout_bound is not None
        single_parent_leaf_without_cardinality = (
            _get_incoming_count(incoming_counts, child) == 1
            and not _is_parent_table(schema, child)
            and not child_has_explicit_cardinality
        )
        if not has_fixed_fanout and (has_explicit_fanout_bound or single_parent_leaf_without_cardinality):
            knobs.append(StructuralKnob(
                kind="relationship_fanout",
                name=f"{parent.lower()}_to_{child.lower()}_fanout",
                parent_table=parent,
                child_table=child,
                min_value=lower,
                max_value=upper,
                default_value=_default_fanout(lower, upper),
                source="bounded" if lower is not None or upper is not None else "unconstrained",
                reason=(
                    f"Child table {child} is derived from parent table {parent}; the per-parent fanout "
                    "is not exactly fixed by extracted facts."
                ),
            ))

    return _dedupe_knobs(knobs)


def _check_fk_zero_parent_conflicts(
    cardinality_bounds: List[TableCardinalityBound],
    schema: Schema,
) -> List[Stage3Issue]:
    issues: List[Stage3Issue] = []
    for rel in (schema.relationships or []):
        parent = rel.referred_table.upper()
        child = rel.referencing_table.upper()
        parent_bound = _get_table_cardinality_bound(cardinality_bounds, parent)
        child_bound = _get_table_cardinality_bound(cardinality_bounds, child)
        if parent_bound.max_rows == 0 and child_bound.min_rows is not None and child_bound.min_rows > 0:
            issues.append(Stage3Issue(
                code="UNSAT_FK_PARENT_CARDINALITY",
                severity="critical",
                target=f"cardinality_constraints[{child}]",
                message=(
                    f"Child table {child} requires at least {child_bound.min_rows} rows, but parent table {parent} "
                    "is constrained to zero rows."
                ),
                suggested_action="Allow parent rows, lower the child row count, or remove the FK-dependent child rows.",
                fact_references=_merge_fact_references(parent_bound.fact_references, child_bound.fact_references),
            ))
    return issues


def _check_structural_milp(
    cardinality_bounds: List[TableCardinalityBound],
    fanout_bounds: List[RelationshipFanoutBound],
    schema: Schema,
) -> List[Stage3Issue]:
    table_names = [table.name.upper() for table in schema.tables]
    if not table_names:
        return []

    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    for table_name in table_names:
        bound = _get_table_cardinality_bound(cardinality_bounds, table_name)
        lower_bounds.append(float(bound.min_rows or 0))
        upper_bounds.append(float(bound.max_rows) if bound.max_rows is not None else np.inf)

    linear_rows: List[List[float]] = []
    lower_limits: List[float] = []
    upper_limits: List[float] = []
    involved_fact_ids: set[int] = set()
    for bound in cardinality_bounds:
        involved_fact_ids.update(bound.fact_references)

    for bound in fanout_bounds:
        parent = bound.parent_table
        child = bound.child_table
        parent_idx = _get_table_index(table_names, parent)
        child_idx = _get_table_index(table_names, child)
        if parent_idx is None or child_idx is None:
            continue
        involved_fact_ids.update(bound.fact_references)
        if bound.min_children_per_parent is not None:
            row = [0.0] * len(table_names)
            row[child_idx] = 1.0
            row[parent_idx] = -float(bound.min_children_per_parent)
            linear_rows.append(row)
            lower_limits.append(0.0)
            upper_limits.append(np.inf)
        if bound.max_children_per_parent is not None:
            row = [0.0] * len(table_names)
            row[child_idx] = 1.0
            row[parent_idx] = -float(bound.max_children_per_parent)
            linear_rows.append(row)
            lower_limits.append(-np.inf)
            upper_limits.append(0.0)

    if not linear_rows:
        return []

    constraints = LinearConstraint(
        A=np.array(linear_rows),
        lb=np.array(lower_limits),
        ub=np.array(upper_limits),
    )
    result = milp(
        c=np.zeros(len(table_names)),
        integrality=np.ones(len(table_names)),
        bounds=Bounds(lb=np.array(lower_bounds), ub=np.array(upper_bounds)),
        constraints=constraints,
        options={"time_limit": 10.0},
    )
    if result.success or result.status == 3:
        return []
    if result.status == 2:
        return [Stage3Issue(
            code="UNSAT_STRUCTURAL_CONSTRAINTS",
            severity="critical",
            target="structural_constraints",
            message="The extracted table cardinality and fanout constraints are mutually infeasible.",
            suggested_action="Relax row-count or fanout bounds, or make one of the structural knobs tunable.",
            fact_references=sorted(involved_fact_ids),
        )]
    return [Stage3Issue(
        code="UNKNOWN_STRUCTURAL_SOLVER_FAILURE",
        severity="high",
        target="structural_constraints",
        message=f"Structural MILP solver returned status {result.status}: {result.message}",
        suggested_action="Add tighter finite bounds or simplify the structural constraints.",
        fact_references=sorted(involved_fact_ids),
    )]


def _has_explicit_cardinality_constraint(
    table_name: str,
    cardinality_bounds: List[TableCardinalityBound],
) -> bool:
    bound = _get_table_cardinality_bound(cardinality_bounds, table_name.upper())
    return bool(bound.fact_references) or bound.max_rows is not None or (bound.min_rows is not None and bound.min_rows > 0)


def _default_cardinality(lower: Optional[int], upper: Optional[int]) -> int:
    if lower is not None and upper is not None:
        return max(lower, min(upper, (lower + upper) // 2))
    if lower is not None and lower > 0:
        return lower
    if upper is not None:
        return max(0, min(upper, DEFAULT_CARDINALITY_KNOB_VALUE))
    return DEFAULT_CARDINALITY_KNOB_VALUE


def _default_fanout(lower: Optional[int], upper: Optional[int]) -> int:
    if lower is not None and upper is not None:
        return max(lower, min(upper, (lower + upper) // 2))
    if lower is not None:
        return lower
    if upper is not None:
        return min(upper, DEFAULT_FANOUT_KNOB_VALUE)
    return DEFAULT_FANOUT_KNOB_VALUE


def _get_table_cardinality_bound(
    bounds: List[TableCardinalityBound],
    table_name: str,
) -> TableCardinalityBound:
    normalized = table_name.upper()
    for bound in bounds:
        if bound.table_name.upper() == normalized:
            return bound
    bound = TableCardinalityBound(table_name=normalized, min_rows=0)
    bounds.append(bound)
    return bound


def _find_fanout_bound(
    bounds: List[RelationshipFanoutBound],
    parent_table: str,
    child_table: str,
) -> Optional[RelationshipFanoutBound]:
    parent = parent_table.upper()
    child = child_table.upper()
    for bound in bounds:
        if bound.parent_table.upper() == parent and bound.child_table.upper() == child:
            return bound
    return None


def _get_or_create_fanout_bound(
    bounds: List[RelationshipFanoutBound],
    parent_table: str,
    child_table: str,
) -> RelationshipFanoutBound:
    existing = _find_fanout_bound(bounds, parent_table, child_table)
    if existing:
        return existing
    bound = RelationshipFanoutBound(parent_table=parent_table.upper(), child_table=child_table.upper())
    bounds.append(bound)
    return bound


def _merge_fact_references(first: List[int], second: List[int]) -> List[int]:
    merged: List[int] = []
    for fact_id in [*first, *second]:
        if fact_id not in merged:
            merged.append(fact_id)
    return sorted(merged)


def _increment_incoming_count(incoming_counts: List[Tuple[str, int]], table_name: str) -> None:
    normalized = table_name.upper()
    for idx, (candidate, count) in enumerate(incoming_counts):
        if candidate == normalized:
            incoming_counts[idx] = (candidate, count + 1)
            return
    incoming_counts.append((normalized, 1))


def _get_incoming_count(incoming_counts: List[Tuple[str, int]], table_name: str) -> int:
    normalized = table_name.upper()
    for candidate, count in incoming_counts:
        if candidate == normalized:
            return count
    return 0


def _dedupe_knobs(knobs: List[StructuralKnob]) -> List[StructuralKnob]:
    deduped: List[StructuralKnob] = []
    for knob in knobs:
        if not any(existing.get_signature() == knob.get_signature() for existing in deduped):
            deduped.append(knob)
    return deduped


def _is_parent_table(schema: Schema, table_name: str) -> bool:
    normalized = table_name.upper()
    return any(rel.referred_table.upper() == normalized for rel in (schema.relationships or []))


def _has_incoming_fanout_bound(
    fanout_bounds: List[RelationshipFanoutBound],
    table_name: str,
) -> bool:
    normalized = table_name.upper()
    return any(bound.child_table.upper() == normalized for bound in fanout_bounds)


def _get_table_index(table_names: List[str], table_name: str) -> Optional[int]:
    normalized = table_name.upper()
    for idx, candidate in enumerate(table_names):
        if candidate.upper() == normalized:
            return idx
    return None


def _group_constraints_by_state_query(
    constraints: List[SQLGroundedConstraint],
) -> List[StateConstraintGroup]:
    grouped: List[StateConstraintGroup] = []
    for idx, constraint in enumerate(constraints):
        normalized_query = " ".join(constraint.state_query.split()).lower()
        group = _get_or_create_state_constraint_group(grouped, normalized_query)
        group.indexed_constraints.append(IndexedStateConstraint(index=idx, constraint=constraint))
    return grouped


def _get_or_create_state_constraint_group(
    groups: List[StateConstraintGroup],
    normalized_query: str,
) -> StateConstraintGroup:
    for group in groups:
        if group.normalized_query == normalized_query:
            return group
    group = StateConstraintGroup(normalized_query=normalized_query, indexed_constraints=[])
    groups.append(group)
    return group


def _check_state_query_group(
    group: StateConstraintGroup,
    schema: Schema,
) -> List[Stage3Issue]:
    issues: List[Stage3Issue] = []
    first_constraint = group.indexed_constraints[0].constraint
    is_valid, result_columns, err_msg = validate_sql_against_schema(first_constraint.state_query, schema)
    if not is_valid:
        return [_issue(
            group.indexed_constraints[0].index,
            "INVALID_STATE_QUERY",
            "critical",
            f"State query is not executable against the global schema: {err_msg}",
            "Repair the state_query so it references only valid global schema tables and columns.",
            first_constraint.fact_references,
        )]

    unsupported_features = _detect_unsupported_state_query_features(first_constraint.state_query)
    if unsupported_features:
        return [_issue(
            group.indexed_constraints[0].index,
            "UNSUPPORTED_STATE_QUERY_EXPRESSION",
            "high",
            (
                "State query uses expressions outside solver v1: "
                + ", ".join(unsupported_features)
            ),
            "Rewrite the constraint into supported linear state-table predicates or route it to a future nonlinear/Big-M solver tier.",
            first_constraint.fact_references,
        )]

    column_kinds = _infer_state_column_kinds(result_columns, schema)
    numeric_columns = _numeric_columns_for_group(group.indexed_constraints, column_kinds)
    column_index = _build_column_index(numeric_columns)

    a_ub: List[List[float]] = []
    b_ub: List[float] = []
    a_eq: List[List[float]] = []
    b_eq: List[float] = []
    equality_literals: List[LiteralRecord] = []
    inequality_literals: List[LiteralRecord] = []
    null_requirements: List[NullRequirementRecord] = []

    for indexed_constraint in group.indexed_constraints:
        original_idx = indexed_constraint.index
        constraint = indexed_constraint.constraint
        left = _normalize_column_name(constraint.left_operand)
        if constraint.operator in {"IS_NULL", "IS_NOT_NULL"}:
            expected_null = constraint.operator == "IS_NULL"
            previous = _find_null_requirement(null_requirements, left)
            if previous is not None and previous.expected_null != expected_null:
                issues.append(_issue(
                    original_idx,
                    "UNSAT_NULL_CONFLICT",
                    "critical",
                    f"Column '{constraint.left_operand}' is required to be both NULL and NOT NULL.",
                    "Remove one of the contradictory nullability constraints or split the state query.",
                    constraint.fact_references,
                ))
            if previous is None:
                null_requirements.append(NullRequirementRecord(column_name=left, expected_null=expected_null))
            continue

        if constraint.right_operand is None:
            issues.append(_issue(
                original_idx,
                "UNSUPPORTED_MISSING_OPERAND",
                "critical",
                f"Operator '{constraint.operator}' requires a right operand for solver feasibility.",
                "Add a column or literal right_operand, or use IS_NULL/IS_NOT_NULL.",
                constraint.fact_references,
            ))
            continue

        right_operand = constraint.right_operand
        if constraint.operator in ORDERED_OPERATORS:
            ordered_issue = _append_ordered_constraint(
                original_idx,
                constraint,
                right_operand,
                column_kinds,
                column_index,
                a_ub,
                b_ub,
            )
            if ordered_issue:
                issues.append(ordered_issue)
            continue

        if constraint.operator == "EQUALS":
            eq_issue = _append_equality_constraint(
                original_idx,
                constraint,
                right_operand,
                column_kinds,
                column_index,
                a_eq,
                b_eq,
                equality_literals,
            )
            if eq_issue:
                issues.append(eq_issue)
            continue

        if constraint.operator == "NOT_EQUALS":
            if right_operand.kind == "literal":
                inequality_literals.append(LiteralRecord(column_name=left, value=right_operand.value))
            elif right_operand.kind == "column" and isinstance(right_operand.value, str):
                right = _normalize_column_name(right_operand.value)
                if right == left:
                    issues.append(_issue(
                        original_idx,
                        "UNSAT_SELF_INEQUALITY",
                        "critical",
                        f"Column '{constraint.left_operand}' cannot be NOT_EQUALS to itself.",
                        "Use a different right operand or remove the impossible constraint.",
                        constraint.fact_references,
                    ))

    issues.extend(_check_literal_conflicts(equality_literals, inequality_literals, group.indexed_constraints))
    if issues:
        return issues

    if column_index:
        linprog_result = linprog(
            c=np.zeros(len(column_index)),
            A_ub=np.array(a_ub) if a_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(a_eq) if a_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=[(None, None)] * len(column_index),
            method="highs",
        )
        if linprog_result.status == 2:
            involved_fact_ids = _collect_fact_references(group.indexed_constraints)
            issues.append(Stage3Issue(
                code="UNSAT_LINEAR_CONSTRAINTS",
                severity="critical",
                target=f"state_constraints[{group.indexed_constraints[0].index}]",
                message=(
                    "The supported numeric predicates for this state table are mutually infeasible "
                    "under continuous linear relaxation."
                ),
                suggested_action="Relax contradictory bounds/comparisons or split the logical rule into feasible state-table predicates.",
                fact_references=involved_fact_ids,
            ))
        elif linprog_result.status not in {0, 3}:
            involved_fact_ids = _collect_fact_references(group.indexed_constraints)
            issues.append(Stage3Issue(
                code="UNKNOWN_SOLVER_FAILURE",
                severity="high",
                target=f"state_constraints[{group.indexed_constraints[0].index}]",
                message=f"Linear feasibility solver returned status {linprog_result.status}: {linprog_result.message}",
                suggested_action="Simplify the state-table predicates or add explicit finite bounds.",
                fact_references=involved_fact_ids,
            ))

    return issues


def _append_ordered_constraint(
    original_idx: int,
    constraint: SQLGroundedConstraint,
    right_operand: BinaryOperand,
    column_kinds: List[ColumnKindRecord],
    column_index: List[ColumnIndexRecord],
    a_ub: List[List[float]],
    b_ub: List[float],
) -> Optional[Stage3Issue]:
    left = _normalize_column_name(constraint.left_operand)
    left_idx = _find_column_index(column_index, left)
    if _find_column_kind(column_kinds, left) != "numeric" or left_idx is None:
        return _issue(
            original_idx,
            "UNSUPPORTED_NON_NUMERIC_ORDERING",
            "high",
            f"Ordered operator '{constraint.operator}' requires numeric left operand '{constraint.left_operand}'.",
            "Use equality/null operators for non-numeric columns or emit a numeric aggregate/column.",
            constraint.fact_references,
        )

    if right_operand.kind == "literal":
        literal = _numeric_literal(right_operand.value)
        if literal is None:
            return _issue(
                original_idx,
                "UNSUPPORTED_NON_NUMERIC_LITERAL",
                "high",
                f"Ordered operator '{constraint.operator}' requires a numeric literal, got {right_operand.value!r}.",
                "Use a numeric literal or switch to an equality/null predicate.",
                constraint.fact_references,
            )
        row = _empty_row(column_index)
        _apply_column_literal_order(row, b_ub, left_idx, constraint.operator, literal)
        a_ub.append(row)
        return None

    if not isinstance(right_operand.value, str):
        return _issue(
            original_idx,
            "UNSUPPORTED_COLUMN_OPERAND",
            "high",
            "Column right_operand must be a string column name.",
            "Use a valid state-query output column name.",
            constraint.fact_references,
        )

    right = _normalize_column_name(right_operand.value)
    right_idx = _find_column_index(column_index, right)
    if _find_column_kind(column_kinds, right) != "numeric" or right_idx is None:
        return _issue(
            original_idx,
            "UNSUPPORTED_NON_NUMERIC_ORDERING",
            "high",
            f"Ordered operator '{constraint.operator}' requires numeric right operand '{right_operand.value}'.",
            "Use numeric state-query columns or switch to equality/null predicates.",
            constraint.fact_references,
        )

    row = _empty_row(column_index)
    _apply_column_column_order(row, b_ub, left_idx, right_idx, constraint.operator)
    a_ub.append(row)
    return None


def _append_equality_constraint(
    original_idx: int,
    constraint: SQLGroundedConstraint,
    right_operand: BinaryOperand,
    column_kinds: List[ColumnKindRecord],
    column_index: List[ColumnIndexRecord],
    a_eq: List[List[float]],
    b_eq: List[float],
    equality_literals: List[LiteralRecord],
) -> Optional[Stage3Issue]:
    left = _normalize_column_name(constraint.left_operand)
    if right_operand.kind == "literal":
        literal = right_operand.value
        previous = _find_literal_record(equality_literals, left)
        if previous is not None and previous.value != literal:
            return _issue(
                original_idx,
                "UNSAT_LITERAL_CONFLICT",
                "critical",
                f"Column '{constraint.left_operand}' is equated to incompatible literals {previous.value!r} and {literal!r}.",
                "Remove one of the contradictory equality predicates.",
                constraint.fact_references,
            )
        if previous is None:
            equality_literals.append(LiteralRecord(column_name=left, value=literal))

        left_idx = _find_column_index(column_index, left)
        if left_idx is not None:
            numeric = _numeric_literal(literal)
            if numeric is None:
                return _issue(
                    original_idx,
                    "UNSUPPORTED_NON_NUMERIC_LITERAL",
                    "high",
                    f"Numeric column '{constraint.left_operand}' is equated to non-numeric literal {literal!r}.",
                    "Use a numeric literal or correct the operand type.",
                    constraint.fact_references,
                )
            row = _empty_row(column_index)
            row[left_idx] = 1.0
            a_eq.append(row)
            b_eq.append(numeric)
        return None

    if not isinstance(right_operand.value, str):
        return _issue(
            original_idx,
            "UNSUPPORTED_COLUMN_OPERAND",
            "high",
            "Column right_operand must be a string column name.",
            "Use a valid state-query output column name.",
            constraint.fact_references,
        )

    right = _normalize_column_name(right_operand.value)
    left_idx = _find_column_index(column_index, left)
    right_idx = _find_column_index(column_index, right)
    if left_idx is not None or right_idx is not None:
        if _find_column_kind(column_kinds, left) != "numeric" or _find_column_kind(column_kinds, right) != "numeric" or left_idx is None or right_idx is None:
            return _issue(
                original_idx,
                "UNSUPPORTED_MIXED_TYPE_EQUALITY",
                "high",
                f"Cannot solve equality between numeric and non-numeric operands: {constraint.left_operand}, {right_operand.value}.",
                "Use matching operand types or a literal predicate.",
                constraint.fact_references,
            )
        row = _empty_row(column_index)
        row[left_idx] = 1.0
        row[right_idx] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)
    return None


def _infer_state_column_kinds(result_columns: List[str], schema: Schema) -> List[ColumnKindRecord]:
    schema_column_kinds: List[Tuple[str, List[ColumnKind]]] = []
    for table in schema.tables:
        for column in table.columns:
            _append_schema_column_kind(schema_column_kinds, column.name.lower(), _kind_from_data_type(column.data_type))

    inferred: List[ColumnKindRecord] = []
    for result_column in result_columns:
        normalized = _normalize_column_name(result_column)
        schema_kinds = _find_schema_column_kinds(schema_column_kinds, normalized)
        unique_kinds = _unique_column_kinds(schema_kinds)
        if len(unique_kinds) == 1:
            inferred.append(ColumnKindRecord(column_name=normalized, kind=unique_kinds[0]))
        else:
            inferred.append(ColumnKindRecord(column_name=normalized, kind=_kind_from_alias(normalized)))
    return inferred


def _numeric_columns_for_group(
    indexed_constraints: List[IndexedStateConstraint],
    column_kinds: List[ColumnKindRecord],
) -> set[str]:
    numeric_columns: set[str] = set()
    for indexed_constraint in indexed_constraints:
        constraint = indexed_constraint.constraint
        left = _normalize_column_name(constraint.left_operand)
        if constraint.operator in ORDERED_OPERATORS or _find_column_kind(column_kinds, left) == "numeric":
            if _find_column_kind(column_kinds, left) == "numeric":
                numeric_columns.add(left)
        if constraint.right_operand and constraint.right_operand.kind == "column" and isinstance(constraint.right_operand.value, str):
            right = _normalize_column_name(constraint.right_operand.value)
            if constraint.operator in ORDERED_OPERATORS or _find_column_kind(column_kinds, right) == "numeric":
                if _find_column_kind(column_kinds, right) == "numeric":
                    numeric_columns.add(right)
    return numeric_columns


def _kind_from_data_type(data_type: Optional[str]) -> ColumnKind:
    normalized = (data_type or "VARCHAR").upper().split("(")[0].strip()
    if normalized in NUMERIC_TYPES:
        return "numeric"
    if normalized in TEXT_TYPES:
        return "text"
    if normalized in DATE_TYPES:
        return "date"
    if normalized in BOOLEAN_TYPES:
        return "boolean"
    return "unknown"


def _detect_unsupported_state_query_features(state_query: str) -> List[str]:
    upper_query = state_query.upper()
    unsupported: List[str] = []
    if re.search(r"\bMIN\s*\([^)]*,", upper_query) or re.search(r"\bMAX\s*\([^)]*,", upper_query):
        _append_unique_text(unsupported, "scalar MIN/MAX gate")

    for expression in _extract_state_query_expressions(state_query):
        classification = classify_expression(expression)
        if classification.solver_tier == "nonlinear_product":
            _append_unique_text(unsupported, "nonlinear product expression")
        elif classification.solver_tier == "piecewise_linear":
            continue
        elif classification.solver_tier == "big_m_gate":
            _append_unique_text(unsupported, "Big-M MIN/MAX gate expression")
        elif classification.solver_tier == "unsupported" and _expression_has_solver_relevant_syntax(expression):
            _append_unique_text(unsupported, "unsupported derived expression")
    return unsupported


def _extract_state_query_expressions(state_query: str) -> List[str]:
    expressions: List[str] = []
    select_clauses = _extract_select_clauses(state_query)
    for clause in select_clauses:
        for select_item in _split_top_level_commas_for_query(clause):
            expression = _strip_select_alias(select_item.strip())
            if expression:
                expressions.append(expression)
                for aggregate_argument in _extract_aggregate_arguments(expression):
                    expressions.append(aggregate_argument)
    return expressions


def _extract_select_clauses(state_query: str) -> List[str]:
    clauses: List[str] = []
    idx = 0
    while idx < len(state_query):
        select_idx = _find_keyword_from(state_query, "SELECT", idx)
        if select_idx is None:
            break
        from_idx = _find_matching_from_for_select(state_query, select_idx + len("SELECT"))
        if from_idx is None:
            break
        clauses.append(state_query[select_idx + len("SELECT"):from_idx].strip())
        idx = from_idx + len("FROM")
    return clauses


def _find_keyword_from(text: str, keyword: str, start_idx: int) -> Optional[int]:
    upper_text = text.upper()
    upper_keyword = keyword.upper()
    idx = start_idx
    while idx <= len(text) - len(keyword):
        if upper_text[idx:idx + len(keyword)] == upper_keyword:
            before_ok = idx == 0 or not upper_text[idx - 1].isalnum()
            after_idx = idx + len(keyword)
            after_ok = after_idx == len(text) or not upper_text[after_idx].isalnum()
            if before_ok and after_ok:
                return idx
        idx += 1
    return None


def _find_matching_from_for_select(text: str, start_idx: int) -> Optional[int]:
    depth = 0
    in_quote = False
    upper_text = text.upper()
    idx = start_idx
    while idx <= len(text) - len("FROM"):
        char = text[idx]
        if char == "'":
            in_quote = not in_quote
            idx += 1
            continue
        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and upper_text[idx:idx + len("FROM")] == "FROM":
                before_ok = idx == 0 or not upper_text[idx - 1].isalnum()
                after_idx = idx + len("FROM")
                after_ok = after_idx == len(text) or not upper_text[after_idx].isalnum()
                if before_ok and after_ok:
                    return idx
        idx += 1
    return None


def _split_top_level_commas_for_query(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    in_quote = False
    start = 0
    for idx, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _strip_select_alias(select_item: str) -> str:
    alias_idx = _find_last_top_level_as(select_item)
    if alias_idx is not None:
        return select_item[:alias_idx].strip()
    return select_item.strip()


def _find_last_top_level_as(text: str) -> Optional[int]:
    depth = 0
    in_quote = False
    upper_text = text.upper()
    last_idx: Optional[int] = None
    idx = 0
    while idx <= len(text) - 2:
        char = text[idx]
        if char == "'":
            in_quote = not in_quote
            idx += 1
            continue
        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and upper_text[idx:idx + 2] == "AS":
                before_ok = idx == 0 or not upper_text[idx - 1].isalnum()
                after_idx = idx + 2
                after_ok = after_idx == len(text) or not upper_text[after_idx].isalnum()
                if before_ok and after_ok:
                    last_idx = idx
        idx += 1
    return last_idx


def _extract_aggregate_arguments(expression: str) -> List[str]:
    arguments: List[str] = []
    for function_name in ["SUM", "AVG"]:
        idx = 0
        upper = expression.upper()
        while idx < len(expression):
            prefix = f"{function_name}("
            start = upper.find(prefix, idx)
            if start == -1:
                break
            open_idx = start + len(function_name)
            close_idx = _find_matching_parenthesis(expression, open_idx)
            if close_idx is None:
                break
            arguments.append(expression[open_idx + 1:close_idx].strip())
            idx = close_idx + 1
    return arguments


def _find_matching_parenthesis(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    in_quote = False
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _expression_has_solver_relevant_syntax(expression: str) -> bool:
    upper = expression.upper()
    return bool(re.search(r"\bCASE\b|\bMIN\s*\(|\bMAX\s*\(|[*+/\-]", upper))


def _append_unique_text(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _kind_from_alias(column_name: str) -> ColumnKind:
    numeric_tokens = {
        "sum", "avg", "min", "max", "count", "total", "amount", "quantity", "qty",
        "rate", "score", "price", "cost", "revenue", "balance", "weight", "metric",
    }
    if any(token in column_name for token in numeric_tokens):
        return "numeric"
    return "unknown"


def _append_schema_column_kind(
    schema_column_kinds: List[Tuple[str, List[ColumnKind]]],
    column_name: str,
    kind: ColumnKind,
) -> None:
    for idx, (candidate, kinds) in enumerate(schema_column_kinds):
        if candidate == column_name:
            schema_column_kinds[idx] = (candidate, [*kinds, kind])
            return
    schema_column_kinds.append((column_name, [kind]))


def _find_schema_column_kinds(
    schema_column_kinds: List[Tuple[str, List[ColumnKind]]],
    column_name: str,
) -> List[ColumnKind]:
    for candidate, kinds in schema_column_kinds:
        if candidate == column_name:
            return kinds
    return []


def _unique_column_kinds(kinds: List[ColumnKind]) -> List[ColumnKind]:
    unique: List[ColumnKind] = []
    for kind in kinds:
        if kind not in unique:
            unique.append(kind)
    return unique


def _build_column_index(columns: set[str]) -> List[ColumnIndexRecord]:
    return [
        ColumnIndexRecord(column_name=column_name, index=idx)
        for idx, column_name in enumerate(sorted(columns))
    ]


def _find_column_kind(column_kinds: List[ColumnKindRecord], column_name: str) -> ColumnKind:
    normalized = column_name.lower()
    for record in column_kinds:
        if record.column_name == normalized:
            return record.kind
    return "unknown"


def _find_column_index(column_index: List[ColumnIndexRecord], column_name: str) -> Optional[int]:
    normalized = column_name.lower()
    for record in column_index:
        if record.column_name == normalized:
            return record.index
    return None


def _find_literal_record(records: List[LiteralRecord], column_name: str) -> Optional[LiteralRecord]:
    normalized = column_name.lower()
    for record in records:
        if record.column_name == normalized:
            return record
    return None


def _find_null_requirement(
    records: List[NullRequirementRecord],
    column_name: str,
) -> Optional[NullRequirementRecord]:
    normalized = column_name.lower()
    for record in records:
        if record.column_name == normalized:
            return record
    return None


def _find_indexed_constraint_by_left(
    indexed_constraints: List[IndexedStateConstraint],
    column_name: str,
) -> Optional[IndexedStateConstraint]:
    normalized = column_name.lower()
    for indexed_constraint in indexed_constraints:
        if _normalize_column_name(indexed_constraint.constraint.left_operand) == normalized:
            return indexed_constraint
    return None


def _apply_column_literal_order(
    row: List[float],
    b_ub: List[float],
    left_idx: int,
    operator: str,
    literal: float,
) -> None:
    if operator == "GT":
        row[left_idx] = -1.0
        b_ub.append(-(literal + EPSILON))
    elif operator == "GTE":
        row[left_idx] = -1.0
        b_ub.append(-literal)
    elif operator == "LT":
        row[left_idx] = 1.0
        b_ub.append(literal - EPSILON)
    elif operator == "LTE":
        row[left_idx] = 1.0
        b_ub.append(literal)


def _apply_column_column_order(
    row: List[float],
    b_ub: List[float],
    left_idx: int,
    right_idx: int,
    operator: str,
) -> None:
    if operator == "GT":
        row[left_idx] = -1.0
        row[right_idx] = 1.0
        b_ub.append(-EPSILON)
    elif operator == "GTE":
        row[left_idx] = -1.0
        row[right_idx] = 1.0
        b_ub.append(0.0)
    elif operator == "LT":
        row[left_idx] = 1.0
        row[right_idx] = -1.0
        b_ub.append(-EPSILON)
    elif operator == "LTE":
        row[left_idx] = 1.0
        row[right_idx] = -1.0
        b_ub.append(0.0)


def _check_literal_conflicts(
    equality_literals: List[LiteralRecord],
    inequality_literals: List[LiteralRecord],
    indexed_constraints: List[IndexedStateConstraint],
) -> List[Stage3Issue]:
    issues: List[Stage3Issue] = []
    for equality in equality_literals:
        for forbidden in inequality_literals:
            if equality.column_name == forbidden.column_name and equality.value == forbidden.value:
                indexed_constraint = _find_indexed_constraint_by_left(indexed_constraints, equality.column_name)
                if indexed_constraint is None:
                    continue
                issues.append(_issue(
                    indexed_constraint.index,
                    "UNSAT_LITERAL_CONFLICT",
                    "critical",
                    f"Column '{indexed_constraint.constraint.left_operand}' is constrained to equal and not equal {equality.value!r}.",
                    "Remove one of the contradictory equality predicates.",
                    indexed_constraint.constraint.fact_references,
                ))
    return issues


def _numeric_literal(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def _normalize_column_name(column_name: str) -> str:
    return column_name.split(".")[-1].lower()


def _empty_row(column_index: List[ColumnIndexRecord]) -> List[float]:
    return [0.0] * len(column_index)


def _issue(
    original_idx: int,
    code: str,
    severity: Literal["low", "medium", "high", "critical"],
    message: str,
    suggested_action: str,
    fact_references: List[int],
) -> Stage3Issue:
    return Stage3Issue(
        code=code,
        severity=severity,
        target=f"state_constraints[{original_idx}]",
        message=message,
        suggested_action=suggested_action,
        fact_references=fact_references,
    )


def _structural_issue(
    target: str,
    code: str,
    severity: Literal["low", "medium", "high", "critical"],
    message: str,
    suggested_action: str,
    fact_references: List[int],
) -> Stage3Issue:
    return Stage3Issue(
        code=code,
        severity=severity,
        target=target,
        message=message,
        suggested_action=suggested_action,
        fact_references=fact_references,
    )


def _collect_fact_references(indexed_constraints: List[IndexedStateConstraint]) -> List[int]:
    fact_ids: set[int] = set()
    for indexed_constraint in indexed_constraints:
        fact_ids.update(indexed_constraint.constraint.fact_references)
    return sorted(fact_ids)

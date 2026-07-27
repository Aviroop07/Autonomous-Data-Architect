"""Constraint = Relation + Condition + fact_references + severity
(Section 2, 11). No category/family tag (Section 11.1) -- the condition's
own Python type is already self-describing.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from src.pipeline.stage2.models.schema import Schema
from src.util.constraint_model.condition.cohesive import (
    Correlated,
    Distributed,
    StateSequence,
)
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RArithmetic,
    RExprUnion,
)
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.condition.validate import (
    ConditionUnion,
    validate_condition,
)
from src.util.constraint_model.relation.nodes import (
    RelationUnion,
    validate_relation_tree,
)
from src.util.constraint_model.relation.schema import synthesize_schema
from src.util.constraint_model.relation.validate import validate_relation


def _references_aggregate(expr: "RExprUnion") -> bool:
    if isinstance(expr, RAggregateRef):
        return True
    if isinstance(expr, RArithmetic):
        return _references_aggregate(expr.left) or _references_aggregate(expr.right)
    return False


def is_softenable(condition: "ConditionUnion") -> bool:
    """Section 11.2's fixed, deterministic, code-level softenability rule,
    keyed on the condition's own kind -- never a per-instance LLM
    judgment. Softenable: Distributed, Correlated, and a "plain aggregate-
    based moment fact" (an RComparison referencing at least one
    RAggregateRef, since MomentTarget was deliberately never given its own
    node -- Section 8.3). Everything else -- StateSequence, and any other
    ordinary predicate shape not recognized as a moment fact (uniqueness/
    NOT NULL have no dedicated node yet either) -- defaults to NOT
    softenable, the conservative reading for "binary integrity property or
    not even fact-derived at all.\""""
    if isinstance(condition, (Distributed, Correlated)):
        return True
    if isinstance(condition, StateSequence):
        return False
    if isinstance(condition, RComparison):
        return _references_aggregate(condition.left) or _references_aggregate(
            condition.right
        )
    return False


class Constraint(BaseModel):
    """Ties a Relation (where in the schema) to a Condition (the
    assertion), plus bookkeeping. `condition` is `ConditionUnion` --
    either an ordinary predicate tree or one of the three cohesive terms,
    never a mix (Section 9.3, enforced by that union's own type shape)."""

    relation: "RelationUnion" = Field(
        description="Where in the schema this constraint's columns live."
    )
    condition: "ConditionUnion" = Field(description="The assertion itself.")
    fact_references: List[int] = Field(
        min_length=1,
        description="Stage 1 atomic-fact IDs this constraint derives from.",
    )
    severity: str = Field(
        default="hard",
        description=(
            "'hard' by default. 'soft' is only ever assigned as a reconciliation "
            "OUTCOME (Section 11.2), never claimed at extraction time."
        ),
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.severity not in ("hard", "soft"):
            errors.append(
                f"Constraint.severity must be 'hard' or 'soft', got '{self.severity}'."
            )
        errors.extend(
            f"Constraint.relation: {e}" for e in validate_relation_tree(self.relation)
        )
        errors.extend(f"Constraint.condition: {e}" for e in self.condition._validate())
        if self.severity == "soft" and not is_softenable(self.condition):
            errors.append(
                f"Constraint.severity='soft' is not allowed for a "
                f"{type(self.condition).__name__} condition -- only Distributed/"
                "Correlated/plain aggregate-based moment facts are softenable "
                "(Section 11.2)."
            )
        return errors


def validate_constraint(constraint: Constraint, schema: Schema) -> List[str]:
    """Full cross-module validation needing the real, schema-declared
    `schema`: this Constraint's own structural validity, then relation/
    validate.py's cross-node checks, then condition/validate.py's column
    resolution and type-compatibility against the Relation's own
    synthesized schema. Each stage's errors are returned as-is if any
    stage fails -- later stages assume earlier ones already passed."""
    structural_errors = constraint._validate()
    if structural_errors:
        return structural_errors

    relation_errors = validate_relation(constraint.relation, schema)
    if relation_errors:
        return relation_errors

    eff, synth_errors = synthesize_schema(constraint.relation, schema)
    if eff is None:
        return synth_errors

    return validate_condition(constraint.condition, eff)

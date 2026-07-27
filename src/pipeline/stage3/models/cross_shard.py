"""Cross-shard constraint models for Stage 3 extraction agents.

These models represent the output of per-shard extraction agents. Each
constraint captures one atomic rule derived from Stage 1 facts, expressed
as an ON clause (table/join/aggregate context) and a CONDITION clause
(pure typed R-AST predicates, no SQL strings).

Architecture:
    Constraint              -- the universal constraint representation
    DistributionConstraint  -- specialized for distribution pins
    DerivedColumnConstraint -- specialized for computed columns
    CorrelatedConstraint    -- specialized for joint/correlation facts
    StateSequenceConstraint -- specialized for state-machine/lifecycle facts

    ExtractionOutput wrappers per agent family:
        StatisticalExtractionOutput
        StructuralExtractionOutput
        LogicExtractionOutput

Design doc: experiments/CONSTRAINT_REPRESENTATION_SPEC.md (v0.2)
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from src.pipeline.stage3.models.condition_nodes import RExprUnion, RPredicate
from src.pipeline.stage3.models.on_nodes import ONBaseTable, ONNode


# ---------------------------------------------------------------------------
# Top-level constraint (the universal output)
# ---------------------------------------------------------------------------


class Constraint(BaseModel):
    """A single constraint emitted by an extraction agent.

    Every constraint traces back to one or more Stage 1 facts via
    fact_references. The ON clause defines the table/join/aggregate
    context. The CONDITION clause is a pure R-AST predicate tree.
    """

    fact_references: List[int] = Field(
        min_length=1,
        description="Stage 1 fact IDs that state this rule. Non-empty.",
    )
    on: ONNode = Field(
        description="Table/join/aggregate context (hybrid, normalized to pure objects)."
    )
    condition: RPredicate = Field(
        description="The rule expressed as typed R-AST nodes. No SQL strings."
    )
    category: Literal["statistical", "structural", "logic", "temporal", "derived"] = (
        Field(description="Constraint family for routing to the correct agent/auditor.")
    )
    severity: Literal["hard", "soft"] = Field(
        default="hard",
        description="Hard = must hold exactly; soft = best-effort / approximate.",
    )
    rename: Optional[dict[str, str]] = Field(
        default=None,
        description="Fallback name mapping for derived columns. Key is the "
        "expression string, value is the desired column name.",
    )

    @field_validator("fact_references")
    @classmethod
    def _no_duplicates(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("fact_references contains duplicates.")
        return v

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.fact_references:
            errors.append("Constraint.fact_references cannot be empty.")
        if self.rename is not None and not self.rename:
            errors.append("Constraint.rename, if provided, must be non-empty.")
        return errors


# ---------------------------------------------------------------------------
# DistributionConstraint (specialized)
# ---------------------------------------------------------------------------

_DISTRIBUTION_FAMILIES = frozenset(
    {"GAUSSIAN", "LOG_NORMAL", "BETA", "POISSON", "CATEGORICAL", "UNIFORM"}
)


class DistributionConstraint(BaseModel):
    """Pin a column to a specific distribution family + parameters.

    Specialized for statistical extraction: avoids the ~ operator in
    conditions. The ON must be a single base table (distributions apply
    to one column in one table).
    """

    fact_references: List[int] = Field(
        min_length=1,
        description="Stage 1 fact IDs that state this distribution.",
    )
    on: ONBaseTable = Field(
        description="The table containing the column. Must be a single base table."
    )
    column: str = Field(description="Column name (lower_snake_case).")
    family: Literal[
        "GAUSSIAN", "LOG_NORMAL", "BETA", "POISSON", "CATEGORICAL", "UNIFORM"
    ] = Field(description="Distribution family.")
    parameters: dict[str, Union[float, List[str], List[float]]] = Field(
        description="Family-specific parameters (e.g. mean/std_dev for GAUSSIAN)."
    )
    if_condition: Optional[RPredicate] = Field(
        default=None,
        description="Structured predicate restricting when this distribution applies. "
        "No SQL strings.",
    )

    @field_validator("parameters")
    @classmethod
    def _validate_params(cls, v: dict, info) -> dict:
        family = info.data.get("family")
        if family is None:
            return v

        if family == "GAUSSIAN":
            required = {"mean", "std_dev"}
            if not required.issubset(v.keys()):
                raise ValueError(
                    f"GAUSSIAN requires parameters {required}, got {set(v.keys())}."
                )
            if v["std_dev"] <= 0:
                raise ValueError("GAUSSIAN std_dev must be positive.")
            if v.get("std_dev", 0) > 10 * abs(v.get("mean", 1)):
                raise ValueError(
                    "GAUSSIAN std_dev is >10x the mean -- likely wrong scale."
                )

        elif family == "LOG_NORMAL":
            required = {"mean", "std_dev"}
            if not required.issubset(v.keys()):
                raise ValueError(
                    f"LOG_NORMAL requires parameters {required}, got {set(v.keys())}."
                )
            if v["std_dev"] <= 0:
                raise ValueError("LOG_NORMAL std_dev must be positive.")

        elif family == "BETA":
            required = {"alpha", "beta"}
            if not required.issubset(v.keys()):
                raise ValueError(
                    f"BETA requires parameters {required}, got {set(v.keys())}."
                )
            if v["alpha"] <= 0 or v["beta"] <= 0:
                raise ValueError("BETA alpha and beta must be positive.")

        elif family == "POISSON":
            required = {"lam"}
            if not required.issubset(v.keys()):
                raise ValueError(
                    f"POISSON requires parameter {required}, got {set(v.keys())}."
                )
            if v["lam"] <= 0:
                raise ValueError("POISSON lambda must be positive.")

        elif family == "CATEGORICAL":
            required = {"categories"}
            if not required.issubset(v.keys()):
                raise ValueError(
                    f"CATEGORICAL requires parameter {required}, got {set(v.keys())}."
                )
            cats = v["categories"]
            if not isinstance(cats, list) or len(cats) == 0:
                raise ValueError("CATEGORICAL categories must be a non-empty list.")
            probs = v.get("probabilities")
            if probs is not None:
                if len(probs) != len(cats):
                    raise ValueError(
                        "CATEGORICAL probabilities length must match categories."
                    )
                if not math.isclose(sum(probs), 1.0, rel_tol=1e-5):
                    raise ValueError("CATEGORICAL probabilities must sum to 1.0.")
                if any(p < 0 for p in probs):
                    raise ValueError("CATEGORICAL probabilities cannot be negative.")

        elif family == "UNIFORM":
            required = {"min_value", "max_value"}
            if not required.issubset(v.keys()):
                raise ValueError(
                    f"UNIFORM requires parameters {required}, got {set(v.keys())}."
                )
            if v["min_value"] > v["max_value"]:
                raise ValueError("UNIFORM min_value > max_value.")

        return v

    @field_validator("fact_references")
    @classmethod
    def _no_duplicates(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("fact_references contains duplicates.")
        return v

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.fact_references:
            errors.append("DistributionConstraint.fact_references cannot be empty.")
        if not self.column.strip():
            errors.append("DistributionConstraint.column cannot be empty.")
        return errors


# ---------------------------------------------------------------------------
# DerivedColumnConstraint (specialized)
# ---------------------------------------------------------------------------


class DerivedColumnConstraint(BaseModel):
    """A column computed from other columns via arithmetic.

    Emitted when a fact implies a derived column that doesn't exist in the
    Stage 2 schema. Stage 3 emits a DerivedColumnConstraint AND a
    BasePatch to add the column to the schema.
    """

    fact_references: List[int] = Field(
        min_length=1,
        description="Stage 1 fact IDs that state this derivation.",
    )
    target_table: str = Field(
        description="Table to add the derived column to (UPPER_SNAKE_CASE)."
    )
    target_column: str = Field(
        description="Name of the derived column (lower_snake_case)."
    )
    expression: RExprUnion = Field(
        description="Arithmetic tree defining the derivation."
    )
    referenced_tables: List[str] = Field(
        min_length=1,
        description="All tables whose columns appear in the expression.",
    )

    @field_validator("fact_references")
    @classmethod
    def _no_duplicates(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("fact_references contains duplicates.")
        return v

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.fact_references:
            errors.append("DerivedColumnConstraint.fact_references cannot be empty.")
        if not self.target_table.strip():
            errors.append("DerivedColumnConstraint.target_table cannot be empty.")
        if not self.target_column.strip():
            errors.append("DerivedColumnConstraint.target_column cannot be empty.")
        if not self.referenced_tables:
            errors.append("DerivedColumnConstraint.referenced_tables cannot be empty.")
        return errors


# ---------------------------------------------------------------------------
# CorrelatedConstraint (specialized) -- Tier B: mirrors constraint_model's
# Correlated node closely enough for the bridge to construct it DIRECTLY
# (no lossy approximation through a generic predicate tree, unlike the
# pre-Tier-B fallback of cramming a correlation fact into an RComparison).
# ---------------------------------------------------------------------------

_CORRELATION_FAMILIES = frozenset(
    {"GAUSSIAN", "STUDENT_T", "CLAYTON", "GUMBEL", "FRANK"}
)


class PairwiseCorrelationSpec(BaseModel):
    """One partially-specified entry of a CorrelatedConstraint's implied
    correlation matrix. Omitting a pair leaves it a free variable for
    Stage 4 -- pairwise entries are never required to be exhaustive."""

    left: str = Field(description="One of CorrelatedConstraint.columns.")
    right: str = Field(description="Another of CorrelatedConstraint.columns.")
    value: float = Field(description="Correlation value in [-1, 1].")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not (-1.0 <= self.value <= 1.0):
            errors.append(
                f"PairwiseCorrelationSpec.value must be in [-1, 1], got {self.value}."
            )
        if self.left == self.right:
            errors.append(
                "PairwiseCorrelationSpec: left and right must be different columns."
            )
        return errors


class CorrelatedConstraint(BaseModel):
    """Joint dependence across an arbitrary-arity column list. `on` provides
    the schema context making every named column accessible -- a single
    table if all columns live there, or an ONJoin reaching every table
    involved. `pairwise` may be partial (a fact stating only a qualitative
    direction with no number should omit that pair entirely, never invent
    an approximate value)."""

    fact_references: List[int] = Field(
        min_length=1, description="Stage 1 fact IDs that state this joint dependence."
    )
    on: ONNode = Field(
        description="Table/join context making every column in `columns` accessible."
    )
    columns: List[str] = Field(
        min_length=2, description="The joint column set (unqualified names)."
    )
    family: Literal["GAUSSIAN", "STUDENT_T", "CLAYTON", "GUMBEL", "FRANK"] = Field(
        default="GAUSSIAN",
        description="Copula family. Default to GAUSSIAN when the fact only states a "
        "qualitative direction with no distributional shape implied.",
    )
    pairwise: List[PairwiseCorrelationSpec] = Field(
        default_factory=list, description="Partial pairwise correlations."
    )
    shared_parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Family-wide shared parameters, e.g. STUDENT_T's degrees-of-freedom.",
    )

    @field_validator("fact_references")
    @classmethod
    def _no_duplicates(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("fact_references contains duplicates.")
        return v

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.fact_references:
            errors.append("CorrelatedConstraint.fact_references cannot be empty.")
        # columns' >= 2-entries rule is enforced by Field(min_length=2) at
        # construction time, matching constraint_model's own Correlated --
        # no redundant runtime check needed here.
        if len(self.columns) != len(set(self.columns)):
            errors.append("CorrelatedConstraint.columns contains duplicates.")
        if self.family not in _CORRELATION_FAMILIES:
            errors.append(
                f"CorrelatedConstraint.family must be one of {sorted(_CORRELATION_FAMILIES)}, "
                f"got '{self.family}'."
            )
        col_set = set(self.columns)
        for i, pw in enumerate(self.pairwise):
            errors.extend(
                f"CorrelatedConstraint.pairwise[{i}]: {e}" for e in pw._validate()
            )
            if pw.left not in col_set:
                errors.append(
                    f"CorrelatedConstraint.pairwise[{i}].left '{pw.left}' not in columns."
                )
            if pw.right not in col_set:
                errors.append(
                    f"CorrelatedConstraint.pairwise[{i}].right '{pw.right}' not in columns."
                )
        return errors


# ---------------------------------------------------------------------------
# StateSequenceConstraint (specialized) -- Tier B: mirrors constraint_model's
# StateSequence node closely enough for direct bridging.
# ---------------------------------------------------------------------------


class StateTransitionSpec(BaseModel):
    """One directed edge in a StateSequenceConstraint's transition graph."""

    from_state: str = Field(description="The sequence_column value transitioned from.")
    to_state: str = Field(description="The sequence_column value transitioned to.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.from_state == self.to_state:
            errors.append(
                "StateTransitionSpec: from_state and to_state must differ "
                "(a self-loop is not a transition)."
            )
        return errors


class StateSequenceConstraint(BaseModel):
    """State-machine fact over a single categorical column's value, e.g. an
    order's status must follow ready -> packed -> shipped -> delivered.
    `on` is single-table (the sequence column lives on one table) -- this is
    a transition-graph invariant on that column's CURRENT value, not a
    window/ordering claim over multiple rows (no event-log table is
    assumed to exist)."""

    fact_references: List[int] = Field(
        min_length=1, description="Stage 1 fact IDs that state this sequencing rule."
    )
    on: ONBaseTable = Field(description="The table the sequence column lives on.")
    sequence_column: str = Field(description="The categorical column tracked as state.")
    allowed_transitions: List[StateTransitionSpec] = Field(default_factory=list)
    forbidden_transitions: List[StateTransitionSpec] = Field(default_factory=list)
    strict: bool = Field(
        default=False,
        description="If True, this fact asserts the sequence is acyclic -- a cycle in "
        "the merged allowed-transitions graph across all facts sharing this "
        "table/sequence_column becomes a conflict. Cycles are allowed "
        "by default (e.g. a legitimate returns/reprocessing loop).",
    )

    @field_validator("fact_references")
    @classmethod
    def _no_duplicates(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("fact_references contains duplicates.")
        return v

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.fact_references:
            errors.append("StateSequenceConstraint.fact_references cannot be empty.")
        if not self.sequence_column.strip():
            errors.append("StateSequenceConstraint.sequence_column cannot be empty.")
        for i, t in enumerate(self.allowed_transitions):
            errors.extend(
                f"StateSequenceConstraint.allowed_transitions[{i}]: {e}"
                for e in t._validate()
            )
        for i, t in enumerate(self.forbidden_transitions):
            errors.extend(
                f"StateSequenceConstraint.forbidden_transitions[{i}]: {e}"
                for e in t._validate()
            )
        allowed_set = {(t.from_state, t.to_state) for t in self.allowed_transitions}
        forbidden_set = {(t.from_state, t.to_state) for t in self.forbidden_transitions}
        conflict = allowed_set & forbidden_set
        if conflict:
            errors.append(
                f"StateSequenceConstraint: transition(s) {sorted(conflict)} are asserted "
                "both allowed and forbidden within this same fact."
            )
        return errors


# ---------------------------------------------------------------------------
# Unified extraction output (single constraint_generator agent)
# ---------------------------------------------------------------------------


class UnifiedExtractionOutput(BaseModel):
    """Output of the single, unified constraint_generator agent (replaces
    the 3 separate statistical/structural/logic extraction outputs above --
    kept for reference/backward compatibility, no longer produced by any
    live agent). One generator now extracts every constraint category in
    one call; the `category` a constraint belongs to is simply which list
    it landed in here, mirroring Stage3Output's own 7-way split."""

    distributions: List[DistributionConstraint] = Field(
        default_factory=list,
        description="Distribution pins (column -> family + parameters).",
    )
    moment_targets: List[Constraint] = Field(
        default_factory=list,
        description="Moment-target constraints (mean/median pins).",
    )
    correlations: List[CorrelatedConstraint] = Field(
        default_factory=list,
        description="Joint/correlation constraints (Tier B: typed, direct-bridgeable).",
    )
    structural_constraints: List[Constraint] = Field(
        default_factory=list,
        description="Structural constraints (cardinality, fanout, uniqueness, aggregation).",
    )
    logic_constraints: List[Constraint] = Field(
        default_factory=list,
        description="Logic constraints (format, cross-column, temporal).",
    )
    derived_columns: List[DerivedColumnConstraint] = Field(
        default_factory=list,
        description="Arithmetic column derivations (e.g. total = price * quantity).",
    )
    state_sequences: List[StateSequenceConstraint] = Field(
        default_factory=list,
        description="State-machine/lifecycle constraints (Tier B: typed, direct-bridgeable).",
    )

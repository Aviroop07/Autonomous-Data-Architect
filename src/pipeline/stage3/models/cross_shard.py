"""Cross-shard constraint models for Stage 3 extraction agents.

These models represent the output of per-shard extraction agents. Each
constraint captures one atomic rule derived from Stage 1 facts, expressed
as an ON clause (table/join/aggregate context) and a CONDITION clause
(pure typed R-AST predicates, no SQL strings).

Architecture:
    Constraint              -- the universal constraint representation
    DistributionConstraint  -- specialized for distribution pins
    DerivedColumnConstraint -- specialized for computed columns

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
# Extraction output wrappers (per agent family)
# ---------------------------------------------------------------------------


class StatisticalExtractionOutput(BaseModel):
    """Output of the Statistical extraction agent.

    Separates distribution pins (specialized) from moment-target
    constraints and correlation constraints (generic Constraint).
    """

    distributions: List[DistributionConstraint] = Field(
        default_factory=list,
        description="Distribution pins (column -> family + parameters).",
    )
    moment_targets: List[Constraint] = Field(
        default_factory=list,
        description="Moment-target constraints (mean/median pins).",
    )
    correlations: List[Constraint] = Field(
        default_factory=list,
        description="Column-correlation constraints.",
    )


class StructuralExtractionOutput(BaseModel):
    """Output of the Structural+Aggregate extraction agent.

    All outputs are generic Constraint objects -- the ON clause
    distinguishes cardinalities, fanouts, aggregations, etc.
    """

    constraints: List[Constraint] = Field(
        default_factory=list,
        description="Structural constraints (cardinality, fanout, uniqueness, aggregation).",
    )


class LogicExtractionOutput(BaseModel):
    """Output of the Logic extraction agent.

    Generic Constraint objects for format/cross-column/temporal rules,
    plus DerivedColumnConstraint separately (specialized: arithmetic
    derivations like `total = price * quantity`, fed to cycle detection).
    """

    constraints: List[Constraint] = Field(
        default_factory=list,
        description="Logic constraints (format, cross-column, temporal).",
    )
    derived: List[DerivedColumnConstraint] = Field(
        default_factory=list,
        description="Arithmetic column derivations (e.g. total = price * quantity).",
    )

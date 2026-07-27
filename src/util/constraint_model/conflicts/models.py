"""The Conflict/ConflictReport shapes every conflicts/ submodule returns.

`softenable` reflects Section 11.2's fixed, kind-based rule: a conflict is
only ever softenable if EVERY Constraint it involves has a softenable
condition (Distributed/Correlated/plain aggregate-based moment facts).
Involving even one never-softenable condition (StateSequence transitions,
or a fact-independent structural equation with no fact behind it at all)
means the conflict can never be resolved by downgrading a side to soft --
it stays a hard, unresolved contradiction until something else changes.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

ConflictKind = Literal[
    "distributed_family_mismatch",
    "distributed_parameter_mismatch",
    "correlated_value_mismatch",
    "correlated_precondition_violation",
    "correlated_infeasible_matrix",
    "moment_value_mismatch",
    "moment_vs_distributed_mismatch",
    "population_reconciliation_infeasible",
    "state_sequence_direct_contradiction",
    "state_sequence_cycle",
    "structural_overconstrained",
]


class Conflict(BaseModel):
    kind: ConflictKind
    summary: str = Field(description="One-line human-readable description.")
    involved_fact_references: List[int] = Field(
        default_factory=list,
        description="Union of fact_references from every Constraint involved, sorted.",
    )
    detail: str = Field(
        description="Fuller explanation, including the actual numeric evidence."
    )
    softenable: bool = Field(
        description="Whether this specific conflict could ever be resolved by downgrading "
        "an involved Constraint to severity='soft' (Section 11.2's kind-based rule)."
    )


class ConflictReport(BaseModel):
    conflicts: List[Conflict] = Field(default_factory=list)
    unsupported: List[str] = Field(
        default_factory=list,
        description="Cases this evaluation could not determine either way (e.g. a "
        "non-chordal correlation pattern) -- NOT conflicts, but not silently "
        "clean either; a human/future-agent should know these were skipped.",
    )

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

"""Stage 3's final output for anything it could not resolve on its own.

Per the Stage 3/Stage 4 division of labor (project memory
stage3_stage4_division_of_labor): Stage 3 must surface what's determined,
flag genuine infeasibility, and PROBE free/unresolved things to Stage 4 --
never guess a value for them itself. This module is the probe half of
that contract; Stage 4's own handling of these probes is out of scope
here.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field

from src.util.algorithms.dof_graph import OverconstrainedBlock
from src.util.constraint_model.conflicts.models import Conflict


class CycleIssue(BaseModel):
    """A genuine (no-fixed-point) derived-column cycle. Carries enough
    structure for the conflict-reconciliation agent to trace back to the
    originating facts -- a bare description string isn't enough to know
    which NL facts to re-examine or which family produced it (always
    `logic`, since DerivedColumnConstraint is only ever emitted by
    logic_extractor). See middleware/cycles.py's detect_derived_cycles."""

    description: str
    nodes: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="'table.column' identifiers around the cycle.",
    )
    fact_references: Tuple[int, ...] = Field(default_factory=tuple)


class ReconciliationVerdict(str, Enum):
    """What the conflict-reconciliation agent decided about one detected
    conflict, after re-examining the original NL facts against what was
    extracted."""

    MISEXTRACTION = "MISEXTRACTION"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    GENUINE_CONTRADICTION = "GENUINE_CONTRADICTION"


class MisextractionFix(BaseModel):
    """One correction the reconciliation agent identified -- which fact it
    should re-derive from, and a natural-language hint describing the
    mistake, to be injected into the (single, unified) generator's next
    extraction-loop round for that shard. A single conflict can require
    fixes to more than one fact."""

    fact_id: int
    guidance: str = Field(
        description="What the prior extraction got wrong and how to fix it, "
        "e.g. 'fact 42 states the distribution is conditional on loyalty_tier "
        "= Platinum -- your prior extraction dropped this condition.'"
    )


class ConflictReconciliation(BaseModel):
    """The conflict-reconciliation agent's structured output for one
    detected conflict within a reconciled group (a Conflict.kind-tagged
    engine finding, an overconstrained block, or a derived-column cycle)."""

    conflict_ref: str = Field(
        description="Identifies which conflict this addresses -- echoed back "
        "exactly from the conflict item it was given."
    )
    verdict: ReconciliationVerdict
    reasoning: str = Field(
        description="Why this verdict -- cite the specific facts and what "
        "they actually say."
    )
    fixes: list[MisextractionFix] = Field(
        default_factory=list,
        description="Populated only when verdict == MISEXTRACTION.",
    )


class GroupReconciliation(BaseModel):
    """The conflict-reconciliation agent's structured output for one WHOLE
    GROUP of conflicts reconciled together (grouped because they share
    schema locality -- see orchestration/stage3/entry.py's
    _group_by_schema_locality). One verdict per conflict_ref it was given
    -- a single call can require different verdicts for different members
    of the same group."""

    verdicts: list[ConflictReconciliation] = Field(
        default_factory=list,
        description="Exactly one entry per conflict_ref given in the request, "
        "in any order.",
    )


class DismissedConflict(BaseModel):
    """A conflict the reconciliation agent judged FALSE_POSITIVE -- kept
    visible (not silently dropped) so a systematically-wrong reconciler can
    be caught, and so the report stays honest about what was actually
    decided vs. never came up."""

    conflict_ref: str
    reason: str
    fact_references: list[int] = Field(default_factory=list)


class VariableProbe(BaseModel):
    """A DOF-graph variable with no pinning constraint anywhere -- a
    genuine degree of freedom. Stage 4 decides how to handle it (most
    likely by generating parameterized code), not Stage 3."""

    variable_name: str = Field(
        description="The DOF graph Variable's name, e.g. 'ORDER.shipping_cost.mean'."
    )
    lower_bound: float | None = Field(
        default=None, description="Known lower bound, if any fact stated one."
    )
    upper_bound: float | None = Field(
        default=None, description="Known upper bound, if any fact stated one."
    )
    fact_references: list[int] = Field(
        default_factory=list,
        description="Facts that mention this variable without pinning it (e.g. a range bound) -- empty if genuinely unmentioned.",
    )


class LostShardReason(str, Enum):
    """Why a shard contributed nothing. The distinction matters to Stage 4:
    EXTRACTION_FAILED means we never got a usable answer at all, whereas
    DETERMINISTIC_CHECK_UNRESOLVED means we got constraints and deliberately
    withheld them because they never passed canonicalization or column
    resolution -- i.e. they referenced things that may not exist."""

    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    DETERMINISTIC_CHECK_UNRESOLVED = "DETERMINISTIC_CHECK_UNRESOLVED"


class LostShard(BaseModel):
    """A shard whose constraints did not make it into the output.

    Recorded rather than merely logged: Stage 4 consumes this object, not the
    log, so a silently-dropped shard would otherwise be indistinguishable from
    a shard that genuinely had nothing to contribute.
    """

    shard_index: int = Field(description="The shard's index in this run.")
    reason: LostShardReason
    detail: str = Field(
        default="",
        description="Human-readable specifics, e.g. how many iterations the "
        "loop burned before its retry budget ran out.",
    )
    fact_references: list[int] = Field(
        default_factory=list,
        description="The facts allocated to this shard, i.e. exactly what went "
        "unrepresented. Lets a caller tell which parts of the spec lost their "
        "constraints rather than only that something was lost.",
    )
    withheld_constraint_count: int = Field(
        default=0,
        description="How many constraints were quarantined rather than shipped. "
        "0 for EXTRACTION_FAILED, since nothing was ever produced.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="The deterministic checker's own unresolved complaints -- "
        "WHY the shard was withheld, not merely that it was. Without these a "
        "run that yields zero constraints is undiagnosable: the loop archives "
        "these errors internally and they were never surfaced anywhere, so the "
        "only way to learn the cause was to add logging and re-run.",
    )


class Stage3AnalysisReport(BaseModel):
    """The complete output of running the DOF graph over a
    ConstraintManifest. `square_variables` is informational (confirms what
    the derivation walk successfully pinned, even though no fact stated it
    directly). `loose_variable_probes` is what Stage 3 hands to Stage 4.
    `overconstrained_blocks` is a feasibility failure Stage 3 must flag,
    not pass along."""

    square_variables: list[str] = Field(default_factory=list)
    loose_variable_probes: list[VariableProbe] = Field(default_factory=list)
    overconstrained_blocks: list[OverconstrainedBlock] = Field(default_factory=list)
    derived_cycle_conflicts: list[CycleIssue] = Field(
        default_factory=list,
        description="Derived-column circular dependencies with no fixed "
        "point (a genuine contradiction, e.g. x = x + 5) -- see "
        "middleware/cycles.py's detect_derived_cycles.",
    )
    dismissed_conflicts: list[DismissedConflict] = Field(
        default_factory=list,
        description="Conflicts the reconciliation agent judged FALSE_POSITIVE "
        "-- kept visible for audit, not silently dropped.",
    )
    conflicts: list[Conflict] = Field(
        default_factory=list,
        description="Unresolved conflicts from the deterministic conflicts/ engine "
        "(distribution/moment/correlation/population-reconciliation/state-sequence "
        "mismatches, plus structural over-determination) after the reconciliation "
        "loop's MISEXTRACTION fixes and FALSE_POSITIVE dismissals -- what remains is "
        "either GENUINE_CONTRADICTION or exhausted its retry budget.",
    )
    unsupported: list[str] = Field(
        default_factory=list,
        description="Cases the deterministic conflicts/ engine or the cross_shard "
        "bridge could not evaluate either way (e.g. a non-chordal correlation "
        "pattern, or a predicate shape with no constraint_model equivalent yet) -- "
        "not conflicts, but not silently clean either.",
    )
    lost_shards: list[LostShard] = Field(
        default_factory=list,
        description="Shards that contributed NO constraints because their "
        "extraction failed or never passed the deterministic checker. Stage 4 "
        "consumes this object rather than the log, so without this field a run "
        "that silently lost a shard is indistinguishable from one where that "
        "shard genuinely had nothing to say -- and any metric computed over the "
        "two would be measuring different things while looking identical.",
    )

    @property
    def is_feasible(self) -> bool:
        return (
            not self.overconstrained_blocks
            and not self.derived_cycle_conflicts
            and not self.conflicts
        )

    @property
    def is_complete(self) -> bool:
        """True iff every shard actually contributed its constraints.

        Deliberately separate from `is_feasible`: a run can be perfectly
        feasible over the constraints it managed to collect while having
        quietly dropped a whole shard's worth. Feasibility asks "do these
        constraints contradict each other"; completeness asks "are these all
        the constraints there should have been".
        """
        return not self.lost_shards

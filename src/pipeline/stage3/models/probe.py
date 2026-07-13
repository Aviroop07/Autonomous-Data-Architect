"""Stage 3's final output for anything it could not resolve on its own.

Per the Stage 3/Stage 4 division of labor (project memory
stage3_stage4_division_of_labor): Stage 3 must surface what's determined,
flag genuine infeasibility, and PROBE free/unresolved things to Stage 4 --
never guess a value for them itself. This module is the probe half of
that contract; Stage 4's own handling of these probes is out of scope
here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.util.algorithms.dof_graph import OverconstrainedBlock


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


class MomentTargetProbe(BaseModel):
    """A MomentTarget fact whose derivation-chain walk bailed (design doc
    section 4.4) -- a stated population statistic Stage 3 could not
    resolve to underlying pinned parameters. Unlike VariableProbe, this
    carries the original stated target, since resolving it (if ever
    attempted) needs that value."""

    table_name: str
    column_name: str
    statistic: str
    target_value: float
    fact_references: list[int] = Field(default_factory=list)


class Stage3AnalysisReport(BaseModel):
    """The complete output of running the DOF graph over a
    ConstraintManifest. `square_variables` is informational (confirms what
    the derivation walk successfully pinned, even though no fact stated it
    directly). `loose_variable_probes` and `unresolved_moment_target_probes`
    are what Stage 3 hands to Stage 4. `overconstrained_blocks` is a
    feasibility failure Stage 3 must flag, not pass along."""

    square_variables: list[str] = Field(default_factory=list)
    loose_variable_probes: list[VariableProbe] = Field(default_factory=list)
    unresolved_moment_target_probes: list[MomentTargetProbe] = Field(
        default_factory=list
    )
    overconstrained_blocks: list[OverconstrainedBlock] = Field(default_factory=list)
    derived_cycle_conflicts: list[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of derived-column circular "
        "dependencies with no fixed point (a genuine contradiction, e.g. "
        "x = x + 5) -- see cycles.py's detect_derived_cycles.",
    )

    @property
    def is_feasible(self) -> bool:
        return not self.overconstrained_blocks and not self.derived_cycle_conflicts

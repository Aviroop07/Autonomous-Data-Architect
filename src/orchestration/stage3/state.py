"""Stage 3's in-flight state objects and the public Stage3Output.

Split out of entry.py so the phase modules (extraction, reconciliation) can
share them without importing entry, which imports THEM -- see the package
docstring for the dependency order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pydantic import BaseModel, Field

from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    StateSequenceConstraint,
    UnifiedExtractionOutput,
)
from src.pipeline.stage3.models.probe import Stage3AnalysisReport


class Stage3Output(BaseModel):
    """Complete Stage 3 output: extracted constraints + global DOF analysis."""

    distributions: List[DistributionConstraint] = Field(default_factory=list)
    moment_targets: List[Constraint] = Field(default_factory=list)
    correlations: List[CorrelatedConstraint] = Field(default_factory=list)
    structural_constraints: List[Constraint] = Field(default_factory=list)
    logic_constraints: List[Constraint] = Field(default_factory=list)
    derived_columns: List[DerivedColumnConstraint] = Field(default_factory=list)
    state_sequences: List[StateSequenceConstraint] = Field(default_factory=list)
    analysis_report: Stage3AnalysisReport = Field(default_factory=Stage3AnalysisReport)
    token_usage: int = 0

    @property
    def total_constraints(self) -> int:
        return (
            len(self.distributions)
            + len(self.moment_targets)
            + len(self.correlations)
            + len(self.structural_constraints)
            + len(self.logic_constraints)
            + len(self.derived_columns)
            + len(self.state_sequences)
        )


@dataclass
class _ShardState:
    index: int
    schema: Schema
    fact_ids: List[int]
    stub_tables: List[str]
    output: UnifiedExtractionOutput = field(default_factory=UnifiedExtractionOutput)
    tokens: int = 0


@dataclass
class _Merged:
    distributions: List[DistributionConstraint]
    moment_targets: List[Constraint]
    correlations: List[CorrelatedConstraint]
    structural: List[Constraint]
    logic: List[Constraint]
    derived: List[DerivedColumnConstraint]
    state_sequences: List[StateSequenceConstraint]


def _merge_all(shard_states: List[_ShardState]) -> _Merged:
    merged = _Merged([], [], [], [], [], [], [])
    for ss in shard_states:
        merged.distributions.extend(ss.output.distributions)
        merged.moment_targets.extend(ss.output.moment_targets)
        merged.correlations.extend(ss.output.correlations)
        merged.structural.extend(ss.output.structural_constraints)
        merged.logic.extend(ss.output.logic_constraints)
        merged.derived.extend(ss.output.derived_columns)
        merged.state_sequences.extend(ss.output.state_sequences)
    return merged

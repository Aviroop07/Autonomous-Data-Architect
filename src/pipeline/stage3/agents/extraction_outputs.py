"""LoopOutputModel wrappers for Stage 3 extraction agent outputs.

These wrap the cross_shard.py extraction outputs, adding LoopOutputModel
inheritance so the AgentLoop infrastructure can call get_errors() for
deterministic validation feedback. The cross_shard.py models themselves
are NOT modified (they remain pure data models).
"""

from __future__ import annotations

from typing import List

from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    StateSequenceConstraint,
    UnifiedExtractionOutput,
)
from src.util.constraint_model.relation.nodes import extract_base_tables
from src.util.orchestration.loop_types import LoopOutputModel


def _validate_constraint(c: Constraint, prefix: str) -> List[str]:
    errors: List[str] = []
    if not c.fact_references:
        errors.append(f"{prefix} fact_references cannot be empty.")
    if not extract_base_tables(c.on):
        errors.append(f"{prefix} ON node has no table reference.")
    if c.condition is None:
        errors.append(f"{prefix} condition is None.")
    return errors


def _validate_distribution(d: DistributionConstraint, idx: int) -> List[str]:
    prefix = f"Distribution[{idx}]"
    errors: List[str] = []
    if not d.fact_references:
        errors.append(f"{prefix} fact_references cannot be empty.")
    if not d.column.strip():
        errors.append(f"{prefix} column is empty.")
    if not extract_base_tables(d.on):
        errors.append(f"{prefix} ON node has no table reference.")
    return errors


def _validate_derived(dc: DerivedColumnConstraint, idx: int) -> List[str]:
    prefix = f"Derived[{idx}]"
    errors: List[str] = []
    if not dc.fact_references:
        errors.append(f"{prefix} fact_references cannot be empty.")
    if not dc.target_table.strip() or not dc.target_column.strip():
        errors.append(f"{prefix} target_table/target_column cannot be empty.")
    if not dc.referenced_tables:
        errors.append(f"{prefix} referenced_tables cannot be empty.")
    return errors


def _validate_correlated(c: CorrelatedConstraint, idx: int) -> List[str]:
    prefix = f"Correlation[{idx}]"
    errors: List[str] = [f"{prefix}: {e}" for e in c._validate()]
    if not extract_base_tables(c.on):
        errors.append(f"{prefix} ON node has no table reference.")
    return errors


def _validate_state_sequence(c: StateSequenceConstraint, idx: int) -> List[str]:
    prefix = f"StateSequence[{idx}]"
    errors: List[str] = [f"{prefix}: {e}" for e in c._validate()]
    if not extract_base_tables(c.on):
        errors.append(f"{prefix} ON node has no table reference.")
    return errors


class UnifiedOutput(LoopOutputModel, UnifiedExtractionOutput):
    """LoopOutputModel wrapper for the single unified constraint_generator
    agent's output. get_errors() is what the AgentLoop retry machinery
    actually reads -- see StatisticalOutput's docstring for why this
    matters (an agent's own _det_errors attribute is otherwise dead code)."""

    def get_errors(self) -> List[str]:
        errors: List[str] = list(getattr(self, "_det_errors", []))
        for i, d in enumerate(self.distributions):
            errors.extend(_validate_distribution(d, i))
        for i, c in enumerate(self.moment_targets):
            errors.extend(_validate_constraint(c, f"MomentTarget[{i}]"))
        for i, c in enumerate(self.correlations):
            errors.extend(_validate_correlated(c, i))
        for i, c in enumerate(self.structural_constraints):
            errors.extend(_validate_constraint(c, f"Structural[{i}]"))
        for i, c in enumerate(self.logic_constraints):
            errors.extend(_validate_constraint(c, f"Logic[{i}]"))
        for i, dc in enumerate(self.derived_columns):
            errors.extend(_validate_derived(dc, i))
        for i, c in enumerate(self.state_sequences):
            errors.extend(_validate_state_sequence(c, i))
        return errors


class AuditReport(LoopOutputModel):
    """Output shape for the constraint auditor -- a second,
    independently-prompted LLM re-reading the original NL facts against
    the extractor's structured output, catching what canonicalize()
    structurally can't (a dropped condition, wrong-table attribution, a
    hallucinated bound, missed conditionality). Mirrors Stage 1's
    IntegrityReport / Stage 2's ConceptualCritiqueReport role, simplified
    to one issues list since Stage 3's extractors don't need the
    missing/introduced/changed/ambiguous taxonomy those carry."""

    is_valid: bool
    issues: List[str] = []
    reasoning: str = ""

    def get_errors(self) -> List[str]:
        # Auditor feedback is consumed directly by the extractor's own
        # build_context() (via ctx.node_outputs, matching Stage 1's
        # fact_extractor <-> verifier convention) -- not through this
        # generic det_errors channel, which is reserved for the
        # extractor's own deterministic canonicalize() errors. This stays
        # empty so an invalid audit doesn't ALSO retry via the det_errors
        # path (the graph's own conditional edge on is_valid handles that).
        return []

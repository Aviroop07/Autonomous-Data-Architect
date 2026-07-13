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
    DerivedColumnConstraint,
    DistributionConstraint,
    LogicExtractionOutput,
    StatisticalExtractionOutput,
    StructuralExtractionOutput,
)
from src.pipeline.stage3.models.on_nodes import extract_tables
from src.util.orchestration.loop_types import LoopOutputModel


def _validate_constraint(c: Constraint, prefix: str) -> List[str]:
    errors: List[str] = []
    if not c.fact_references:
        errors.append(f"{prefix} fact_references cannot be empty.")
    if not extract_tables(c.on):
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
    if not extract_tables(d.on):
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


class StatisticalOutput(LoopOutputModel, StatisticalExtractionOutput):
    """LoopOutputModel wrapper for statistical extraction output.

    get_errors() is what the AgentLoop infrastructure actually calls to
    drive retries (it always overwrites state.det_errors from this method's
    return value -- an agent's own canonicalize()-computed _det_errors
    attribute, set in invoke(), is otherwise silently discarded). Reading
    it back here is what makes the real FK-PK canonicalization check
    (extractor agent.py's _validate_output) actually reach the retry loop
    instead of being dead code."""

    def get_errors(self) -> List[str]:
        errors: List[str] = list(getattr(self, "_det_errors", []))
        for i, d in enumerate(self.distributions):
            errors.extend(_validate_distribution(d, i))
        for i, c in enumerate(self.moment_targets):
            errors.extend(_validate_constraint(c, f"MomentTarget[{i}]"))
        for i, c in enumerate(self.correlations):
            errors.extend(_validate_constraint(c, f"Correlation[{i}]"))
        return errors


class StructuralOutput(LoopOutputModel, StructuralExtractionOutput):
    """LoopOutputModel wrapper for structural extraction output."""

    def get_errors(self) -> List[str]:
        errors: List[str] = list(getattr(self, "_det_errors", []))
        for i, c in enumerate(self.constraints):
            errors.extend(_validate_constraint(c, f"Structural[{i}]"))
        return errors


class LogicOutput(LoopOutputModel, LogicExtractionOutput):
    """LoopOutputModel wrapper for logic extraction output."""

    def get_errors(self) -> List[str]:
        errors: List[str] = list(getattr(self, "_det_errors", []))
        for i, c in enumerate(self.constraints):
            errors.extend(_validate_constraint(c, f"Logic[{i}]"))
        for i, dc in enumerate(self.derived):
            errors.extend(_validate_derived(dc, i))
        return errors


class AuditReport(LoopOutputModel):
    """Shared output shape for the 3 family-specific auditor agents
    (statistical/structural/logic). Each auditor gets its own tailored
    prompt (different semantic failure modes per family), but the
    structured output they produce is the same shape -- a second,
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

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
        return errors

"""LoopOutputModel wrappers for Stage 3 extraction agent outputs.

These wrap the cross_shard.py extraction outputs, adding LoopOutputModel
inheritance so the AgentLoop infrastructure can call get_errors() for
deterministic validation feedback. The cross_shard.py models themselves
are NOT modified (they remain pure data models).
"""

from __future__ import annotations

from typing import List, Optional

from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    LogicExtractionOutput,
    StatisticalExtractionOutput,
    StructuralExtractionOutput,
)
from src.util.orchestration.loop_types import LoopOutputModel


def _validate_constraint(c: Constraint, prefix: str) -> List[str]:
    errors: List[str] = []
    if not c.fact_references:
        errors.append(f"{prefix} fact_references cannot be empty.")
    on = c.on
    table = getattr(on, "table", None)
    if table is None:
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
    table = getattr(d.on, "table", None)
    if table is None:
        errors.append(f"{prefix} ON node has no table reference.")
    return errors


class StatisticalOutput(LoopOutputModel, StatisticalExtractionOutput):
    """LoopOutputModel wrapper for statistical extraction output."""

    def get_errors(self) -> List[str]:
        errors: List[str] = []
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
        errors: List[str] = []
        for i, c in enumerate(self.constraints):
            errors.extend(_validate_constraint(c, f"Structural[{i}]"))
        return errors


class LogicOutput(LoopOutputModel, LogicExtractionOutput):
    """LoopOutputModel wrapper for logic extraction output."""

    def get_errors(self) -> List[str]:
        errors: List[str] = []
        for i, c in enumerate(self.constraints):
            errors.extend(_validate_constraint(c, f"Logic[{i}]"))
        return errors

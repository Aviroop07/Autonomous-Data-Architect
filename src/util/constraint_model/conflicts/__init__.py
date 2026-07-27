"""Deterministic, LLM-free conflict detection over a Constraint set (Section
6, 8.2.1's chordal/PD-completion, 9.2's StateSequence graph merge). Exposes
a plain ConflictReport for a FUTURE reconciliation loop to consume and
judge -- this package deliberately makes no MISEXTRACTION/FALSE_POSITIVE/
GENUINE_CONTRADICTION/SOFTEN judgment itself, per explicit instruction:
build the evaluation API now, wire it into an LLM loop later.

Public entry point: evaluate.evaluate_constraints(constraints, schema).
"""

from src.util.constraint_model.conflicts.evaluate import evaluate_constraints
from src.util.constraint_model.conflicts.models import (
    Conflict,
    ConflictKind,
    ConflictReport,
)

__all__ = ["evaluate_constraints", "Conflict", "ConflictKind", "ConflictReport"]

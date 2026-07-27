"""Deterministic canonicalize() checker -- the middle node of Stage 3's
3-node per-shard loop (generator -> deterministic_checker -> auditor).

Pulled out of the generator's own invoke() (where the 3 old family
extractors each ran it internally) into its own explicit LoopAgent, per
the Stage 3 redesign: "one generator, one deterministic checker, one LLM
auditor in the loop." A 0-token LoopAgent is directly supported by the
existing AgentLoop framework -- LoopAgent.invoke()'s own docstring notes
"Token count is 0 for deterministic validator nodes."

State (the pending output to check, and the schema to check it against)
is threaded through instance fields set in build_context() and read in
invoke() -- the same pattern the old extractors used internally for their
own _last_schema/_validate_output split, now made an explicit node.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.agents.extraction_outputs import UnifiedOutput
from src.util.constraint_model.condition.predicates import extract_columns
from src.pipeline.stage3.models.cross_shard import Constraint as CSConstraint
from src.pipeline.stage3.models.cross_shard import CorrelatedConstraint
from src.pipeline.stage3.models.cross_shard import DistributionConstraint
from src.pipeline.stage3.models.cross_shard import StateSequenceConstraint
from src.pipeline.stage3.models.grain import (
    CanonicalizationFailure,
    _SchemaView,
    canonicalize,
)
from src.pipeline.stage3.models.on_sql_normalize import normalize_on
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

logger = logging.getLogger(__name__)


class DetCheckOutput(LoopOutputModel):
    """Output of the deterministic checker node. `errors` IS get_errors()'s
    return value -- this node's whole purpose is to produce that list."""

    errors: List[str] = []

    def get_errors(self) -> List[str]:
        return self.errors


def _columns_to_validate(item: object) -> List[str]:
    """The set of column names `item` references outside its own `on` tree
    -- i.e. everything that must resolve against the ON tree's accessible
    columns once canonicalize() has reduced it to a Grain. Differs by
    constraint shape since each has a different set of column-bearing
    fields (a generic Constraint's `condition`, a DistributionConstraint's
    `column`/`if_condition`, a CorrelatedConstraint's `columns`, a
    StateSequenceConstraint's `sequence_column`)."""
    cols: set[str] = set()
    if isinstance(item, DistributionConstraint):
        cols.add(item.column)
        if item.if_condition is not None:
            cols.update(extract_columns(item.if_condition))
    elif isinstance(item, CorrelatedConstraint):
        cols.update(item.columns)
    elif isinstance(item, StateSequenceConstraint):
        cols.add(item.sequence_column)
    elif isinstance(item, CSConstraint):
        cols.update(extract_columns(item.condition))
    return sorted(cols)


class DeterministicCheckerLoopAgent(LoopAgent):
    """LoopAgent for the deterministic ON-tree canonicalization node."""

    def __init__(self) -> None:
        self._pending_output: Optional[UnifiedOutput] = None
        self._schema: Optional[Schema] = None

    def _canonicalize_list(
        self, items: list, label: str, schema: Schema, view: _SchemaView
    ) -> List[str]:
        """Normalizes (replacing any ONSubquery with its structured
        equivalent, in place on `item.on`), canonicalizes every item's ON
        tree, and -- only once canonicalization succeeds, since there's no
        Grain to check against otherwise -- validates that every column
        the item's own fields reference (condition, if_condition, column,
        columns, partition_by, sequence_column, order_by) actually resolves
        unambiguously against that Grain. One error string per failure."""
        errors: List[str] = []
        for i, item in enumerate(items):
            normalized, norm_err = normalize_on(item.on)
            if normalized is None:
                errors.append(f"{label}[{i}] ON normalization failed: {norm_err}")
                continue
            item.on = normalized
            result = canonicalize(item.on, schema)
            if isinstance(result, CanonicalizationFailure):
                errors.append(
                    f"{label}[{i}] ON canonicalization failed: {result.reason}"
                )
                continue
            for col in _columns_to_validate(item):
                col_err = result.validate_column(col, view)
                if col_err is not None:
                    errors.append(f"{label}[{i}] column '{col}' invalid: {col_err}")
        return errors

    def _canonicalize_all(self, output: UnifiedOutput, schema: Schema) -> List[str]:
        view = _SchemaView.from_schema(schema)
        errors: List[str] = []
        errors.extend(
            self._canonicalize_list(output.distributions, "Distribution", schema, view)
        )
        errors.extend(
            self._canonicalize_list(output.moment_targets, "MomentTarget", schema, view)
        )
        errors.extend(
            self._canonicalize_list(output.correlations, "Correlation", schema, view)
        )
        errors.extend(
            self._canonicalize_list(
                output.structural_constraints, "Structural", schema, view
            )
        )
        errors.extend(
            self._canonicalize_list(output.logic_constraints, "Logic", schema, view)
        )
        errors.extend(
            self._canonicalize_list(
                output.state_sequences, "StateSequence", schema, view
            )
        )
        return errors

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        del query  # deterministic node -- state comes from build_context, not the query
        if self._pending_output is None or self._schema is None:
            return DetCheckOutput(errors=[]), 0
        errors = self._canonicalize_all(self._pending_output, self._schema)
        return DetCheckOutput(errors=errors), 0

    def build_context(self, ctx: LoopContext) -> str:
        generator_output = ctx.node_outputs.get("generator")
        self._pending_output = (
            generator_output if isinstance(generator_output, UnifiedOutput) else None
        )

        context_data: Dict[str, Any] = {}
        try:
            context_data = json.loads(ctx.initial_context)
        except json.JSONDecodeError:
            logger.warning(
                "[DeterministicChecker] Failed to parse initial_context as JSON."
            )
        except TypeError, AttributeError:
            logger.warning(
                "[DeterministicChecker] initial_context was not a JSON string."
            )

        schema_raw = context_data.get("schema")
        schema: Optional[Schema] = None
        if isinstance(schema_raw, Schema):
            schema = schema_raw
        elif isinstance(schema_raw, dict):
            try:
                schema = Schema(**schema_raw)
            except Exception:
                pass
        self._schema = schema

        return "deterministic canonicalize() pass over the generator's latest output"

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, DetCheckOutput)
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=(
                "canonicalization clean"
                if not output.errors
                else f"{len(output.errors)} canonicalization error(s)"
            ),
            was_improvement=None,
            tokens=0,
        )

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

from src.util.schema_model.schema import Schema
from src.pipeline.stage3.agents.extraction_outputs import UnifiedOutput
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import (
    RComparison,
    RPredicateUnion,
    extract_columns,
)
from src.util.constraint_model.relation.nodes import Aggregate
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


# Columns whose domain is a COUNT, and therefore provably >= 0. `child_count`
# is the synthetic column a Fanout exposes (see grain.py's
# COUNT_CHILDREN_LEFT_JOIN signature); a COUNT aggregate's alias is added
# per-item below, since it depends on the ON tree.
_INHERENTLY_NON_NEGATIVE = {"child_count"}


def _non_negative_columns(item: object) -> set[str]:
    """Columns in `item`'s scope that cannot be negative: the fanout's
    child_count, plus the alias of any COUNT aggregate in its ON tree."""
    cols = set(_INHERENTLY_NON_NEGATIVE)
    node = getattr(item, "on", None)
    if isinstance(node, Aggregate) and node.fn == "COUNT":
        cols.add(node.alias)
    return cols


def _vacuous_comparisons(pred: "RPredicateUnion", non_negative: set[str]) -> List[str]:
    """Comparisons that EVERY possible value satisfies, so they constrain
    nothing while still canonicalizing cleanly and consuming a degree of
    freedom downstream.

    Caught live: a fanout emitted `child_count >= 0`. A count is always >= 0,
    so that asserts nothing, yet it validated fine (child_count is a real
    column at a fanout grain) and became a DOF variable handed to Stage 4.
    This was previously only discouraged by a prompt rule -- i.e. the model
    policing itself -- which is the wrong layer for a purely mechanical check.
    """
    errors: List[str] = []

    def walk(node: object) -> None:
        if isinstance(node, RComparison):
            left, right = node.left, node.right
            if isinstance(left, RColumnRef) and isinstance(right, RLiteral):
                name, value = left.name, right.value
                if name in non_negative and isinstance(value, (int, float)):
                    if (
                        (node.op == ">=" and value <= 0)
                        or (node.op == ">" and value < 0)
                        or (node.op == "!=" and value < 0)
                    ):
                        errors.append(
                            f"'{name} {node.op} {value}' is vacuous -- {name} is a "
                            f"count and can never be negative, so this asserts "
                            f"nothing. State the real bound (e.g. '> 1' for "
                            f"'multiple', '>= 1' for 'at least one') or emit no "
                            f"constraint."
                        )
        for attr in ("operands",):
            for child in getattr(node, attr, []) or []:
                walk(child)
        for attr in ("operand", "antecedent", "consequent"):
            child = getattr(node, attr, None)
            if child is not None:
                walk(child)

    walk(pred)
    return errors


def _conditions_of(item: object) -> List["RPredicateUnion"]:
    """Every predicate tree `item` carries, whatever its shape."""
    out: List["RPredicateUnion"] = []
    cond = getattr(item, "condition", None)
    if cond is not None:
        out.append(cond)
    if_cond = getattr(item, "if_condition", None)
    if if_cond is not None:
        out.append(if_cond)
    return out


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
            non_negative = _non_negative_columns(item)
            for cond in _conditions_of(item):
                for msg in _vacuous_comparisons(cond, non_negative):
                    errors.append(f"{label}[{i}] vacuous constraint: {msg}")
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
        errors.extend(self._check_derived_columns(output.derived_columns, view))
        return errors

    def _check_derived_columns(self, items: list, view: _SchemaView) -> List[str]:
        """Validate derived columns against the schema.

        DerivedColumnConstraint carries no `on` tree, so it cannot go through
        _canonicalize_list -- which is why it was the one output list checked by
        nothing at all. Its table references are still checkable, and a
        derivation naming a table the shard does not contain is exactly the kind
        of hallucination the deterministic pass exists to catch before it
        reaches the DOF graph.
        """
        errors: List[str] = []
        for index, item in enumerate(items):
            target = getattr(item, "target_table", None)
            if target and target not in view.tables:
                errors.append(
                    f"DerivedColumn[{index}]: target_table '{target}' is not a table "
                    f"in this shard's schema."
                )
            for referenced in getattr(item, "referenced_tables", []) or []:
                if referenced not in view.tables:
                    errors.append(
                        f"DerivedColumn[{index}]: referenced_tables names "
                        f"'{referenced}', which is not a table in this shard's schema."
                    )
        return errors

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        del query  # deterministic node -- state comes from build_context, not the query
        if self._schema is None:
            # An empty error list here means "checked and clean", and the loop
            # acts on it by advancing to the auditor. Not being ABLE to check is
            # a different thing, and build_context has already logged why at
            # ERROR level. Returning empty is still the right routing decision
            # -- sending the generator back to retry cannot conjure a schema it
            # was never given -- but it must not be silent.
            logger.warning(
                "[DeterministicChecker] No usable schema; reporting no errors "
                "WITHOUT having checked anything. This is not a clean pass."
            )
            return DetCheckOutput(errors=[]), 0
        if self._pending_output is None:
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
        except (TypeError, AttributeError):
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
            except Exception as exc:
                # Was a bare `pass`. Losing the schema here disables EVERY
                # check this node performs, and invoke() then returned an empty
                # error list -- indistinguishable from a genuine clean pass, so
                # the loop routed straight on to the auditor. Silence was the
                # worst possible response to it.
                logger.error(
                    "[DeterministicChecker] Could not reconstruct the shard Schema "
                    "from context (%s: %s). Every canonicalization, column-resolution "
                    "and vacuous-bound check will be SKIPPED for this round.",
                    type(exc).__name__,
                    exc,
                )
        elif schema_raw is not None:
            logger.error(
                "[DeterministicChecker] Context 'schema' has unexpected type %s; "
                "expected Schema or dict. All checks will be SKIPPED this round.",
                type(schema_raw).__name__,
            )
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

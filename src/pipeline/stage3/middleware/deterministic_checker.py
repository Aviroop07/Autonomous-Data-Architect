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

import logging
from typing import List, Optional

from pydantic import BaseModel

from src.pipeline.stage3.agents.extraction_outputs import UnifiedOutput
from src.pipeline.stage3.models.shard_context import Stage3ShardContext
from src.util.schema_model.schema import Schema
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


class RejectedItem(BaseModel):
    """One extraction item the deterministic checker refused, identified
    structurally rather than by parsing its error string.

    `list_name` is the UnifiedOutput attribute holding the item, so a caller
    can act on the rejection -- dropping just the offender -- without knowing
    how the message was formatted. Before this existed, the only record of
    WHICH item failed was the "{Label}[{index}]" prefix inside a free-text
    error, so the whole shard had to be withheld when any single item failed.
    """

    label: str
    index: int
    list_name: str
    reason: str

    @property
    def message(self) -> str:
        return f"{self.label}[{self.index}] {self.reason}"


class DetCheckOutput(LoopOutputModel):
    """Output of the deterministic checker node. `errors` IS get_errors()'s
    return value -- this node's whole purpose is to produce that list."""

    errors: List[str] = []
    rejected: List[RejectedItem] = []

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
        self,
        items: list,
        label: str,
        schema: Schema,
        view: _SchemaView,
        list_name: str = "",
        rejected: Optional[List[RejectedItem]] = None,
    ) -> List[str]:
        """Normalizes (replacing any ONSubquery with its structured
        equivalent, in place on `item.on`), canonicalizes every item's ON
        tree, and -- only once canonicalization succeeds, since there's no
        Grain to check against otherwise -- validates that every column
        the item's own fields reference (condition, if_condition, column,
        columns, partition_by, sequence_column, order_by) actually resolves
        unambiguously against that Grain. One error string per failure."""
        errors: List[str] = []

        def reject(index: int, reason: str) -> None:
            """Record one rejection in both forms: the free-text error the retry
            loop feeds back to the generator, and the structured record a caller
            needs to drop just this item instead of the whole shard."""
            errors.append(f"{label}[{index}] {reason}")
            if rejected is not None:
                rejected.append(
                    RejectedItem(
                        label=label,
                        index=index,
                        list_name=list_name,
                        reason=reason,
                    )
                )

        for i, item in enumerate(items):
            normalized, norm_err = normalize_on(item.on)
            if normalized is None:
                reject(i, f"ON normalization failed: {norm_err}")
                continue
            item.on = normalized
            result = canonicalize(item.on, schema)
            if isinstance(result, CanonicalizationFailure):
                reject(i, f"ON canonicalization failed: {result.reason}")
                continue
            for col in _columns_to_validate(item):
                col_err = result.validate_column(col, view)
                if col_err is not None:
                    reject(i, f"column '{col}' invalid: {col_err}")
            non_negative = _non_negative_columns(item)
            for cond in _conditions_of(item):
                for msg in _vacuous_comparisons(cond, non_negative):
                    reject(i, f"vacuous constraint: {msg}")
        return errors

    # (label, UnifiedOutput attribute) for every list this node checks. Kept in
    # one place so a new extraction shape cannot be added to UnifiedOutput and
    # silently go unchecked, which is how derived_columns ended up validated by
    # nothing at all.
    _CHECKED_LISTS: List[tuple[str, str]] = [
        ("Distribution", "distributions"),
        ("MomentTarget", "moment_targets"),
        ("Correlation", "correlations"),
        ("Structural", "structural_constraints"),
        ("Logic", "logic_constraints"),
        ("StateSequence", "state_sequences"),
    ]

    def _canonicalize_all(
        self,
        output: UnifiedOutput,
        schema: Schema,
        rejected: Optional[List[RejectedItem]] = None,
    ) -> List[str]:
        view = _SchemaView.from_schema(schema)
        errors: List[str] = []
        for label, list_name in self._CHECKED_LISTS:
            errors.extend(
                self._canonicalize_list(
                    getattr(output, list_name),
                    label,
                    schema,
                    view,
                    list_name=list_name,
                    rejected=rejected,
                )
            )
        errors.extend(
            self._check_derived_columns(output.derived_columns, view, rejected)
        )
        return errors

    def _check_derived_columns(
        self,
        items: list,
        view: _SchemaView,
        rejected: Optional[List[RejectedItem]] = None,
    ) -> List[str]:
        """Validate derived columns against the schema.

        DerivedColumnConstraint carries no `on` tree, so it cannot go through
        _canonicalize_list -- which is why it was the one output list checked by
        nothing at all. Its table references are still checkable, and a
        derivation naming a table the shard does not contain is exactly the kind
        of hallucination the deterministic pass exists to catch before it
        reaches the DOF graph.
        """
        errors: List[str] = []

        def reject(index: int, reason: str) -> None:
            errors.append(f"DerivedColumn[{index}]: {reason}")
            if rejected is not None:
                rejected.append(
                    RejectedItem(
                        label="DerivedColumn",
                        index=index,
                        list_name="derived_columns",
                        reason=reason,
                    )
                )

        for index, item in enumerate(items):
            target = getattr(item, "target_table", None)
            if target and target not in view.tables:
                reject(
                    index,
                    f"target_table '{target}' is not a table in this shard's schema.",
                )
            for referenced in getattr(item, "referenced_tables", []) or []:
                if referenced not in view.tables:
                    reject(
                        index,
                        f"referenced_tables names '{referenced}', which is not a "
                        f"table in this shard's schema.",
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
        rejected: List[RejectedItem] = []
        errors = self._canonicalize_all(self._pending_output, self._schema, rejected)
        return DetCheckOutput(errors=errors, rejected=rejected), 0

    def build_context(self, ctx: LoopContext[Stage3ShardContext]) -> str:
        generator_output = ctx.node_outputs.get("generator")
        self._pending_output = (
            generator_output if isinstance(generator_output, UnifiedOutput) else None
        )

        ctx_data = ctx.initial_context
        if isinstance(ctx_data, Stage3ShardContext):
            self._schema = ctx_data.shard_schema
        else:
            # Not being able to check is not the same as a clean pass. This
            # guard was previously a Type-3 Schema-reconstruction ladder that
            # existed only to survive the JSON round-trip (now eliminated).
            self._schema = None
            logger.error(
                "[DeterministicChecker] initial_context is %s, not "
                "Stage3ShardContext. Every canonicalization, column-resolution "
                "and vacuous-bound check will be SKIPPED for this round.",
                type(ctx_data).__name__,
            )

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

"""Phase 1: the per-shard generator -> deterministic_checker -> auditor loop.

Building the loop's graph is separate from running it so that Phase 1's bulk
fan-out and a targeted misextraction rerun can share the same configuration
while using different execution strategies -- see _rerun_shard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage3.models.shard_context import Stage3ShardContext
from src.util.schema_model.schema import Schema
from src.pipeline.stage3.agents.constraint_auditor.agent import (
    ConstraintAuditorLoopAgent,
)
from src.pipeline.stage3.agents.constraint_generator.agent import (
    ConstraintGeneratorLoopAgent,
)
from src.pipeline.stage3.middleware.deterministic_checker import (
    DeterministicCheckerLoopAgent,
)
from src.pipeline.stage3.models.cross_shard import UnifiedExtractionOutput
from src.pipeline.stage3.models.probe import LostShardReason
from src.orchestration.stage3.context import _build_shard_context
from src.util.core.agent_provider import AgentProvider
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.rounds import rounds_to_max_iter as _rounds_to_max_iter
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    ErrorRefreshConfig,
    GraphEdge,
    LoopConfig,
    LoopResult,
)
from typing import Dict, List

logger = logging.getLogger(__name__)


# AgentLoop's retry budget is consumed once per NODE EXECUTION, not once per
# pass through the graph (see util/orchestration/loop.py's `while
# budget.try_consume()`). The generator graph has three nodes, so one full
# audited round -- generator, then det_checker, then auditor -- costs three
# units. Converting rounds to raw iterations anywhere other than here is how
# Phase 1 ended up with max_iter=3, which afforded exactly one pass and made
# the det_checker->generator and auditor->generator retry edges unreachable:
# validation errors and audit findings were computed, logged, then discarded.
#
# A test asserts this equals the real node count so it cannot drift if a node
# is added to the graph.
GENERATOR_GRAPH_NODE_COUNT = 3


def rounds_to_max_iter(rounds: int) -> int:
    """Convert "N full audited rounds" into the raw per-node iteration budget
    AgentLoop actually counts. `rounds=1` means one pass and no retry."""
    return _rounds_to_max_iter(rounds, GENERATOR_GRAPH_NODE_COUNT)


def _build_generator_loop_config(
    max_iter: int,
    model: Optional[str],
    provider: Optional[AgentProvider] = None,
) -> LoopConfig:
    """Builds one shard's generator -> deterministic_checker -> auditor
    LoopConfig.

    Graph shape:
        generator -----------------------------------> det_checker
        det_checker --[has canonicalize errors]------> generator  (retry)
        det_checker -----------------------------------> auditor
        auditor --[is_valid == False]------------------> generator (retry)
        auditor -----------------------------------------> end

    Edge order matters -- AgentLoop._evaluate_edges is first-match-wins.

    Split out from execution (a separate function used to just call
    AgentLoop(config).run(...) directly) so Phase 1 extraction and
    misextraction reruns can batch many shards' configs through
    run_parallel_loops() in one call -- see util/orchestration/
    parallel_loop.py's module docstring for why that matters: failure
    isolation (one shard's parse hiccup no longer crashes every other
    shard's in-progress work) and cross-shard retry-budget reallocation.

    Takes the raw `max_iter` directly (not a "logical retries" count) --
    callers that want the old "N full audited rounds" semantics convert
    themselves (see _run_generator_loop's `max_retries * 4`); Phase 1's
    own construction in orchestrate() passes its auto-scaled raw value
    directly, since its policy differs from reruns/diagnostics.
    """
    generator = ConstraintGeneratorLoopAgent(model=model, provider=provider)
    checker = DeterministicCheckerLoopAgent()
    auditor = ConstraintAuditorLoopAgent(model=model, provider=provider)

    return LoopConfig(
        agents={
            "generator": AgentRoleConfig(
                agent_factory=lambda: generator,
                det_error_sources=["det_checker"],
            ),
            "det_checker": AgentRoleConfig(agent_factory=lambda: checker),
            "auditor": AgentRoleConfig(agent_factory=lambda: auditor),
        },
        graph={
            "edges": [
                GraphEdge(from_node="generator", to_node="det_checker"),
                GraphEdge(
                    from_node="det_checker",
                    to_node="generator",
                    condition=EdgeCondition(
                        fn=lambda o: bool(getattr(o, "errors", []))
                    ),
                ),
                GraphEdge(from_node="det_checker", to_node="auditor"),
                GraphEdge(
                    from_node="auditor",
                    to_node="generator",
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node="auditor", to_node="end"),
            ]
        },
        start_node="generator",
        max_iter=max_iter,
        error_refresh=ErrorRefreshConfig(trigger_node="generator"),
    )


def _count_constraints(output: UnifiedExtractionOutput) -> int:
    """Total constraints across every shape the extraction output carries.

    Derived from `model_fields` rather than a hardcoded field list, so adding a
    new constraint shape to UnifiedExtractionOutput cannot silently stop being
    counted -- the same class of drift that made AtomicFact.from_raw lose two
    fields.
    """
    total = 0
    for name in type(output).model_fields:
        value = getattr(output, name, None)
        if isinstance(value, list):
            total += len(value)
    return total


def _drop_rejected(output: UnifiedExtractionOutput, rejected: list) -> int:
    """Remove exactly the items the deterministic checker refused, in place.

    Returns how many were dropped. Deletes by descending index within each list
    so earlier indices stay valid as later ones are removed -- rejections carry
    the index the checker saw, and removing front-to-back would shift the rest.

    Several rejections can name the SAME item (one constraint can fail column
    resolution and be vacuous at once), hence the de-duplication: double-counting
    would misreport how much was dropped.
    """
    by_list: dict[str, set[int]] = {}
    for item in rejected:
        list_name = getattr(item, "list_name", "")
        index = getattr(item, "index", None)
        if not list_name or index is None:
            continue
        by_list.setdefault(list_name, set()).add(int(index))

    dropped = 0
    for list_name, indices in by_list.items():
        target = getattr(output, list_name, None)
        if not isinstance(target, list):
            continue
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(target):
                del target[index]
                dropped += 1
    return dropped


@dataclass(frozen=True)
class ShardExtractionResult:
    """What one shard's generator loop produced, plus whether it was lost.

    A plain (output, tokens) tuple could not carry the "this shard contributed
    nothing, and here is why" signal, which is exactly the information that was
    previously confined to a log line.
    """

    output: UnifiedExtractionOutput
    tokens: int
    lost_reason: Optional[LostShardReason] = None
    detail: str = ""
    withheld_constraint_count: int = 0
    # The deterministic checker's unresolved complaints -- WHY the shard was
    # withheld. Defaulted to an empty tuple rather than a list so the dataclass
    # stays frozen-hashable and no caller can mutate a shared default.
    errors: tuple[str, ...] = ()


def _extract_generator_output(
    result: Optional["LoopResult"],
) -> ShardExtractionResult:
    """Pulls the generator node's final output out of a completed LoopResult.

    `result` is None when run_parallel_loops() isolated a failure for this unit
    (see its own docstring) -- treated as an empty output rather than
    propagating the failure, so one shard's crash cannot take the run down.

    A shard that produced nothing is now REPORTED rather than only logged, and
    a shard whose deterministic checker never converged has its constraints
    WITHHELD rather than shipped: they failed canonicalization or column
    resolution, so they may reference columns that do not exist, and Stage 4
    generating data against them is worse than Stage 4 not having them. Either
    way the caller learns which facts went unrepresented.
    """
    if result is None:
        return ShardExtractionResult(
            output=UnifiedExtractionOutput(),
            tokens=0,
            lost_reason=LostShardReason.EXTRACTION_FAILED,
            detail="run_parallel_loops isolated a failure for this shard.",
        )
    output = result.node_outputs.get("generator")
    if not isinstance(output, UnifiedExtractionOutput):
        logger.error(
            "[Stage 3] Discarding a shard's extraction: the generator node's "
            "final output was %s, not UnifiedExtractionOutput. This shard "
            "contributes NO constraints.",
            type(output).__name__,
        )
        return ShardExtractionResult(
            output=UnifiedExtractionOutput(),
            tokens=result.total_tokens,
            lost_reason=LostShardReason.EXTRACTION_FAILED,
            detail=(
                f"generator produced {type(output).__name__}, not "
                "UnifiedExtractionOutput."
            ),
        )
    checker_output = result.node_outputs.get("det_checker")
    checker_errors: tuple[str, ...] = tuple(
        str(e) for e in (getattr(checker_output, "errors", None) or [])
    )
    rejected = list(getattr(checker_output, "rejected", None) or [])

    # Drop ONLY the items the checker actually refused, and do it regardless of
    # whether the retry budget was exhausted. A constraint the deterministic
    # checker refused on the final pass must not ship either way -- it may
    # reference a column that does not exist, and Stage 4 generating data
    # against it is worse than Stage 4 not having it.
    #
    # This used to withhold the ENTIRE shard, which on a live
    # university_hospital_messy run meant ZERO constraints out of 159k tokens
    # because ONE of three was vacuous (`child_count >= 0`, which cannot be
    # false since a count is never negative). The other two were fine. The
    # checker always knew which item it refused; that was simply never expressed
    # in a form a caller could act on.
    dropped = _drop_rejected(output, rejected)
    if dropped and not result.det_errors_exhausted:
        logger.warning(
            "[Stage 3] Dropping %d constraint(s) the deterministic checker "
            "refused on its final pass; %d survive. Refused: %s",
            dropped,
            _count_constraints(output),
            checker_errors,
        )

    if result.det_errors_exhausted:
        if dropped and _count_constraints(output):
            logger.error(
                "[Stage 3] Shard extraction exhausted its retry budget after "
                "%d iteration(s). DROPPING the %d constraint(s) the "
                "deterministic checker refused and keeping the %d that passed. "
                "Refused: %s",
                result.iteration_count,
                dropped,
                _count_constraints(output),
                checker_errors,
            )
            return ShardExtractionResult(
                output=output,
                tokens=result.total_tokens,
                withheld_constraint_count=dropped,
                errors=checker_errors,
            )

        # Nothing survived (or the checker gave no actionable rejections), so
        # the shard really did contribute nothing.
        withheld = dropped or _count_constraints(output)
        logger.error(
            "[Stage 3] Shard extraction exhausted its retry budget with "
            "UNRESOLVED deterministic errors after %d iteration(s); "
            "WITHHOLDING its %d constraint(s) rather than shipping constraints "
            "that never passed canonicalization or column resolution. "
            "Unresolved: %s",
            result.iteration_count,
            withheld,
            checker_errors or "(the checker reported none on its final pass)",
        )
        return ShardExtractionResult(
            output=UnifiedExtractionOutput(),
            tokens=result.total_tokens,
            lost_reason=LostShardReason.DETERMINISTIC_CHECK_UNRESOLVED,
            detail=(
                f"deterministic checker still reporting errors after "
                f"{result.iteration_count} iteration(s)."
            ),
            withheld_constraint_count=withheld,
            errors=checker_errors,
        )
    return ShardExtractionResult(output=output, tokens=result.total_tokens)


async def _run_generator_loop(
    context: Stage3ShardContext,
    model: Optional[str],
    max_retries: int,
    provider: Optional[AgentProvider] = None,
) -> ShardExtractionResult:
    """Runs one shard's generator loop directly (no cross-shard budget
    sharing) -- kept for any single-shard caller, e.g. isolated diagnostic
    scripts, and for shard reruns (_rerun_shard). Live Phase 1 extraction
    goes through run_parallel_loops() instead, with its own auto-scaled
    raw max_iter (see orchestrate()'s Phase 1 construction) rather than
    this function's "N full audited rounds" policy."""
    config = _build_generator_loop_config(
        rounds_to_max_iter(max_retries), model, provider
    )
    result = await AgentLoop(config).run(context)
    return _extract_generator_output(result)


async def _rerun_shard(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    guidance: str,
    model: Optional[str],
    max_retries: int,
    provider: Optional[AgentProvider] = None,
) -> ShardExtractionResult:
    """Re-run a shard's WHOLE generator loop with reconciliation guidance
    injected -- there is only one generator now, so a MISEXTRACTION fix
    always targets the owning shard's single loop, not a specific family.

    Deliberately kept as its own function (rather than inlined into
    _build_generator_loop_config + run_parallel_loops like Phase 1's
    extraction fan-out) and run through the simpler run_parallel(), not
    run_parallel_loops() -- misextraction reruns are a small, targeted
    fan-out (only the shards a reconciliation round actually flagged),
    so cross-shard retry-budget reallocation matters far less here than
    for Phase 1's bulk extraction. Each rerun still gets its own
    independent budget.

    Kept as a named module-level function because the reconciliation module
    calls it by name; the LLM boundary itself is injected via `provider`
    (util/core/agent_provider.py), so a test does not need to replace this
    function to run offline.
    """
    context = _build_shard_context(
        shard, fact_ids, facts_map, stub_tables, reconciliation_guidance=guidance
    )
    return await _run_generator_loop(context, model, max_retries, provider)

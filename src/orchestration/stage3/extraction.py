"""Phase 1: the per-shard generator -> deterministic_checker -> auditor loop.

Building the loop's graph is separate from running it so that Phase 1's bulk
fan-out and a targeted misextraction rerun can share the same configuration
while using different execution strategies -- see _rerun_shard.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
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
from src.orchestration.stage3.context import _serialize_context
from src.util.orchestration.loop import AgentLoop
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
    return max(1, rounds) * GENERATOR_GRAPH_NODE_COUNT


def _build_generator_loop_config(max_iter: int, model: Optional[str]) -> LoopConfig:
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
    generator = ConstraintGeneratorLoopAgent(model=model)
    checker = DeterministicCheckerLoopAgent()
    auditor = ConstraintAuditorLoopAgent(model=model)

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


def _extract_generator_output(
    result: Optional["LoopResult"],
) -> Tuple[UnifiedExtractionOutput, int]:
    """Pulls the generator node's final output out of a completed
    LoopResult. `result` is None when run_parallel_loops() isolated a
    failure for this unit (see its own docstring) -- treated as an empty
    output rather than propagating the failure, the same way a shard that
    ran but produced nothing extractable is already handled."""
    if result is None:
        return UnifiedExtractionOutput(), 0
    output = result.node_outputs.get("generator")
    if not isinstance(output, UnifiedExtractionOutput):
        output = UnifiedExtractionOutput()
    return output, result.total_tokens


async def _run_generator_loop(
    context_str: str,
    model: Optional[str],
    max_retries: int,
) -> Tuple[UnifiedExtractionOutput, int]:
    """Runs one shard's generator loop directly (no cross-shard budget
    sharing) -- kept for any single-shard caller, e.g. isolated diagnostic
    scripts, and for shard reruns (_rerun_shard). Live Phase 1 extraction
    goes through run_parallel_loops() instead, with its own auto-scaled
    raw max_iter (see orchestrate()'s Phase 1 construction) rather than
    this function's "N full audited rounds" policy."""
    config = _build_generator_loop_config(rounds_to_max_iter(max_retries), model)
    result = await AgentLoop(config).run(context_str)
    return _extract_generator_output(result)


async def _rerun_shard(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    guidance: str,
    model: Optional[str],
    max_retries: int,
) -> Tuple[UnifiedExtractionOutput, int]:
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
    independent budget. This shape also keeps _rerun_shard directly
    monkeypatchable by name, which existing tests rely on.
    """
    context_str = _serialize_context(
        shard, fact_ids, facts_map, stub_tables, reconciliation_guidance=guidance
    )
    return await _run_generator_loop(context_str, model, max_retries)

"""Re-run Stage 2 with smaller chunks when the first attempt clearly saturated.

The per-call extraction capacity that drives chunking is a property of the MODEL,
so any hardcoded default is wrong for some model, and when it is too LARGE the
failure is silent: the extractor models a fraction of the domain and every stage
downstream succeeds on the fragment. Measured: one over-large chunk produced a
9-table schema from a 41-table domain with 86 of 121 facts unrepresented, and
nothing errored.

This closes that loop using a signal that needs no per-model calibration -- the
share of required facts that found no home in the final schema.

WHY THE END-TO-END SIGNAL AND NOT A PER-CHUNK ONE. The obvious cheaper design is
to detect a saturated chunk right after its extraction call, from the share of
its own facts the returned conceptual model cites, and re-split only that chunk.
That was measured on real artifacts and REJECTED: citation coverage does not
separate the cases. The catastrophic run cited 29% of its facts, but a run that
produced the best schema of the session (43 tables, 3 of 121 unrepresented) had
chunks citing 59%, 60%, 77% and 94% -- because how diligently the extractor
populates source_fact_ids varies on its own, independently of saturation. A
threshold that catches 29% cannot distinguish 59% healthy from 59% saturated, so
it would re-chunk good work and pay for it.

The end-to-end count is trustworthy where that one is not, because it reads the
provenance the MAPPER stamps rather than what the extractor volunteered -- the
same distinction that turned an earlier "37 of 52 facts uncovered" false alarm
into a real measurement. The price is that the signal only exists after a full
Stage 2 pass, so a retry costs a whole pass. That is why this fires only on the
unambiguous case and only once.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, List, Optional, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.util.schema_model.registry import TableFactRegistry

from src.orchestration.stage2.entry import _REQUIRED_FACT_TAGS
from src.orchestration.stage2.models import Output

logger = logging.getLogger(__name__)

# Same share the saturation warning uses, and grounded the same way: the
# collapsed run left 71% of required facts unrepresented, every adequately
# chunked run left 2-17%. A third separates those cleanly. Deliberately NOT
# tuned finer -- the gap it sits in is an order of magnitude wide, and a
# precise value would imply a precision the measurement does not have.
SATURATION_UNCOVERED_SHARE = 1 / 3

# One retry. The signal costs a full Stage 2 pass to obtain, so an unbounded
# loop could multiply the most expensive stage several times over on exactly the
# inputs that are already slowest. One halving is enough to escape the observed
# failure (121 facts in one chunk -> 4 chunks recovered the whole domain), and if
# it is not, the result is reported rather than retried again.
MAX_RECHUNK_ATTEMPTS = 1


def uncovered_share(output: Output, facts: List[AtomicFact]) -> float:
    """Share of REQUIRED facts with no home in the final schema, in [0, 1].

    Returns 0.0 when there is nothing to cover -- an empty fact set is not
    evidence of saturation, and treating it as 1.0 would make a degenerate spec
    trigger a pointless retry.
    """
    required = {
        f.id for f in facts if any(tag in f.tags for tag in _REQUIRED_FACT_TAGS)
    }
    if not required:
        return 0.0
    uncovered = set(output.uncovered_fact_ids or []) & required
    return len(uncovered) / len(required)


def is_saturated(output: Output, facts: List[AtomicFact]) -> bool:
    return uncovered_share(output, facts) >= SATURATION_UNCOVERED_SHARE


async def orchestrate_adaptive(
    plan: ChunkedPlan,
    facts: List[AtomicFact],
    run: Callable[[ChunkedPlan], Awaitable[Tuple[Output, int, TableFactRegistry]]],
    rechunk: Optional[Callable[[int], ChunkedPlan]] = None,
    max_attempts: int = MAX_RECHUNK_ATTEMPTS,
) -> Tuple[Output, int, TableFactRegistry]:
    """Run Stage 2, and retry with smaller chunks if the result clearly saturated.

    `run` performs one Stage 2 pass over a plan; `rechunk(n_chunks_wanted)`
    produces a finer plan. Both are injected rather than imported so this module
    depends on neither Stage 2's orchestrator nor Stage 1's chunker -- which also
    makes the control logic testable without any LLM.

    Tokens are SUMMED across attempts, not replaced: a retry genuinely costs what
    both passes cost, and reporting only the winner's tokens would understate the
    price of this mechanism in exactly the runs where it fired.

    Returns the better attempt by uncovered-fact count. "Better" is measured, not
    assumed to be the retry -- a smaller chunking is not guaranteed to help, and
    on a model that was not saturated it can only add fragmentation.
    """
    output, tokens, registry = await run(plan)
    total_tokens = tokens

    if rechunk is None or max_attempts <= 0:
        return output, total_tokens, registry

    share = uncovered_share(output, facts)
    if share < SATURATION_UNCOVERED_SHARE:
        return output, total_tokens, registry

    wanted = max(2, len(plan.chunks) * 2)
    finer = rechunk(wanted)
    if len(finer.chunks) <= len(plan.chunks):
        logger.warning(
            "  [Stage 2] %.0f%% of required facts are unrepresented, but the "
            "chunker cannot split further than %d chunk(s) -- the budget is "
            "already at the floor, or the facts are indivisible. Keeping this "
            "result; it is the best available, not a good one.",
            100 * share,
            len(plan.chunks),
        )
        return output, total_tokens, registry

    logger.warning(
        "  [Stage 2] %.0f%% of required facts are unrepresented after %d chunk(s), "
        "which is the signature of one extraction call being handed more domain "
        "than it can model. Retrying with %d chunks. This costs a second full "
        "Stage 2 pass and is done at most %d time(s).",
        100 * share,
        len(plan.chunks),
        len(finer.chunks),
        max_attempts,
    )

    retry_output, retry_tokens, retry_registry = await run(finer)
    total_tokens += retry_tokens
    retry_share = uncovered_share(retry_output, facts)

    if retry_share < share:
        logger.info(
            "  [Stage 2] re-chunking improved coverage: %.0f%% -> %.0f%% "
            "unrepresented across %d -> %d chunks.",
            100 * share,
            100 * retry_share,
            len(plan.chunks),
            len(finer.chunks),
        )
        return retry_output, total_tokens, retry_registry

    logger.warning(
        "  [Stage 2] re-chunking did NOT improve coverage (%.0f%% -> %.0f%%); "
        "keeping the original result. Saturation was not the cause here.",
        100 * share,
        100 * retry_share,
    )
    return output, total_tokens, registry

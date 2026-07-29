"""Shared, resilient parallel execution for this project's fan-out points
-- every stage that dispatches N independent async units concurrently
(Stage 2's per-fact-cluster ER extraction and per-conflict-component
adjudication; Stage 3's per-shard constraint extraction, per-schema-
locality-group reconciliation, and per-shard misextraction reruns).

Before this module, each of those 5 call sites hand-rolled its own bare
asyncio.gather(), with two real, repeated problems:

  1. No failure isolation -- a single unit's transient failure (e.g. one
     shard's structured-output parse error exhausting its retries) raised
     out of gather() and aborted every OTHER unit's in-progress work too,
     discarding all of it. Confirmed live: a Stage 3 run lost a shard that
     had just correctly extracted a state_sequences constraint because a
     SIBLING shard's unrelated parse failure crashed the whole gather().
  2. No retry-budget sharing -- every AgentLoop-based unit got a fixed,
     independent iteration budget with no way for a unit that converges
     early (using less than its share) to help a struggling sibling.

run_parallel() fixes (1) for any awaitable, loop-based or not.
run_parallel_loops() adds (2) on top, for AgentLoop-based units
specifically, via RetryBudget's shared mutable state (loop_types.py) --
safe under asyncio's cooperative single-threaded scheduling, no locks
needed: a donation from one unit's completion is visible to a sibling's
very next try_consume() check, since nothing else can run in between.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Generic, List, Optional, Sequence, TypeVar

from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    InputT,
    LoopConfig,
    LoopResult,
    RetryBudget,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_parallel(
    coros: Sequence[Awaitable[T]], *, labels: Optional[Sequence[str]] = None
) -> List[Optional[T]]:
    """Runs every coroutine concurrently, isolating failures: one unit's
    exception is logged and recorded as None, never propagated to abort
    the others. Returns one entry per input, in the same order."""
    if labels is not None and len(labels) != len(coros):
        raise ValueError("labels, if given, must be the same length as coros.")

    async def _run_one(i: int) -> Optional[T]:
        label = labels[i] if labels else str(i)
        try:
            return await coros[i]
        except Exception:
            logger.exception(
                f"[run_parallel] unit '{label}' failed -- isolated, "
                f"siblings unaffected."
            )
            return None

    return list(await asyncio.gather(*[_run_one(i) for i in range(len(coros))]))


@dataclass
class ParallelLoopSpec(Generic[InputT]):
    """One AgentLoop unit to run as part of a run_parallel_loops() batch."""

    config: LoopConfig
    initial_context: InputT
    label: str = ""


def _redistribute(
    amount: int,
    budgets: List[RetryBudget],
    done: List[bool],
    *,
    exclude: int,
) -> None:
    """Splits `amount` unused iterations evenly across every spec still
    IN FLIGHT (done[i] is False) other than `exclude`, the unit that just
    finished and donated it. `done` -- not the result value -- is the
    correct eligibility signal: a unit that already finished (success OR
    failure) must never receive a donation meant for one still running,
    and a finished-but-failed unit's result is indistinguishable from
    "still running" if judged by its LoopResult alone (both are falsy/
    absent), which is exactly why this needs its own explicit flag."""
    donor = budgets[exclude]
    recipients = [i for i in range(len(budgets)) if i != exclude and not done[i]]
    if not recipients:
        return
    share = amount // len(recipients)
    if share <= 0:
        return
    for i in recipients:
        transferred = donor.donate(share)
        budgets[i].receive(transferred)


async def run_parallel_loops(
    specs: List[ParallelLoopSpec],
) -> List[Optional[LoopResult]]:
    """Runs every spec's AgentLoop concurrently. Returns one LoopResult
    per spec (None on failure -- logged, never raised) in the same
    order. Retry budgets are live-shared: a loop that finishes under
    budget donates its remainder to whichever sibling loops are still
    running at that moment."""
    budgets = [RetryBudget(spec.config.max_iter) for spec in specs]
    results: List[Optional[LoopResult]] = [None] * len(specs)
    done = [False] * len(specs)

    async def _run_one(i: int) -> None:
        spec = specs[i]
        try:
            results[i] = await AgentLoop(spec.config).run(
                spec.initial_context, budgets[i]
            )
        except Exception:
            logger.exception(
                f"[run_parallel_loops] loop '{spec.label or i}' failed -- "
                f"isolated, siblings unaffected."
            )
        finally:
            done[i] = True
            leftover = budgets[i].remaining
            if leftover > 0:
                _redistribute(leftover, budgets, done, exclude=i)

    await asyncio.gather(*[_run_one(i) for i in range(len(specs))])
    return results

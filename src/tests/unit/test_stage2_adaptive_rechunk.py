"""Stage 2 retries with smaller chunks only when the result clearly saturated.

The control signal is the share of REQUIRED facts with no home in the final
schema -- chosen because it needs no per-model calibration, unlike the extraction
capacity constant it compensates for.

A cheaper per-chunk signal was measured and rejected first: citation coverage
does not separate the cases (the catastrophic run cited 29% of its facts, but the
BEST run of that session had chunks at 59%, 60%, 77% and 94%, because
source_fact_ids population varies on its own). These tests therefore pin the
end-to-end behaviour, including the cases where it must NOT fire.

No LLM anywhere: `run` and `rechunk` are injected, so this exercises the control
logic itself rather than a model's mood.
"""

from __future__ import annotations

import logging

import pytest

from src.orchestration.stage2.adaptive import (
    MAX_RECHUNK_ATTEMPTS,
    SATURATION_UNCOVERED_SHARE,
    is_saturated,
    orchestrate_adaptive,
    uncovered_share,
)
from src.orchestration.stage2.models import Output
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.util.schema_model.registry import TableFactRegistry


def _facts(n: int) -> list[AtomicFact]:
    return [
        AtomicFact(id=i, fact=f"fact {i}", tags=[FactTag.STRUCTURAL])
        for i in range(1, n + 1)
    ]


def _plan(facts: list[AtomicFact], n_chunks: int) -> ChunkedPlan:
    size = -(-len(facts) // n_chunks)
    chunks = [facts[i : i + size] for i in range(0, len(facts), size)]
    return ChunkedPlan(core_modeling_facts=list(facts), chunks=chunks)


def _output(plan: ChunkedPlan, uncovered: list[int]) -> Output:
    return Output(
        segments=[], plan=plan, final_global_schema=None, uncovered_fact_ids=uncovered
    )


def _runner(results: list[Output], tokens: int = 100):
    """Returns successive canned Outputs, recording the plans it was given."""
    seen: list[ChunkedPlan] = []

    async def run(plan: ChunkedPlan):
        seen.append(plan)
        return (
            results[min(len(seen) - 1, len(results) - 1)],
            tokens,
            TableFactRegistry(),
        )

    return run, seen


class TestUncoveredShare:
    def test_counts_only_required_facts(self):
        facts = _facts(10)
        out = _output(_plan(facts, 1), uncovered=[1, 2, 3])
        assert uncovered_share(out, facts) == pytest.approx(0.3)

    def test_no_facts_is_zero_not_one(self):
        """A degenerate spec must not look like total saturation and trigger a
        pointless retry."""
        out = _output(ChunkedPlan(core_modeling_facts=[], chunks=[]), uncovered=[])
        assert uncovered_share(out, []) == 0.0

    def test_ids_outside_the_fact_set_are_ignored(self):
        facts = _facts(4)
        out = _output(_plan(facts, 1), uncovered=[1, 99])
        assert uncovered_share(out, facts) == pytest.approx(0.25)

    def test_saturation_boundary(self):
        facts = _facts(3)
        assert is_saturated(_output(_plan(facts, 1), [1]), facts)  # 1/3
        assert not is_saturated(_output(_plan(facts, 1), []), facts)


class TestRetryFires:
    @pytest.mark.asyncio
    async def test_a_saturated_result_is_retried_with_more_chunks(self):
        facts = _facts(12)
        coarse, fine = _plan(facts, 1), _plan(facts, 4)
        bad = _output(coarse, uncovered=list(range(1, 10)))  # 75%
        good = _output(fine, uncovered=[1])  # 8%
        run, seen = _runner([bad, good])

        out, tokens, _ = await orchestrate_adaptive(
            coarse, facts, run=run, rechunk=lambda n: fine
        )
        assert len(seen) == 2
        assert len(seen[1].chunks) > len(seen[0].chunks)
        assert out is good
        assert tokens == 200, "both passes must be billed, not just the winner"

    @pytest.mark.asyncio
    async def test_the_worse_retry_is_discarded(self):
        """Finer chunks are not guaranteed to help; the better attempt wins on
        MEASUREMENT, not on being second."""
        facts = _facts(12)
        coarse, fine = _plan(facts, 1), _plan(facts, 4)
        first = _output(coarse, uncovered=list(range(1, 7)))  # 50%
        worse = _output(fine, uncovered=list(range(1, 11)))  # 83%
        run, _ = _runner([first, worse])

        out, tokens, _ = await orchestrate_adaptive(
            coarse, facts, run=run, rechunk=lambda n: fine
        )
        assert out is first
        assert tokens == 200

    @pytest.mark.asyncio
    async def test_it_retries_at_most_once(self):
        facts = _facts(12)
        coarse, fine = _plan(facts, 1), _plan(facts, 4)
        bad = _output(coarse, uncovered=list(range(1, 12)))
        run, seen = _runner([bad, bad, bad])
        await orchestrate_adaptive(coarse, facts, run=run, rechunk=lambda n: fine)
        assert len(seen) == 1 + MAX_RECHUNK_ATTEMPTS


class TestRetryDoesNotFire:
    @pytest.mark.asyncio
    async def test_a_healthy_result_is_not_retried(self):
        """The expensive case: a needless retry doubles the cost of the slowest
        stage."""
        facts = _facts(12)
        coarse = _plan(facts, 2)
        good = _output(coarse, uncovered=[1])
        run, seen = _runner([good])
        out, tokens, _ = await orchestrate_adaptive(
            coarse, facts, run=run, rechunk=lambda n: _plan(facts, 6)
        )
        assert len(seen) == 1
        assert out is good
        assert tokens == 100

    @pytest.mark.asyncio
    async def test_no_rechunker_means_no_retry(self):
        facts = _facts(12)
        coarse = _plan(facts, 1)
        bad = _output(coarse, uncovered=list(range(1, 12)))
        run, seen = _runner([bad])
        await orchestrate_adaptive(coarse, facts, run=run, rechunk=None)
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_it_gives_up_when_the_chunker_cannot_split_further(self, caplog):
        """Saturated but already at the floor -- retrying an identical plan would
        buy nothing and cost a full pass."""
        facts = _facts(12)
        coarse = _plan(facts, 3)
        bad = _output(coarse, uncovered=list(range(1, 12)))
        run, seen = _runner([bad])
        with caplog.at_level(logging.WARNING):
            out, tokens, _ = await orchestrate_adaptive(
                coarse, facts, run=run, rechunk=lambda n: coarse
            )
        assert len(seen) == 1
        assert tokens == 100
        assert "cannot split further" in caplog.text

    @pytest.mark.asyncio
    async def test_an_empty_fact_set_never_retries(self):
        empty = ChunkedPlan(core_modeling_facts=[], chunks=[])
        run, seen = _runner([_output(empty, [])])
        await orchestrate_adaptive(empty, [], run=run, rechunk=lambda n: empty)
        assert len(seen) == 1


class TestThresholdIsGrounded:
    def test_it_separates_the_measured_failure_from_the_measured_healthy_runs(self):
        """71% unrepresented was the collapse; 2-17% were the healthy runs. The
        threshold must sit strictly between, with room on both sides."""
        assert 0.17 < SATURATION_UNCOVERED_SHARE < 0.71

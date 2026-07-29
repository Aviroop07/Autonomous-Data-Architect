"""Stage 2's shard-and-merge is unreachable without a budget override.

The parallel extract-then-merge path is Stage 2's headline mechanism, but it
only runs when Stage 1 emits more than one chunk, and the per-chunk budget is
derived from the model's live-queried context window. Measured against real
artifacts, the most complex saved run produced 88 facts totalling 1,963
fact-tokens against a budget of roughly 623,000 -- a factor of ~317. A genuine
second chunk would require a specification yielding some 28,000 facts.

So every faithful run takes the single-chunk path, and the merge machinery is
never entered on a real run. `AblationConfig.forced_multi_chunk()` is the only
way to exercise it; these tests pin that the knob reaches the chunker and that
a run without it is unchanged.
"""

from __future__ import annotations

from typing import List

from src.pipeline.stage1.middleware.budget_chunker import (
    BudgetChunker,
    estimate_fact_tokens,
)
from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag
from src.util.config.ablation import AblationConfig


def _facts(n: int, *, segments: bool = True) -> List[AtomicFact]:
    """Facts long enough that the token estimate is not dominated by per-fact
    overhead, so a budget divides them predictably.

    `segment_text` matters and is easy to get wrong: the chunker packs SEGMENTS,
    not individual facts, so facts sharing a segment cannot be separated no
    matter how small the budget. Leaving it at its default empty string puts
    every fact in one segment and makes the input unshardable -- which is a
    property of the input, not a defect, and `segments=False` covers it.
    """
    return [
        AtomicFact(
            id=i,
            fact=(
                f"Entity number {i} records a measured quantity together with the "
                f"timestamp at which observation {i} was taken and the operator "
                f"who recorded it."
            ),
            tags=[FactTag.STRUCTURAL],
            segment_text=(f"Sentence {i} of the specification." if segments else ""),
            start_char=(i * 100 if segments else -1),
            end_char=(i * 100 + 40 if segments else -1),
        )
        for i in range(1, n + 1)
    ]


def test_the_default_config_sets_no_override() -> None:
    """A faithful run must be unaffected: None means "ask the model"."""
    assert AblationConfig().chunk_budget_tokens is None
    assert AblationConfig.full().chunk_budget_tokens is None
    assert AblationConfig.no_sharding().chunk_budget_tokens is None
    assert AblationConfig.bayesian_chunking().chunk_budget_tokens is None


def test_forced_multi_chunk_carries_the_budget() -> None:
    cfg = AblationConfig.forced_multi_chunk(490)
    assert cfg.chunk_budget_tokens == 490
    # And leaves the rest of the pipeline in its default shape, so the only
    # variable under test is the chunk count.
    assert cfg.enable_enrichment
    assert cfg.enable_sharding
    assert not cfg.use_bayesian_chunker


def test_a_budget_above_the_total_yields_exactly_one_chunk() -> None:
    """The single-chunk path every real run takes."""
    facts = _facts(40)
    total = sum(estimate_fact_tokens(f) for f in facts)
    plan = BudgetChunker(budget_tokens=total + 1).fit(facts)
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0]) == 40


def test_shrinking_the_budget_is_what_makes_extraction_shard() -> None:
    """Monotone in the direction that matters: a smaller budget never yields
    fewer chunks, and a small enough one yields more than one -- which is the
    entire point of the knob."""
    facts = _facts(40)
    total = sum(estimate_fact_tokens(f) for f in facts)

    counts = []
    for divisor in (1, 2, 4, 8):
        plan = BudgetChunker(budget_tokens=max(1, total // divisor)).fit(facts)
        counts.append(len(plan.chunks))

    assert counts[0] == 1, f"a full budget must not shard: {counts}"
    assert counts[-1] > 1, f"a small budget must shard: {counts}"
    assert counts == sorted(counts), f"chunk count must not decrease: {counts}"


def test_every_fact_survives_chunking_at_any_budget() -> None:
    """The property that makes the ablation trustworthy. If shrinking the budget
    silently dropped facts, a shard-and-merge measurement would be comparing
    against a smaller problem rather than the same one split up."""
    facts = _facts(40)
    total = sum(estimate_fact_tokens(f) for f in facts)

    for divisor in (1, 2, 3, 5, 8):
        plan = BudgetChunker(budget_tokens=max(1, total // divisor)).fit(facts)
        seen = [f.id for chunk in plan.chunks for f in chunk]
        assert sorted(seen) == [f.id for f in facts], (
            f"facts lost or duplicated at budget {total // divisor}: "
            f"{len(seen)} vs {len(facts)}"
        )


def test_no_chunk_is_empty() -> None:
    facts = _facts(40)
    total = sum(estimate_fact_tokens(f) for f in facts)
    plan = BudgetChunker(budget_tokens=max(1, total // 6)).fit(facts)
    assert plan.chunks
    assert all(chunk for chunk in plan.chunks), (
        "an empty chunk would spawn an extraction call with nothing to extract"
    )


def test_facts_sharing_one_segment_cannot_shard_at_any_budget() -> None:
    """The granularity of chunking is the SEGMENT, so this is a property of the
    input rather than a limit of the knob: a specification whose facts all trace
    to one span of text is unshardable, and the chunker says so in a warning
    rather than splitting mid-span. Worth pinning because it is the failure mode
    of a naive ablation -- shrink the budget, observe one chunk, and wrongly
    conclude the override does not work."""
    facts = _facts(40, segments=False)
    total = sum(estimate_fact_tokens(f) for f in facts)
    plan = BudgetChunker(budget_tokens=max(1, total // 8)).fit(facts)
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0]) == 40, "and no fact may be dropped in the process"


def test_a_single_fact_never_shards_however_small_the_budget() -> None:
    """A span is not split mid-fact -- an oversized single fact is emitted alone
    rather than truncated, which the chunker logs."""
    plan = BudgetChunker(budget_tokens=1).fit(_facts(1))
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0]) == 1

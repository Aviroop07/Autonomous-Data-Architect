"""A chunk must fit the budget. That is the whole job.

Found while chasing Stage 1's segmentation instability. Two live runs on the same
3,863-char specification produced 11 and 43 segments -- a ~4x swing, licensed by
the extractor prompt, which says outright "you choose its span (part of a
sentence, a sentence, or several related sentences)". Coarse segmentation is
therefore normal, and a single segment can carry more facts than a prompt holds.

The chunker handled that by emitting the segment whole, reasoning in a comment
that splitting "would cut a source span in half". That reasoning was wrong: a
segment is a GROUP of facts, and each fact keeps its own segment_text,
start_char and end_char. Splitting the group cuts no span -- it costs
co-location, nothing more -- while emitting it whole produces a chunk that
exceeds the model's context budget, which is the single failure chunking exists
to prevent.

Worse, the guard had a silent path. It read `if seg_tokens > budget and not
current`, so whenever anything was already pending, the warning never fired and
the oversized segment still went out whole. Measured before the fix: one small
segment followed by a 288-token segment against a 144-token budget yielded a
288-token chunk and no log line at all.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from src.pipeline.stage1.middleware.budget_chunker import (
    BudgetChunker,
    estimate_fact_tokens,
)
from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag


def _fact(idx: int, span: Tuple[int, int], words: int = 12) -> AtomicFact:
    return AtomicFact(
        id=idx,
        fact=" ".join(f"w{idx}x{j}" for j in range(words)),
        tags=[FactTag.STRUCTURAL],
        segment_text="segment text",
        start_char=span[0],
        end_char=span[1],
    )


def _tokens(facts: List[AtomicFact]) -> int:
    return sum(estimate_fact_tokens(f) for f in facts)


def _one_small_then_one_oversized() -> Tuple[List[AtomicFact], int]:
    """Facts sharing a (start_char, end_char) form ONE segment, so this is a
    small span followed by a single span carrying twelve facts."""
    facts = [_fact(1, (0, 10))] + [_fact(i, (100, 900)) for i in range(2, 14)]
    budget = _tokens(facts[1:]) // 2
    return facts, budget


def test_no_chunk_exceeds_the_budget() -> None:
    """The measured failure: this produced a chunk at 2x the budget."""
    facts, budget = _one_small_then_one_oversized()
    plan = BudgetChunker(budget_tokens=budget).fit(facts)

    oversized = [c for c in plan.chunks if _tokens(c) > budget]
    assert not oversized, (
        f"{len(oversized)} chunk(s) over the {budget}-token budget: "
        f"{[_tokens(c) for c in oversized]}"
    )


def test_splitting_loses_no_fact_and_keeps_document_order() -> None:
    """What makes the split safe. Provenance survives on each fact
    individually, so the only casualty is co-location."""
    facts, budget = _one_small_then_one_oversized()
    plan = BudgetChunker(budget_tokens=budget).fit(facts)

    out = [f for chunk in plan.chunks for f in chunk]
    assert [f.id for f in out] == [f.id for f in facts]
    assert all(f.start_char >= 0 and f.segment_text for f in out), (
        "every fact must still carry its own span provenance after the split"
    )


def test_the_split_is_reported(caplog) -> None:
    """The old guard went silent exactly when something was already pending,
    which is why this went unnoticed. It must warn regardless."""
    facts, budget = _one_small_then_one_oversized()
    with caplog.at_level(
        logging.WARNING, logger="src.pipeline.stage1.middleware.budget_chunker"
    ):
        BudgetChunker(budget_tokens=budget).fit(facts)

    assert any("over the" in r.getMessage() for r in caplog.records), (
        "an oversized segment must be reported even when a chunk is in progress"
    )


def test_it_warns_even_when_nothing_is_pending() -> None:
    """The path the old code did cover -- kept so the fix cannot regress it."""
    facts = [_fact(i, (100, 900)) for i in range(1, 13)]
    budget = _tokens(facts) // 2
    plan = BudgetChunker(budget_tokens=budget).fit(facts)
    assert not [c for c in plan.chunks if _tokens(c) > budget]
    assert len(plan.chunks) > 1


def test_a_single_oversized_fact_goes_out_alone_rather_than_truncated() -> None:
    """The genuine limit. Cutting inside one fact would truncate a statement,
    not merely separate two, so this one case still exceeds the budget -- by
    design, and loudly."""
    facts = [_fact(1, (0, 50), words=400)]
    budget = 10
    plan = BudgetChunker(budget_tokens=budget).fit(facts)
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0]) == 1, "the fact must survive intact"


def test_several_oversized_segments_each_get_split() -> None:
    """Two distinct oversized spans must not be merged into one chunk while
    being split -- each is packed independently."""
    facts = [_fact(i, (100, 200)) for i in range(1, 9)] + [
        _fact(i, (300, 400)) for i in range(9, 17)
    ]
    budget = _tokens(facts[:8]) // 2
    plan = BudgetChunker(budget_tokens=budget).fit(facts)

    assert not [c for c in plan.chunks if _tokens(c) > budget]
    assert [f.id for chunk in plan.chunks for f in chunk] == [f.id for f in facts]


def test_a_segment_that_fits_is_never_split() -> None:
    """The fix must not fire when it is not needed -- a segment inside budget
    keeps its facts together, which is the property the original guard was
    protecting and which still matters."""
    facts = [_fact(i, (100, 900)) for i in range(1, 5)]
    budget = _tokens(facts) * 4
    plan = BudgetChunker(budget_tokens=budget).fit(facts)
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0]) == 4

"""Chunking is bounded by EXTRACTION CAPACITY, not just by the context window.

The context-window criterion alone never split anything: it was measured at
623,145 tokens against a 2,992-token input, so `fit` always returned one chunk,
Stage 2's shard-and-merge path was structurally unreachable, and one
er_extractor call was asked to model a whole 41-entity domain -- producing 9
tables and leaving 86 of 121 facts unrepresented. Re-chunking the same facts to
a capacity ceiling took that to 43 tables and 3 unrepresented.

These tests pin the ceiling's PLUMBING -- that it binds, that it can be turned
off, that an explicit budget still wins, and that an unknown context window no
longer degrades to a single unbounded chunk. The VALUE of the constant is an
empirical question and deliberately not asserted here beyond its order of
magnitude; see the constant's own comment for the sweep.
"""

from __future__ import annotations

import logging

from src.pipeline.stage1.middleware.budget_chunker import (
    _CAPACITY_ENV_VAR,
    _DEFAULT_EXTRACTION_CAPACITY_TOKENS,
    _FALLBACK_EXTRACTION_CAPACITY_TOKENS,
    _MEASURED_ON,
    BudgetChunker,
    _resolve_default_capacity,
    estimate_fact_tokens,
)
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.pipeline.stage1.models.rephrased_nl import AtomicFact


def _facts(n: int, words: int = 40) -> list[AtomicFact]:
    """n facts, each long enough that a handful exceeds the capacity ceiling.

    Each gets its OWN source span, because _group_into_segments keys on
    (start_char, end_char) and facts sharing a span are deliberately kept
    together -- giving them all one span would test packing of a single
    indivisible segment instead of packing across segments.
    """
    body = " ".join(["alpha"] * words)
    return [
        AtomicFact(
            id=i,
            fact=f"Fact {i}: {body}.",
            tags=[FactTag.STRUCTURAL],
            start_char=i * 100,
            end_char=i * 100 + 90,
        )
        for i in range(1, n + 1)
    ]


class TestCapacityCeilingBinds:
    def test_a_large_fact_set_is_split_without_any_network_lookup(self):
        """No provider/model given, so the context window is unknowable here --
        the capacity ceiling alone must still chunk."""
        facts = _facts(60)
        plan = BudgetChunker(provider="nonexistent", model="nonexistent").fit(facts)
        assert len(plan.chunks) > 1

    def test_capacity_can_be_disabled_restoring_the_old_behaviour(self):
        facts = _facts(60)
        plan = BudgetChunker(
            provider="nonexistent",
            model="nonexistent",
            extraction_capacity_tokens=None,
        ).fit(facts)
        assert len(plan.chunks) == 1

    def test_an_explicit_budget_is_never_clamped_by_capacity(self):
        """Ablations and calibration sweeps address this module directly; quietly
        capping a caller's stated budget would make them measure something else."""
        facts = _facts(60)
        big = sum(estimate_fact_tokens(f) for f in facts) + 1_000
        plan = BudgetChunker(budget_tokens=big).fit(facts)
        assert len(plan.chunks) == 1

    def test_a_smaller_explicit_budget_still_splits(self):
        facts = _facts(60)
        plan = BudgetChunker(budget_tokens=200).fit(facts)
        assert len(plan.chunks) > 1

    def test_every_fact_survives_chunking(self):
        """A packing change must never lose or duplicate a fact."""
        facts = _facts(60)
        plan = BudgetChunker(provider="nonexistent", model="nonexistent").fit(facts)
        packed = [f.id for chunk in plan.chunks for f in chunk]
        assert sorted(packed) == [f.id for f in facts]
        assert len(packed) == len(set(packed))

    def test_no_chunk_exceeds_the_capacity_unless_one_fact_does(self):
        facts = _facts(60)
        plan = BudgetChunker(provider="nonexistent", model="nonexistent").fit(facts)
        for chunk in plan.chunks:
            total = sum(estimate_fact_tokens(f) for f in chunk)
            assert total <= _DEFAULT_EXTRACTION_CAPACITY_TOKENS or len(chunk) == 1

    def test_a_small_fact_set_still_fits_one_chunk(self):
        """The ceiling must not fragment work that genuinely fits."""
        plan = BudgetChunker(provider="nonexistent", model="nonexistent").fit(_facts(2))
        assert len(plan.chunks) == 1

    def test_single_fact_needs_no_packing(self):
        plan = BudgetChunker().fit(_facts(1))
        assert len(plan.chunks) == 1
        assert len(plan.chunks[0]) == 1


class TestCapacityConstant:
    def test_is_far_below_any_real_context_window(self):
        """The whole point: this must be the binding ceiling, not the window.
        A modern window is >=32k tokens; if this ever approached that, chunking
        would silently revert to never splitting."""
        assert 0 < _DEFAULT_EXTRACTION_CAPACITY_TOKENS < 32_000

    def test_is_large_enough_to_hold_a_useful_number_of_facts(self):
        """Guards the other direction -- a tiny value would fragment every spec
        into single-fact chunks and destroy the cross-fact context an ER
        extractor needs to see relationships at all."""
        one_fact = estimate_fact_tokens(_facts(1)[0])
        assert _DEFAULT_EXTRACTION_CAPACITY_TOKENS >= 5 * one_fact


class TestModelPortability:
    """The capacity ceiling is MODEL-specific, so it must be overridable and it
    must say where its value came from. A default calibrated on one model is
    wrong for another, and when it is too large the failure is silent
    under-modelling -- so the seam matters more than the number.
    """

    def test_env_var_overrides_the_measured_default(self, monkeypatch):
        monkeypatch.setenv(_CAPACITY_ENV_VAR, "1800")
        assert _resolve_default_capacity() == 1800

    def test_absent_env_var_uses_the_measured_default(self, monkeypatch):
        monkeypatch.delenv(_CAPACITY_ENV_VAR, raising=False)
        assert _resolve_default_capacity() == _FALLBACK_EXTRACTION_CAPACITY_TOKENS

    def test_non_numeric_override_falls_back_rather_than_crashing(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(_CAPACITY_ENV_VAR, "lots")
        with caplog.at_level(logging.WARNING):
            assert _resolve_default_capacity() == _FALLBACK_EXTRACTION_CAPACITY_TOKENS
        assert _CAPACITY_ENV_VAR in caplog.text

    def test_zero_disables_the_ceiling(self, monkeypatch):
        """An explicit opt-out has to remain possible -- it restores the
        context-window-only behaviour for anyone who wants it."""
        monkeypatch.setenv(_CAPACITY_ENV_VAR, "0")
        assert _resolve_default_capacity() <= 0

    def test_the_override_is_logged_with_the_model_it_was_measured_on(
        self, monkeypatch, caplog
    ):
        """Whoever reads the log must be able to tell that the built-in number
        came from a specific model, not from first principles."""
        monkeypatch.setenv(_CAPACITY_ENV_VAR, "1500")
        with caplog.at_level(logging.INFO):
            _resolve_default_capacity()
        assert _MEASURED_ON in caplog.text

    def test_the_measured_on_note_names_a_model_and_a_date(self):
        assert any(ch.isdigit() for ch in _MEASURED_ON)
        assert len(_MEASURED_ON) > 10


class TestEvenPacking:
    """Chunks are packed toward an EVEN share, not greedily filled.

    Greedy filling honours "no chunk exceeds the budget" while still leaving the
    first chunk pressed against it -- 1,128 tokens over a 900 budget gave
    [40 facts, 9 facts], so only the 9-fact remainder gained headroom. Now that
    the budget represents how much one call can MODEL, headroom is the point.
    """

    def test_chunks_are_of_comparable_size(self):
        facts = _facts(60)
        chunks = BudgetChunker(provider="nonexistent", model="nonexistent").fit(
            facts
        ).chunks
        sizes = [sum(estimate_fact_tokens(f) for f in c) for c in chunks]
        assert len(sizes) > 1
        # No chunk may be more than twice any other. Greedy filling produced a
        # 40-vs-9 fact split, which this rejects.
        assert max(sizes) <= 2 * min(sizes)

    def test_no_chunk_exceeds_the_real_budget(self):
        """The even target is tighter than the budget, so the invariant the
        budget exists to guarantee must still hold exactly."""
        facts = _facts(60)
        chunks = BudgetChunker(budget_tokens=900).fit(facts).chunks
        for chunk in chunks:
            total = sum(estimate_fact_tokens(f) for f in chunk)
            assert total <= 900 or len(chunk) == 1

    def test_a_tiny_remainder_is_folded_back(self):
        """Segment granularity leaves a tail; a chunk holding almost nothing is a
        whole extraction call wasted, so it merges when the result still fits."""
        facts = _facts(49)
        chunks = BudgetChunker(budget_tokens=900).fit(facts).chunks
        sizes = [sum(estimate_fact_tokens(f) for f in c) for c in chunks]
        if len(sizes) > 1:
            # the smallest chunk must not be a sliver
            assert min(sizes) > 0.25 * max(sizes)

    def test_coalescing_never_breaks_the_budget(self):
        for n in (20, 35, 49, 60, 90):
            facts = _facts(n)
            chunks = BudgetChunker(budget_tokens=900).fit(facts).chunks
            for chunk in chunks:
                total = sum(estimate_fact_tokens(f) for f in chunk)
                assert total <= 900 or len(chunk) == 1, f"n={n}"

    def test_facts_are_conserved_through_packing_and_coalescing(self):
        for n in (20, 49, 60):
            facts = _facts(n)
            chunks = BudgetChunker(budget_tokens=900).fit(facts).chunks
            packed = sorted(f.id for c in chunks for f in c)
            assert packed == [f.id for f in facts], f"n={n}"

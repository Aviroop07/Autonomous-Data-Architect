"""Tests for src/pipeline/stage1/middleware/budget_chunker.py.

The requirement this module exists to satisfy is scale invariance: 1 fact up to
hundreds of facts, with no tuning in between. Both ends are tested here, plus
the two things a packer must never do -- lose a fact, or split a source span.
"""

from __future__ import annotations

from unittest.mock import patch

from src.pipeline.stage1.middleware.budget_chunker import (
    BudgetChunker,
    estimate_fact_tokens,
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact


def _facts(n: int, *, per_segment: int = 3, text: str = "Entity has attribute") -> list:
    """n facts spread over ceil(n/per_segment) source spans."""
    return [
        AtomicFact(
            id=i + 1,
            fact=f"{text} number {i}.",
            tags=[],
            start_char=(i // per_segment) * 100,
            end_char=(i // per_segment) * 100 + 99,
        )
        for i in range(n)
    ]


def _all_ids(plan) -> list[int]:
    return sorted(f.id for chunk in plan.chunks for f in chunk)


class TestDegenerateEnd:
    """1 fact must work with no special handling by the caller."""

    def test_zero_facts(self):
        plan = BudgetChunker(budget_tokens=1000).fit([])
        assert plan.chunks == [[]]

    def test_one_fact(self):
        plan = BudgetChunker(budget_tokens=1000).fit(_facts(1))
        assert len(plan.chunks) == 1
        assert len(plan.chunks[0]) == 1

    def test_one_fact_with_an_absurdly_small_budget(self):
        """Even a budget smaller than the single fact yields one chunk -- there
        is nothing to pack, and emitting zero chunks would lose the fact."""
        plan = BudgetChunker(budget_tokens=1).fit(_facts(1))
        assert len(plan.chunks) == 1


class TestFitsInOnePrompt:
    def test_everything_under_budget_is_one_chunk(self):
        facts = _facts(60)
        plan = BudgetChunker(budget_tokens=1_000_000).fit(facts)
        assert len(plan.chunks) == 1
        assert _all_ids(plan) == list(range(1, 61))

    def test_unknown_budget_falls_back_to_one_chunk(self):
        """If the context window cannot be determined, behave the way the
        pipeline did before this module existed rather than guessing a split.

        get_context_window is patched to raise rather than left to fail on its
        own: a unit test must not depend on the network being absent, and
        letting the real lookup run would make this test do exactly what
        conftest's --live gate exists to prevent."""
        with patch(
            "src.util.core.context_window.get_context_window",
            side_effect=RuntimeError("no network"),
        ):
            plan = BudgetChunker(budget_tokens=None).fit(_facts(30))
        assert len(plan.chunks) == 1

    def test_non_positive_budget_falls_back_to_one_chunk(self):
        """Overhead can exceed a small model's whole window. That is a
        degenerate budget, not a reason to emit one chunk per fact."""
        with patch(
            "src.util.core.context_window.get_context_window", return_value=1_000
        ):
            plan = BudgetChunker(budget_tokens=None).fit(_facts(30))
        assert len(plan.chunks) == 1


class TestPackingWhenOverBudget:
    def test_splits_into_multiple_chunks(self):
        facts = _facts(60)
        total = sum(estimate_fact_tokens(f) for f in facts)
        plan = BudgetChunker(budget_tokens=total // 4).fit(facts)
        assert len(plan.chunks) > 1

    def test_no_fact_is_lost_or_duplicated(self):
        facts = _facts(60)
        total = sum(estimate_fact_tokens(f) for f in facts)
        plan = BudgetChunker(budget_tokens=total // 5).fit(facts)
        assert _all_ids(plan) == list(range(1, 61))

    def test_every_chunk_respects_the_budget(self):
        facts = _facts(60)
        total = sum(estimate_fact_tokens(f) for f in facts)
        budget = total // 4
        plan = BudgetChunker(budget_tokens=budget).fit(facts)
        for chunk in plan.chunks:
            assert sum(estimate_fact_tokens(f) for f in chunk) <= budget

    def test_a_source_span_is_never_split_across_chunks(self):
        """Facts sharing a span are one statement; handing two ER agents halves
        of it would be worse than an oversized chunk."""
        facts = _facts(60, per_segment=5)
        total = sum(estimate_fact_tokens(f) for f in facts)
        plan = BudgetChunker(budget_tokens=total // 6).fit(facts)
        seen: dict[tuple[int, int], int] = {}
        for idx, chunk in enumerate(plan.chunks):
            for f in chunk:
                key = (f.start_char, f.end_char)
                assert seen.setdefault(key, idx) == idx, (
                    f"span {key} appears in chunks {seen[key]} and {idx}"
                )

    def test_document_order_is_preserved(self):
        facts = _facts(60)
        total = sum(estimate_fact_tokens(f) for f in facts)
        plan = BudgetChunker(budget_tokens=total // 4).fit(facts)
        flat = [f.id for chunk in plan.chunks for f in chunk]
        assert flat == sorted(flat)

    def test_one_oversized_segment_goes_out_alone(self):
        """A single span bigger than the whole budget cannot be packed. It must
        still be emitted -- oversized and alone -- not dropped."""
        facts = _facts(10, per_segment=10, text="x" * 400)
        plan = BudgetChunker(budget_tokens=50).fit(facts)
        assert _all_ids(plan) == list(range(1, 11))


class TestAblationSwitchIsReal:
    """budget_chunker's docstring claims the Bayesian sampler stays selectable.
    If nothing routes to it, that claim is false and the 404-line module is
    simply dead code -- so assert the routing, not just the flag."""

    def test_default_config_selects_the_budget_chunker(self):
        from src.util.config.ablation import AblationConfig

        assert AblationConfig.full().use_bayesian_chunker is False

    def test_ablation_factory_selects_the_sampler(self):
        from src.util.config.ablation import AblationConfig

        assert AblationConfig.bayesian_chunking().use_bayesian_chunker is True

    def test_stage1_entry_branches_on_the_flag(self):
        """Guards against the flag existing but never being read."""
        import inspect

        from src.orchestration.stage1 import entry

        src = inspect.getsource(entry)
        assert "use_bayesian_chunker" in src
        assert "BudgetChunker" in src
        assert "BayesianChunker" in src


class TestScaleInvariance:
    def test_chunk_count_grows_with_input_not_with_tuning(self):
        """The same chunker instance handles 1 and 300 facts with no knobs."""
        budget = 400
        counts = []
        for n in (1, 10, 50, 150, 300):
            plan = BudgetChunker(budget_tokens=budget).fit(_facts(n))
            assert _all_ids(plan) == list(range(1, n + 1))
            counts.append(len(plan.chunks))
        assert counts == sorted(counts), f"chunk count not monotonic: {counts}"

    def test_large_input_is_fast(self):
        """The sampler this replaced was 12,000 sweeps of O(S^2). Packing is
        linear, so 600 facts must not take meaningful time."""
        import time

        facts = _facts(600)
        t0 = time.time()
        plan = BudgetChunker(budget_tokens=500).fit(facts)
        assert time.time() - t0 < 1.0
        assert _all_ids(plan) == list(range(1, 601))

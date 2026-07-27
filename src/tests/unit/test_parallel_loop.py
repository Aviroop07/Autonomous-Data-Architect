"""Tests for src/util/orchestration/parallel_loop.py.

Covers both the general resilient-gather primitive (run_parallel) and
the AgentLoop-specific retry-budget-reallocation layer on top
(run_parallel_loops) -- the shared infrastructure meant to replace every
hand-rolled asyncio.gather() fan-out across Stage 2/3 orchestration.
"""

from __future__ import annotations

import asyncio

import pytest

from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    GraphEdge,
    HistoryEntry,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
)
from src.util.orchestration.parallel_loop import (
    ParallelLoopSpec,
    run_parallel,
    run_parallel_loops,
)


# ---------------------------------------------------------------------------
# run_parallel -- general resilient gather
# ---------------------------------------------------------------------------


class TestRunParallel:
    def test_all_succeed_returns_in_order(self):
        async def make(v):
            return v

        results = asyncio.run(run_parallel([make(1), make(2), make(3)]))
        assert results == [1, 2, 3]

    def test_one_failure_isolated_others_still_return(self):
        async def ok(v):
            return v

        async def boom():
            raise ValueError("kaboom")

        results = asyncio.run(run_parallel([ok(1), boom(), ok(3)]))
        assert results == [1, None, 3]

    def test_label_length_mismatch_raises(self):
        async def make(v):
            return v

        async def _run():
            coros = [make(1), make(2)]
            try:
                await run_parallel(coros, labels=["only-one"])
            finally:
                # run_parallel raises before scheduling these -- close them
                # explicitly so pytest doesn't warn about unawaited coroutines.
                for c in coros:
                    c.close()

        with pytest.raises(ValueError):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# Stub agents for run_parallel_loops tests
# ---------------------------------------------------------------------------


class SimpleOutput(LoopOutputModel):
    value: str

    def get_errors(self) -> list[str]:
        return []


class ValidatedOutput(LoopOutputModel):
    value: str
    is_valid: bool = True

    def get_errors(self) -> list[str]:
        return [] if self.is_valid else ["bad"]


def _single_node_config(value: str, max_iter: int = 5) -> LoopConfig:
    class _Agent(LoopAgent):
        async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
            return SimpleOutput(value=value), 1

        def build_context(self, ctx: LoopContext) -> str:
            return ctx.initial_context

        def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
            return HistoryEntry(round=round_num, node=node, changes_summary="ok")

    agent = _Agent()
    return LoopConfig(
        agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
        graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
        start_node="generator",
        max_iter=max_iter,
    )


def _failing_config(max_iter: int = 5) -> LoopConfig:
    class _Agent(LoopAgent):
        async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
            raise RuntimeError("agent exploded")

        def build_context(self, ctx: LoopContext) -> str:
            return ctx.initial_context

        def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
            return HistoryEntry(round=round_num, node=node, changes_summary="")

    agent = _Agent()
    return LoopConfig(
        agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
        graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
        start_node="generator",
        max_iter=max_iter,
    )


class TestRunParallelLoopsBasic:
    def test_multiple_independent_specs_all_converge(self):
        specs = [
            ParallelLoopSpec(
                config=_single_node_config("a"), initial_context="", label="a"
            ),
            ParallelLoopSpec(
                config=_single_node_config("b"), initial_context="", label="b"
            ),
        ]
        results = asyncio.run(run_parallel_loops(specs))
        assert len(results) == 2
        assert results[0] is not None and results[0].final_output.value == "a"  # type: ignore[union-attr]
        assert results[1] is not None and results[1].final_output.value == "b"  # type: ignore[union-attr]

    def test_one_loop_failure_isolated_from_siblings(self):
        specs = [
            ParallelLoopSpec(
                config=_single_node_config("ok"), initial_context="", label="ok"
            ),
            ParallelLoopSpec(config=_failing_config(), initial_context="", label="bad"),
        ]
        results = asyncio.run(run_parallel_loops(specs))
        assert results[0] is not None
        assert results[0].final_output.value == "ok"  # type: ignore[union-attr]
        assert results[1] is None


class TestRunParallelLoopsBudgetReallocation:
    def test_early_finisher_donates_to_a_genuinely_concurrent_struggler(self):
        """Real, timing-dependent integration test (not the simulated
        version in test_util_loop.py): two ACTUAL AgentLoop instances run
        concurrently via run_parallel_loops(). Loop A converges in 1
        iteration (donating its remaining budget). Loop B needs more
        retries than its own max_iter=2 alone would allow, and only
        succeeds because A's donation arrives while B is still going.
        Both agents yield via asyncio.sleep(0) so the two tasks actually
        interleave instead of one running to completion before the other
        starts."""
        b_call_count = [0]

        class _FastAgent(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                await asyncio.sleep(0)
                return SimpleOutput(value="fast"), 1

            def build_context(self, ctx: LoopContext) -> str:
                return ""

            def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="")

        class _StrugglingAgent(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                await asyncio.sleep(0)
                b_call_count[0] += 1
                is_valid = b_call_count[0] >= 3
                return ValidatedOutput(value="v", is_valid=is_valid), 1

            def build_context(self, ctx: LoopContext) -> str:
                return ctx.initial_context

            def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
                assert isinstance(output, ValidatedOutput)
                return HistoryEntry(
                    round=round_num,
                    node=node,
                    changes_summary="valid" if output.is_valid else "invalid",
                )

        fast_agent = _FastAgent()
        fast_config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: fast_agent)},
            graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
            start_node="generator",
            # Converges in 1 iteration -- max_iter=5 leaves a genuine 4-unit
            # surplus to donate (max_iter=1 would leave nothing: remaining
            # would already be 0 the moment the single iteration is used).
            max_iter=5,
        )

        struggling_agent = _StrugglingAgent()
        struggling_config = LoopConfig(
            agents={
                "validator": AgentRoleConfig(agent_factory=lambda: struggling_agent)
            },
            graph={
                "edges": [
                    GraphEdge(
                        from_node="validator",
                        to_node="validator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="validator",
            # Needs 3 calls to succeed; max_iter=2 alone is NOT enough --
            # only A's donation makes the 3rd call possible.
            max_iter=2,
        )

        specs = [
            ParallelLoopSpec(config=fast_config, initial_context="", label="fast"),
            ParallelLoopSpec(
                config=struggling_config, initial_context="", label="struggling"
            ),
        ]
        results = asyncio.run(run_parallel_loops(specs))

        assert results[0] is not None and results[0].final_output.value == "fast"  # type: ignore[union-attr]
        assert results[1] is not None
        assert results[1].final_output.is_valid is True  # type: ignore[union-attr]
        assert b_call_count[0] == 3

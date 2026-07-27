"""AgentLoop treats a raising node as a failed iteration, not a fatal error.

Before this, `await agent.invoke(query)` in loop.py sat outside any try/except,
so one transient provider error discarded every token the loop had already
spent -- and on Stage 1's serial loops, killed the stage. An exception is just
a harsher validation failure, and the loop already exists to retry those.

The contract that must NOT change: a loop that never produces any output still
propagates, because run_parallel_loops maps an exception to None and None is
how every downstream consumer detects a dead shard.
"""

from __future__ import annotations

import asyncio

import pytest

from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    GraphEdge,
    HistoryEntry,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
)


class SimpleOutput(LoopOutputModel):
    value: str = ""

    def get_errors(self) -> list[str]:
        return []


class _ScriptedAgent(LoopAgent):
    """Raises for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times: int, value: str = "ok") -> None:
        self.fail_times = fail_times
        self.value = value
        self.calls = 0

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient failure {self.calls}")
        return SimpleOutput(value=self.value), 1

    def build_context(self, ctx: LoopContext) -> str:
        return ctx.initial_context

    def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
        return HistoryEntry(round=round_num, node=node, changes_summary="")


def _config(agent: LoopAgent, max_iter: int = 10) -> LoopConfig:
    return LoopConfig(
        agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
        graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
        start_node="generator",
        max_iter=max_iter,
    )


class TestTransientFailureIsRetried:
    def test_one_failure_then_success(self):
        agent = _ScriptedAgent(fail_times=1)
        result = asyncio.run(AgentLoop(_config(agent)).run("ctx"))
        assert result.final_output is not None
        assert result.final_output.value == "ok"  # type: ignore[union-attr]
        assert agent.calls == 2

    def test_two_failures_then_success_is_still_under_the_cap(self):
        agent = _ScriptedAgent(fail_times=2)
        result = asyncio.run(AgentLoop(_config(agent)).run("ctx"))
        assert result.final_output is not None
        assert agent.calls == 3

    def test_a_failure_does_not_consume_the_success_path(self):
        """The retry must re-run the SAME node, not advance the graph."""
        agent = _ScriptedAgent(fail_times=1)
        result = asyncio.run(AgentLoop(_config(agent)).run("ctx"))
        assert result.final_node == "generator"


class TestPermanentFailure:
    def test_never_produced_output_propagates(self):
        """Preserves the run_parallel_loops -> None contract for a dead loop."""
        agent = _ScriptedAgent(fail_times=99)
        with pytest.raises(RuntimeError):
            asyncio.run(AgentLoop(_config(agent)).run("ctx"))

    def test_gives_up_after_the_consecutive_cap_not_the_whole_budget(self):
        """A hard-down provider must not burn every remaining iteration."""
        agent = _ScriptedAgent(fail_times=99)
        with pytest.raises(RuntimeError):
            asyncio.run(AgentLoop(_config(agent, max_iter=100)).run("ctx"))
        assert agent.calls == AgentLoop._MAX_CONSECUTIVE_INVOKE_FAILURES

    def test_partial_output_is_kept_rather_than_raised(self):
        """Once a node has produced something, a later hard failure returns the
        last good output instead of discarding the run."""

        class _FailsAfterFirstSuccess(LoopAgent):
            def __init__(self) -> None:
                self.calls = 0

            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                self.calls += 1
                if self.calls == 1:
                    return SimpleOutput(value="first"), 1
                raise RuntimeError("provider died")

            def build_context(self, ctx: LoopContext) -> str:
                return ctx.initial_context

            def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="")

        agent = _FailsAfterFirstSuccess()
        # Loop back on itself so the node runs more than once.
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="generator"),
                ]
            },
            start_node="generator",
            max_iter=10,
        )
        result = asyncio.run(AgentLoop(config).run("ctx"))
        assert result.final_output is not None
        assert result.final_output.value == "first"  # type: ignore[union-attr]


class TestConsecutiveCounterResets:
    def test_a_success_clears_the_failure_streak(self):
        """Two failures, a success, then two more failures must not trip the
        cap -- otherwise a merely flaky provider is treated as a dead one."""

        class _Flaky(LoopAgent):
            def __init__(self) -> None:
                self.calls = 0

            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                self.calls += 1
                if self.calls in (1, 2, 4, 5):
                    raise RuntimeError("flaky")
                return SimpleOutput(value=str(self.calls)), 1

            def build_context(self, ctx: LoopContext) -> str:
                return ctx.initial_context

            def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="")

        agent = _Flaky()
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={"edges": [GraphEdge(from_node="generator", to_node="generator")]},
            start_node="generator",
            max_iter=8,
        )
        result = asyncio.run(AgentLoop(config).run("ctx"))
        # Reached call 6 without ever hitting 3 consecutive failures.
        assert agent.calls >= 6
        assert result.final_output is not None

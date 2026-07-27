"""
Offline unit tests for src/util/orchestration/loop.py and loop_types.py.

No LLM calls -- all agents are LoopAgent stubs.
Run with: pytest src/tests/unit/test_util_loop.py -v
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Optional

import pytest

from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    ErrorRefreshConfig,
    GraphEdge,
    HistoryEntry,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
    LoopResult,
    RetryBudget,
)


# ---------------------------------------------------------------------------
# Shared stub output models
# ---------------------------------------------------------------------------


class SimpleOutput(LoopOutputModel):
    value: str

    def get_errors(self) -> list[str]:
        return []


class ValidatedOutput(LoopOutputModel):
    value: str
    is_valid: bool = True
    error_messages: list[str] = []

    def get_errors(self) -> list[str]:
        return self.error_messages


class GradedOutput(LoopOutputModel):
    grade: str
    feedback: str = ""

    def get_errors(self) -> list[str]:
        return [self.feedback] if self.grade == "fail" and self.feedback else []


# ---------------------------------------------------------------------------
# Stub LoopAgent helpers
# ---------------------------------------------------------------------------


def make_stub_agent(
    responses: Sequence[LoopOutputModel],
    node_name: str = "stub",
    record_queries: Optional[list[str]] = None,
) -> LoopAgent:
    """Cycles through responses; records queries if record_queries list provided."""
    call_count = [0]

    class _StubAgent(LoopAgent):
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
            if record_queries is not None:
                record_queries.append(query)
            self.queries.append(query)
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx], 10

        def build_context(self, ctx: LoopContext) -> str:
            parts: list[str] = []
            if ctx.det_errors:
                parts.append("ERRORS:\n" + "\n".join(ctx.det_errors))
            if ctx.initial_context:
                parts.append(ctx.initial_context)
            return "\n\n".join(parts) if parts else "task"

        def emit_history(
            self,
            output: LoopOutputModel,
            prior: Optional[LoopOutputModel],
            round_num: int,
            node: str,
        ) -> HistoryEntry:
            return HistoryEntry(
                round=round_num,
                node=node,
                changes_summary=f"round {round_num}",
                was_improvement=None if prior is None else True,
            )

    return _StubAgent()


def make_validator_agent(errors_fn) -> LoopAgent:  # type: ignore[type-arg]
    """Validator that calls errors_fn(query) -> list[str] and returns ValidatedOutput."""

    class _ValidatorAgent(LoopAgent):
        async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
            errors = errors_fn(query)
            return ValidatedOutput(
                value=query, is_valid=len(errors) == 0, error_messages=errors
            ), 0

        def build_context(self, ctx: LoopContext) -> str:
            gen_out = ctx.node_outputs.get("generator") or ctx.node_outputs.get(
                "architect"
            )
            if isinstance(gen_out, (SimpleOutput, ValidatedOutput)):
                return gen_out.value
            return ""

        def emit_history(
            self,
            output: LoopOutputModel,
            prior: Optional[LoopOutputModel],
            round_num: int,
            node: str,
        ) -> HistoryEntry:
            assert isinstance(output, ValidatedOutput)
            return HistoryEntry(
                round=round_num,
                node=node,
                changes_summary="valid" if output.is_valid else "invalid",
            )

    return _ValidatorAgent()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_edge_from_node_raises(self):
        with pytest.raises(ValueError, match="from_node 'nonexistent'"):
            LoopConfig(
                agents={
                    "generator": AgentRoleConfig(
                        agent_factory=lambda: make_stub_agent([SimpleOutput(value="x")])
                    )
                },
                graph={"edges": [GraphEdge(from_node="nonexistent", to_node="end")]},
                start_node="generator",
            )

    def test_invalid_edge_to_node_raises(self):
        with pytest.raises(ValueError, match="to_node 'ghost'"):
            LoopConfig(
                agents={
                    "generator": AgentRoleConfig(
                        agent_factory=lambda: make_stub_agent([SimpleOutput(value="x")])
                    )
                },
                graph={"edges": [GraphEdge(from_node="generator", to_node="ghost")]},
                start_node="generator",
            )

    def test_invalid_start_node_raises(self):
        with pytest.raises(ValueError, match="start_node 'missing'"):
            LoopConfig(
                agents={
                    "generator": AgentRoleConfig(
                        agent_factory=lambda: make_stub_agent([SimpleOutput(value="x")])
                    )
                },
                graph={"edges": []},
                start_node="missing",
            )

    def test_valid_config_constructs_ok(self):
        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: make_stub_agent([SimpleOutput(value="x")])
                )
            },
            graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
            start_node="generator",
        )
        assert config.start_node == "generator"


# ---------------------------------------------------------------------------
# Single-node loop
# ---------------------------------------------------------------------------


class TestSingleNodeLoop:
    def test_runs_once_and_returns(self):
        agent = make_stub_agent([SimpleOutput(value="hello")])
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
            start_node="generator",
        )
        result = asyncio.run(AgentLoop(config).run("task"))
        assert isinstance(result, LoopResult)
        assert isinstance(result.final_output, SimpleOutput)
        assert result.final_output.value == "hello"
        assert result.iteration_count == 1
        assert result.total_tokens == 10

    def test_history_has_one_entry(self):
        agent = make_stub_agent([SimpleOutput(value="x")])
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
            start_node="generator",
        )
        result = asyncio.run(AgentLoop(config).run("task"))
        assert len(result.history) == 1
        assert result.history[0].node == "generator"
        assert result.history[0].round == 1


# ---------------------------------------------------------------------------
# agent_factory caching
# ---------------------------------------------------------------------------


class TestAgentFactoryCaching:
    def test_factory_called_once_across_iterations(self):
        factory_calls = [0]
        gen_count = [0]

        class _CountingAgent(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                gen_count[0] += 1
                return SimpleOutput(value=f"v{gen_count[0]}"), 10

            def build_context(self, ctx: LoopContext) -> str:
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        def my_factory() -> LoopAgent:
            factory_calls[0] += 1
            return _CountingAgent()

        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=my_factory)},
            graph={
                "edges": [
                    GraphEdge(
                        from_node="generator",
                        to_node="generator",
                        condition=EdgeCondition(field="value", op="eq", value="v1"),
                    ),
                    GraphEdge(from_node="generator", to_node="end"),
                ]
            },
            start_node="generator",
        )

        asyncio.run(AgentLoop(config).run("task"))
        assert factory_calls[0] == 1
        assert gen_count[0] == 2


# ---------------------------------------------------------------------------
# Generator-validator cycle (det_error_sources)
# ---------------------------------------------------------------------------


class TestGeneratorValidatorCycle:
    def _make_config(
        self, gen_responses: Sequence[LoopOutputModel], errors_fn
    ) -> tuple[LoopConfig, LoopAgent]:
        gen_agent = make_stub_agent(gen_responses, node_name="generator")
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(
                    agent_factory=lambda: val_agent,
                ),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=6,
        )
        return config, gen_agent

    def test_stops_when_validator_passes(self):
        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            return ["bad"] if call_count[0] < 2 else []

        config, _ = self._make_config(
            gen_responses=[SimpleOutput(value="v1"), SimpleOutput(value="v2")],
            errors_fn=errors_fn,
        )
        result = asyncio.run(AgentLoop(config).run(""))
        assert isinstance(result.final_output, ValidatedOutput)
        assert result.final_output.is_valid

    def test_det_errors_reach_generator_context(self):
        queries: list[str] = []
        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            return ["value must not be bad"] if call_count[0] < 2 else []

        gen_agent = make_stub_agent(
            [SimpleOutput(value="bad"), SimpleOutput(value="good")],
            record_queries=queries,
        )
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=5,
        )

        asyncio.run(AgentLoop(config).run(""))
        # Second generator invocation must see the validator's error in its context.
        assert len(queries) >= 2
        assert "value must not be bad" in queries[1]

    def test_respects_max_iter(self):
        def always_fail(query: str) -> list[str]:
            return ["always bad"]

        config, _ = self._make_config(
            gen_responses=[SimpleOutput(value="bad")],
            errors_fn=always_fail,
        )
        result = asyncio.run(AgentLoop(config).run(""))
        assert result.iteration_count == config.max_iter

    def test_get_errors_populates_unresolved_issues(self):
        """Infrastructure must overwrite unresolved_issues from output.get_errors()."""

        def errors_fn(query: str) -> list[str]:
            return ["structural error"]

        gen_agent = make_stub_agent([SimpleOutput(value="x")], node_name="generator")
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(agent_factory=lambda: gen_agent),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=2,
        )

        result = asyncio.run(AgentLoop(config).run(""))
        validator_entry = next(h for h in result.history if h.node == "validator")
        assert "structural error" in validator_entry.unresolved_issues

    def test_node_outputs_accessible_after_run(self):
        gen_agent = make_stub_agent(
            [SimpleOutput(value="final")], node_name="generator"
        )

        def no_errors(query: str) -> list[str]:
            return []

        val_agent = make_validator_agent(no_errors)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(agent_factory=lambda: gen_agent),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=2,
        )

        result = asyncio.run(AgentLoop(config).run(""))
        assert "generator" in result.node_outputs
        gen_out = result.node_outputs["generator"]
        assert isinstance(gen_out, SimpleOutput)
        assert gen_out.value == "final"


# ---------------------------------------------------------------------------
# EMA issue tracking
# ---------------------------------------------------------------------------


class TestEMATracking:
    def test_persistent_issues_surface_in_ema(self):
        class _PersistentBadAgent(LoopAgent):
            _count: int = 0

            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                self._count += 1
                return ValidatedOutput(
                    value=f"v{self._count}",
                    is_valid=False,
                    error_messages=["bad output"],
                ), 10

            def build_context(self, ctx: LoopContext) -> str:
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        agent = _PersistentBadAgent()
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={
                "edges": [
                    GraphEdge(
                        from_node="generator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="generator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=5,
            ema_alpha=0.8,
            ema_persistent_threshold=0.5,
        )

        result = asyncio.run(AgentLoop(config).run(""))
        assert len(result.history) == 5
        # Persistent errors must still be flagged after exhaustion
        assert all(
            len(h.unresolved_issues) > 0
            for h in result.history
            if h.node == "generator"
        )


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_converges_when_generator_unchanged(self):
        """Convergence fires when:
        - the generator produces the same output twice (was_improvement=False)
        - the critic has no structural errors (get_errors() == [])
        - the critic routes back to the generator
        """

        class _SameOutputGen(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return SimpleOutput(value="same"), 10

            def build_context(self, ctx: LoopContext) -> str:
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                was_improvement: Optional[bool] = None
                if isinstance(prior, SimpleOutput) and isinstance(output, SimpleOutput):
                    was_improvement = output.value != prior.value
                return HistoryEntry(
                    round=round_num,
                    node=node,
                    changes_summary="same",
                    was_improvement=was_improvement,
                )

        class _RoutingOnlyCritic(LoopAgent):
            """Routes back to generator every time but has no structural errors."""

            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return GradedOutput(grade="reject", feedback=""), 0

            def build_context(self, ctx: LoopContext) -> str:
                gen_out = ctx.node_outputs.get("generator")
                return str(gen_out) if gen_out else ""

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(
                    round=round_num, node=node, changes_summary="reject"
                )

        gen_agent = _SameOutputGen()
        crit_agent = _RoutingOnlyCritic()

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(agent_factory=lambda: gen_agent),
                "critic": AgentRoleConfig(agent_factory=lambda: crit_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="critic"),
                    GraphEdge(
                        from_node="critic",
                        to_node="generator",
                        condition=EdgeCondition(field="grade", op="eq", value="reject"),
                    ),
                    GraphEdge(from_node="critic", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=10,
        )

        result = asyncio.run(AgentLoop(config).run("task"))
        assert result.converged
        assert result.iteration_count < 10


# ---------------------------------------------------------------------------
# Stateful LoopAgent (fix_history pattern)
# ---------------------------------------------------------------------------


class TestStatefulLoopAgent:
    def test_agent_accumulates_state_across_rounds(self):
        """LoopAgent instance fields survive across loop iterations."""

        class _FixTrackingAgent(LoopAgent):
            def __init__(self) -> None:
                self.fix_history: list[str] = []

            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return SimpleOutput(value="output"), 5

            def build_context(self, ctx: LoopContext) -> str:
                if ctx.det_errors:
                    self.fix_history.extend(ctx.det_errors)
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        fix_count = [0]

        def errors_fn(query: str) -> list[str]:
            fix_count[0] += 1
            return [f"error_{fix_count[0]}"] if fix_count[0] <= 2 else []

        gen_agent = _FixTrackingAgent()
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "architect": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="architect", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="architect",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="architect",
            max_iter=8,
        )

        asyncio.run(AgentLoop(config).run(""))
        # gen_agent.fix_history accumulated errors from validator across retry rounds
        assert len(gen_agent.fix_history) == 2
        assert "error_1" in gen_agent.fix_history
        assert "error_2" in gen_agent.fix_history


# ---------------------------------------------------------------------------
# RetryBudget -- external, shareable iteration budget
# ---------------------------------------------------------------------------


class TestRetryBudgetUnit:
    def test_try_consume_decrements_until_exhausted(self):
        budget = RetryBudget(remaining=2)
        assert budget.try_consume() is True
        assert budget.remaining == 1
        assert budget.try_consume() is True
        assert budget.remaining == 0
        assert budget.try_consume() is False
        assert budget.remaining == 0

    def test_donate_transfers_at_most_available(self):
        budget = RetryBudget(remaining=3)
        transferred = budget.donate(10)
        assert transferred == 3
        assert budget.remaining == 0

    def test_donate_never_goes_negative(self):
        budget = RetryBudget(remaining=0)
        assert budget.donate(5) == 0
        assert budget.remaining == 0

    def test_receive_increases_remaining(self):
        budget = RetryBudget(remaining=1)
        budget.receive(4)
        assert budget.remaining == 5


class TestAgentLoopWithExternalBudget:
    def test_no_budget_arg_behaves_as_before(self):
        """Backward-compat guard: omitting `budget` must behave exactly
        like the pre-RetryBudget implementation."""
        agent = make_stub_agent([SimpleOutput(value="hello")])
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
            start_node="generator",
        )
        result = asyncio.run(AgentLoop(config).run("task"))
        assert result.iteration_count == 1
        assert result.final_output.value == "hello"  # type: ignore[union-attr]

    def test_external_budget_caps_iterations(self):
        def always_fail(query: str) -> list[str]:
            return ["always bad"]

        gen_agent = make_stub_agent([SimpleOutput(value="bad")], node_name="generator")
        validator = make_validator_agent(always_fail)
        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(agent_factory=lambda: gen_agent),
                "validator": AgentRoleConfig(agent_factory=lambda: validator),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=100,  # config's own cap is generous -- budget is what binds
        )
        budget = RetryBudget(remaining=3)
        result = asyncio.run(AgentLoop(config).run("", budget))
        assert result.iteration_count == 3
        assert budget.remaining == 0

    def test_pre_exhausted_budget_runs_zero_iterations(self):
        agent = make_stub_agent([SimpleOutput(value="hello")])
        config = LoopConfig(
            agents={"generator": AgentRoleConfig(agent_factory=lambda: agent)},
            graph={"edges": [GraphEdge(from_node="generator", to_node="end")]},
            start_node="generator",
        )
        budget = RetryBudget(remaining=0)
        result = asyncio.run(AgentLoop(config).run("task", budget))
        assert result.iteration_count == 0
        assert result.final_output is None

    def test_donated_budget_extends_a_struggling_loop(self):
        """Simulates the real cross-loop reallocation scenario: a loop
        that would otherwise exhaust its budget unconverged succeeds once
        a sibling donates extra iterations into the SAME shared budget
        object mid-run (via a side effect on the validator's 1st call).

        Budget=2 buys exactly one generator+validator pass (each node
        execution consumes 1 unit) before try_consume() would return
        False on the retry -- without a donation this loop would exit
        unconverged after that single round. The donation happens
        synchronously inside the validator's first call, so it is
        already visible to the very next try_consume() check (this
        loop's own retry), exactly like a sibling loop's completion
        callback would apply it in run_parallel_loops()."""
        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate a sibling loop finishing early and donating,
                # with a comfortable margin (only 2 more units strictly
                # needed for one more generator+validator round).
                shared_budget.receive(4)
            return [] if call_count[0] >= 2 else ["still bad"]

        gen_agent = make_stub_agent([SimpleOutput(value="v")], node_name="generator")
        validator = make_validator_agent(errors_fn)
        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(agent_factory=lambda: gen_agent),
                "validator": AgentRoleConfig(agent_factory=lambda: validator),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=100,
        )
        shared_budget = RetryBudget(remaining=2)  # would exhaust after round 1 alone
        result = asyncio.run(AgentLoop(config).run("", shared_budget))
        assert result.final_output is not None
        assert result.final_output.is_valid is True  # type: ignore[union-attr]
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Error accumulation + refresh (ErrorRefreshConfig)
# ---------------------------------------------------------------------------


class TestErrorAccumulation:
    def test_errors_accumulate_between_generator_runs(self):
        """Errors from the validator accumulate in the sink and are visible
        to the generator on the next round."""

        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                return ["error_A"]
            if call_count[0] == 2:
                return ["error_B"]
            return []

        gen_queries: list[str] = []

        gen_agent = make_stub_agent(
            [
                SimpleOutput(value="v1"),
                SimpleOutput(value="v2"),
                SimpleOutput(value="v3"),
            ],
            record_queries=gen_queries,
        )
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=6,
            error_refresh=ErrorRefreshConfig(trigger_node="generator"),
        )

        asyncio.run(AgentLoop(config).run(""))

        # Second generator call should see error_A (accumulated from round 1).
        assert len(gen_queries) >= 2
        assert "error_A" in gen_queries[1]

    def test_refresh_archives_and_resets_sink(self):
        """After the trigger node runs, the sink is archived to
        det_error_history and reset to empty."""

        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            if call_count[0] <= 2:
                return ["error_A"]
            return []

        captured_ctx: list[LoopContext] = []

        class _RecordingGen(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return SimpleOutput(value="v"), 10

            def build_context(self, ctx: LoopContext) -> str:
                captured_ctx.append(ctx)
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        gen_agent = _RecordingGen()
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=6,
            error_refresh=ErrorRefreshConfig(trigger_node="generator"),
        )

        asyncio.run(AgentLoop(config).run(""))

        # Lifecycle:
        # Gen1(iter1): sink=[], history=[], refresh archives nothing (empty)
        # Val1(iter2): sink=["error_A"] (call_count=1)
        # Gen2(iter3): sink=["error_A"], history=[], captured_ctx[1]
        #   -> refresh archives (3, ["error_A"]), sink=[]
        # Val2(iter4): sink=["error_A"] (call_count=2)
        # Gen3(iter5): sink=["error_A"], history=[(3,["error_A"])], captured_ctx[2]
        #   -> refresh archives (5, ["error_A"]), sink=[]
        # Val3(iter6): sink=[] (call_count=3, returns []), routes to end

        assert len(captured_ctx) >= 3

        # Gen1: empty sink, empty history (nothing to archive).
        assert captured_ctx[0].det_errors == []
        assert captured_ctx[0].det_error_history == []

        # Gen2: sink has error_A from Val1, no history yet.
        assert "error_A" in captured_ctx[1].det_errors
        assert captured_ctx[1].det_error_history == []

        # Gen3: sink has error_A from Val2, history has archived batch from Gen2.
        assert "error_A" in captured_ctx[2].det_errors
        assert len(captured_ctx[2].det_error_history) >= 1
        archived = [e for _, _, errs in captured_ctx[2].det_error_history for e in errs]
        assert "error_A" in archived

    def test_generator_sees_history_not_current_sink_after_refresh(self):
        """The generator at round N sees errors from rounds N-1 (in sink)
        and rounds 1..N-2 (in history)."""

        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                return ["error_A"]
            if call_count[0] == 2:
                return ["error_B"]
            return []

        captured_ctx: list[LoopContext] = []

        class _RecordingGen(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return SimpleOutput(value="v"), 10

            def build_context(self, ctx: LoopContext) -> str:
                captured_ctx.append(ctx)
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        gen_agent = _RecordingGen()
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=8,
            error_refresh=ErrorRefreshConfig(trigger_node="generator"),
        )

        asyncio.run(AgentLoop(config).run(""))

        # Find the context where error_B is in the sink (after round 2
        # validator found error_B but before next generator refresh).
        found_b_in_sink = False
        found_a_in_history = False
        for ctx in captured_ctx:
            if "error_B" in ctx.det_errors:
                found_b_in_sink = True
            for _, _, errs in ctx.det_error_history:
                if "error_A" in errs:
                    found_a_in_history = True

        assert found_b_in_sink, "error_B should appear in the sink at some point"
        assert found_a_in_history, "error_A should be archived in history"

    def test_empty_sink_not_archived(self):
        """If the sink is empty when refresh fires, nothing is archived."""

        captured_ctx: list[LoopContext] = []

        class _RecordingGen(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return SimpleOutput(value="v"), 10

            def build_context(self, ctx: LoopContext) -> str:
                captured_ctx.append(ctx)
                return "task"

            def emit_history(
                self,
                output: LoopOutputModel,
                prior: Optional[LoopOutputModel],
                round_num: int,
                node: str,
            ) -> HistoryEntry:
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        gen_agent = _RecordingGen()

        def no_errors(query: str) -> list[str]:
            return []

        val_agent = make_validator_agent(no_errors)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=4,
            error_refresh=ErrorRefreshConfig(trigger_node="generator"),
        )

        asyncio.run(AgentLoop(config).run(""))

        # No errors ever produced, so history should be empty.
        for ctx in captured_ctx:
            assert ctx.det_error_history == []

    def test_no_error_refresh_preserves_old_behavior(self):
        """Without error_refresh set, det_errors uses det_error_sources
        filtering exactly as before."""

        call_count = [0]

        def errors_fn(query: str) -> list[str]:
            call_count[0] += 1
            return ["error_A"] if call_count[0] < 2 else []

        queries: list[str] = []
        gen_agent = make_stub_agent(
            [SimpleOutput(value="bad"), SimpleOutput(value="good")],
            record_queries=queries,
        )
        val_agent = make_validator_agent(errors_fn)

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=5,
            # No error_refresh -- old behavior.
        )

        asyncio.run(AgentLoop(config).run(""))
        assert len(queries) >= 2
        assert "error_A" in queries[1]

    def test_multiple_nodes_all_contribute_to_sink(self):
        """Errors from both validator and auditor accumulate in the sink."""

        call_count = [0]

        class _AlternatingValidator(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                call_count[0] += 1
                if call_count[0] == 1:
                    return ValidatedOutput(
                        value="v", is_valid=False, error_messages=["val_error"]
                    ), 0
                return ValidatedOutput(value="v", is_valid=True), 0

            def build_context(self, ctx: LoopContext) -> str:
                gen_out = ctx.node_outputs.get("generator")
                return gen_out.value if isinstance(gen_out, SimpleOutput) else ""

            def emit_history(self, output, prior, round_num, node):
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        captured_ctx: list[LoopContext] = []

        class _RecordingGen(LoopAgent):
            async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
                return SimpleOutput(value="v"), 10

            def build_context(self, ctx: LoopContext) -> str:
                captured_ctx.append(ctx)
                return "task"

            def emit_history(self, output, prior, round_num, node):
                return HistoryEntry(round=round_num, node=node, changes_summary="x")

        gen_agent = _RecordingGen()
        val_agent = _AlternatingValidator()

        config = LoopConfig(
            agents={
                "generator": AgentRoleConfig(
                    agent_factory=lambda: gen_agent,
                    det_error_sources=["validator"],
                ),
                "validator": AgentRoleConfig(agent_factory=lambda: val_agent),
            },
            graph={
                "edges": [
                    GraphEdge(from_node="generator", to_node="validator"),
                    GraphEdge(
                        from_node="validator",
                        to_node="generator",
                        condition=EdgeCondition(field="is_valid", op="eq", value=False),
                    ),
                    GraphEdge(from_node="validator", to_node="end"),
                ]
            },
            start_node="generator",
            max_iter=6,
            error_refresh=ErrorRefreshConfig(trigger_node="generator"),
        )

        asyncio.run(AgentLoop(config).run(""))

        # Second generator call: sink should contain val_error.
        assert len(captured_ctx) >= 2
        assert "val_error" in captured_ctx[1].det_errors

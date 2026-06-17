"""
AgentLoop -- config-driven multi-agent iterative refinement executor.

See loop_types.py for all data models and configuration objects.

Every agent participating in a loop must:
  1. Inherit LoopAgent and implement invoke(), build_context(), emit_history().
  2. Use an output model that inherits LoopOutputModel and implements get_errors().

The infrastructure calls agent.build_context(ctx) to produce the query string,
agent.invoke(query) to get the model's output, then output.get_errors() to
populate entry.unresolved_issues and state.det_errors. The agent's emit_history()
provides the history summary; infrastructure overwrites unresolved_issues after.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
    LoopResult,
    _LoopState,
)


class AgentLoop:
    """Config-driven multi-agent iterative refinement loop.

    Fully offline-testable: inject stub LoopAgent instances via agent_factory.
    """

    def __init__(
        self,
        config: LoopConfig,
        agent_factory: Optional[Callable[[str, AgentRoleConfig], LoopAgent]] = None,
    ) -> None:
        """
        Args:
            config: Validated LoopConfig.
            agent_factory: Optional test-only override. Signature:
                (node_name, role_config) -> LoopAgent
                If None, agents are built from role.agent_factory().
        """
        self.config = config
        self._test_factory = agent_factory
        self._agents: dict[str, LoopAgent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, initial_context: str) -> LoopResult:
        """Run the loop from start_node until a stop condition is met."""
        state = _LoopState(initial_context=initial_context)
        current_node = self.config.start_node
        final_output: Optional[LoopOutputModel] = None
        final_node: str = current_node

        for iteration in range(1, self.config.max_iter + 1):
            state.iteration = iteration
            role = self.config.agents[current_node]
            agent = self._get_agent(current_node, role)

            # 1. Build context from prior outputs, history, routed det_errors.
            det_errors_for_ctx = self._select_det_errors(
                current_node, role, dict(state.det_errors)
            )
            ctx = LoopContext(
                initial_context=initial_context,
                current_node=current_node,
                iteration=iteration,
                node_outputs=dict(state.node_outputs),
                history=list(state.history),
                det_errors=det_errors_for_ctx,
                det_errors_by_node=dict(state.det_errors),
                ema_issues=self._get_ema_issues(state),
            )
            query = agent.build_context(ctx)

            # 2. Invoke.
            prior_output = state.node_outputs.get(current_node)
            output, tokens = await agent.invoke(query)
            state.total_tokens += tokens
            state.tokens_by_node[current_node] = (
                state.tokens_by_node.get(current_node, 0) + tokens
            )

            # 3. Infrastructure extracts errors from output and owns det_errors + history.
            errors = output.get_errors()
            state.det_errors[current_node] = errors

            entry = agent.emit_history(output, prior_output, iteration, current_node)
            entry = entry.model_copy(
                update={"unresolved_issues": errors, "tokens": tokens}
            )
            state.history.append(entry)

            # 4. Update EMA.
            self._update_ema(state, errors)

            # 5. Advance state.
            state.node_outputs[current_node] = output
            final_output = output
            final_node = current_node

            # 6. Evaluate outgoing edges.
            next_node = self._evaluate_edges(current_node, output, state)
            if next_node == "end" or next_node is None:
                break

            # 7. Convergence: generator cycled back with no improvement and no errors.
            if (
                next_node == self.config.start_node
                and current_node != self.config.start_node
            ):
                start_entry = next(
                    (
                        h
                        for h in reversed(state.history)
                        if h.node == self.config.start_node
                    ),
                    None,
                )
                if (
                    start_entry is not None
                    and start_entry.was_improvement is False
                    and not errors
                ):
                    state.converged = True
                    break

            current_node = next_node

        return LoopResult(
            final_output=final_output,
            final_node=final_node,
            total_tokens=state.total_tokens,
            tokens_by_node=dict(state.tokens_by_node),
            iteration_count=state.iteration,
            history=state.history,
            converged=state.converged,
            det_errors_exhausted=bool(state.det_errors.get(final_node)),
            node_outputs=dict(state.node_outputs),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_agent(self, node: str, role: AgentRoleConfig) -> LoopAgent:
        if node not in self._agents:
            if self._test_factory is not None:
                self._agents[node] = self._test_factory(node, role)
            else:
                self._agents[node] = role.agent_factory()
        return self._agents[node]

    def _select_det_errors(
        self,
        node: str,
        role: AgentRoleConfig,
        det_errors_by_node: dict[str, list[str]],
    ) -> list[str]:
        if role.det_error_sources is None:
            return det_errors_by_node.get(node, [])
        errors: list[str] = []
        for source in role.det_error_sources:
            errors.extend(det_errors_by_node.get(source, []))
        return errors

    def _evaluate_edges(
        self, current_node: str, output: LoopOutputModel, state: _LoopState
    ) -> Optional[str]:
        for edge in self.config.graph.get("edges", []):
            if edge.from_node == current_node and edge.matches(output):
                return edge.to_node
        return None

    def _update_ema(self, state: _LoopState, new_issues: list[str]) -> None:
        alpha = self.config.ema_alpha
        for key in state.ema_weights:
            state.ema_weights[key] *= 1 - alpha
        for issue in new_issues:
            state.ema_weights[issue] = (
                state.ema_weights.get(issue, 0.0) * (1 - alpha) + alpha
            )

    def _get_ema_issues(self, state: _LoopState) -> list[Tuple[str, float]]:
        threshold = self.config.ema_persistent_threshold
        return sorted(
            [(k, v) for k, v in state.ema_weights.items() if v >= threshold],
            key=lambda x: -x[1],
        )

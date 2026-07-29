"""Unified constraint-extraction agent for Stage 3.

Replaces the 3 separate statistical/structural/logic extractors with one
Generator node. Deterministic canonicalize() checking has been pulled out
into its own explicit node (middleware/deterministic_checker.py) -- this
agent's own invoke() only calls the LLM and applies the cheap, schema-free
structural checks (empty fact_references, etc.) via UnifiedOutput.get_errors().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage3.agents.extraction_outputs import AuditReport, UnifiedOutput
from src.pipeline.stage3.models.cross_shard import UnifiedExtractionOutput
from src.pipeline.stage3.models.shard_context import Stage3ShardContext
from src.util.schema_model.render import schema_to_prompt_text
from src.util.core.agent import AgentType
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def _facts_to_text(fact_ids: List[int], facts_map: Dict[int, AtomicFact]) -> str:
    lines: List[str] = ["## ALLOCATED FACTS"]
    for fid in sorted(fact_ids):
        fact = facts_map.get(fid)
        if fact is not None:
            lines.append(f"- [id={fid}] {fact.fact}")
    return "\n".join(lines)


class ConstraintGeneratorLoopAgent(LoopAgent):
    """LoopAgent for the unified constraint-generation node."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[AgentProvider] = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._agent: Optional[AgentType] = None
        self._errored_ids_history: set[int] = set()

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = resolve_agent_provider(self._provider).build(
                system_prompt=system_prompt,
                output_structure=UnifiedExtractionOutput,
                model=self._model,
                name="constraint_generator",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=UnifiedExtractionOutput,
            query=query,
        )
        assert isinstance(parsed, UnifiedExtractionOutput)
        wrapped = UnifiedOutput(
            distributions=parsed.distributions,
            moment_targets=parsed.moment_targets,
            correlations=parsed.correlations,
            structural_constraints=parsed.structural_constraints,
            logic_constraints=parsed.logic_constraints,
            derived_columns=parsed.derived_columns,
            state_sequences=parsed.state_sequences,
        )
        return wrapped, tokens

    def build_context(self, ctx: LoopContext[Stage3ShardContext]) -> str:
        ctx_data = ctx.initial_context

        parts: List[str] = []
        parts.append(schema_to_prompt_text(ctx_data.shard_schema, ctx_data.stub_tables))
        parts.append(_facts_to_text(ctx_data.fact_ids, ctx_data.facts_map))

        prior_output = ctx.node_outputs.get("generator")
        if isinstance(prior_output, UnifiedOutput) and self._errored_ids_history:

            def _accepted(items):
                return [
                    it
                    for it in items
                    if not any(
                        fid in self._errored_ids_history
                        for fid in getattr(it, "fact_references", ())
                    )
                ]

            # Derived from the model's own fields rather than enumerated by
            # hand. The hand-written version listed six of the seven constraint
            # lists -- state_sequences was missing -- so on every retry the
            # model was shown a snapshot with its state-machine constraints
            # absent and told to "keep these unchanged", which silently dropped
            # them. Enumerating fields at a call site is exactly the kind of
            # thing that goes stale when an eighth list is added.
            accepted = UnifiedExtractionOutput(
                **{
                    field: _accepted(value)
                    for field in UnifiedExtractionOutput.model_fields
                    if isinstance(value := getattr(prior_output, field, None), list)
                }
            )
            parts.append(
                "## ACCEPTED OUTPUT (keep these unchanged)\n"
                + accepted.model_dump_json(indent=2)
            )

        if ctx_data.reconciliation_guidance:
            parts.append(
                "## RECONCILIATION GUIDANCE (a conflict-reconciliation pass found a "
                "misextraction in a prior round -- fix this)\n"
                + ctx_data.reconciliation_guidance
            )

        if ctx.det_error_history:
            history_parts: List[str] = []
            for round_num, trigger_node, errs in ctx.det_error_history:
                batch_lines = "\n".join(f"  - {e}" for e in errs)
                history_parts.append(
                    f"--- Round {round_num} (triggered after {trigger_node}) ---\n"
                    + batch_lines
                )
            history_block = "\n".join(history_parts)
            parts.append(
                "## OLDER ERROR HISTORY "
                "(review before discarding -- some may already be fixed)\n"
                + history_block
            )

        if ctx.det_errors:
            feedback = "\n".join(f"- {err}" for err in ctx.det_errors)
            parts.append(
                "## VALIDATION FEEDBACK (correct these issues and re-propose)\n"
                + feedback
            )

        audit_output = ctx.node_outputs.get("auditor")
        if isinstance(audit_output, AuditReport) and not audit_output.is_valid:
            audit_feedback = "\n".join(f"- {issue}" for issue in audit_output.issues)
            parts.append(
                "## AUDIT FEEDBACK (a second reader found these problems -- fix them)\n"
                + audit_feedback
            )

        parts.append(
            "## TASK\nExtract every constraint from the facts above, across ALL 7 "
            "categories: distributions, moment_targets, correlations, "
            "structural_constraints, logic_constraints, derived_columns, and "
            "state_sequences. Do not default to only logic/structural facts -- "
            "explicitly check whether any fact describes a lifecycle/state-machine "
            "ordering (state_sequences) or a joint dependence between columns "
            "(correlations) before falling back to a plain logic_constraint."
        )
        return "\n\n".join(parts)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, UnifiedOutput)
        total = (
            len(output.distributions)
            + len(output.moment_targets)
            + len(output.correlations)
            + len(output.structural_constraints)
            + len(output.logic_constraints)
            + len(output.derived_columns)
            + len(output.state_sequences)
        )

        det_errors = getattr(output, "_det_errors", [])
        if det_errors:
            for items in (
                output.distributions,
                output.moment_targets,
                output.correlations,
                output.structural_constraints,
                output.logic_constraints,
                output.derived_columns,
            ):
                for it in items:
                    self._errored_ids_history.update(it.fact_references)

        if prior is None:
            return HistoryEntry(
                round=round_num,
                node=node,
                changes_summary=f"extracted {total} constraints",
                was_improvement=None,
            )

        assert isinstance(prior, UnifiedOutput)
        prior_total = (
            len(prior.distributions)
            + len(prior.moment_targets)
            + len(prior.correlations)
            + len(prior.structural_constraints)
            + len(prior.logic_constraints)
            + len(prior.derived_columns)
            + len(prior.state_sequences)
        )
        delta = total - prior_total
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=f"{total} constraints ({delta:+d} vs prior)",
            was_improvement=(delta != 0),
        )

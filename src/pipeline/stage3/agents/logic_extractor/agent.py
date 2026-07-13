"""Logic extraction agent for Stage 3.

Follows the Stage 1 fact_extractor LoopAgent convention: stateful subclass
combining LLM-call logic and loop participation in one class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.stage3.agents.extraction_outputs import AuditReport, LogicOutput
from src.pipeline.stage3.models.cross_shard import LogicExtractionOutput
from src.pipeline.stage3.models.grain import CanonicalizationFailure, canonicalize
from src.pipeline.stage3.models.on_nodes import ONBaseTable
from src.pipeline.stage2.models.schema import Schema
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def _build_agent(system_prompt: str, model: Optional[str] = None) -> AgentType:
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=LogicExtractionOutput,
        model=model,
        name="logic_extractor",
    )


def _schema_to_text(schema: Schema, stub_tables: Optional[List[str]] = None) -> str:
    """Render schema shard + stub tables into prompt-friendly text."""
    lines: List[str] = ["## SCHEMA SHARD"]
    for table in schema.tables:
        lines.append(f"### {table.name}")
        lines.append(f"  Primary key: {', '.join(table.primary_key)}")
        for col in table.columns:
            nullable = "NULL" if col.is_nullable else "NOT NULL"
            lines.append(f"  {col.name}: {col.data_type} {nullable}")
        for fk in schema.relationships or []:
            if fk.referencing_table == table.name:
                lines.append(
                    f"  FK: {fk.referencing_column} -> "
                    f"{fk.referred_table} (its primary key)"
                )

    if stub_tables:
        lines.append("\n## STUB TABLES (cross-shard, schema-only)")
        for stub in stub_tables:
            lines.append(f"### {stub} (stub)")
            lines.append("  (columns not available -- use for ON-tree references only)")

    return "\n".join(lines)


def _facts_to_text(fact_ids: List[int], facts_map: dict) -> str:
    """Render allocated facts into prompt-friendly text."""
    lines: List[str] = ["## ALLOCATED FACTS"]
    for fid in sorted(fact_ids):
        fact = facts_map.get(fid)
        if fact is not None:
            lines.append(f"- [id={fid}] {fact.fact}")
    return "\n".join(lines)


class LogicExtractorLoopAgent(LoopAgent):
    """LoopAgent for the logic extraction node.

    Stateful across retries: tracks which fact IDs have errored and
    accumulates accepted outputs.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model
        self._agent: Optional[AgentType] = None
        self._errored_ids_history: set[int] = set()
        self._last_schema: Optional[Schema] = None

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = _build_agent(system_prompt=system_prompt, model=self._model)
        return self._agent

    def _validate_output(
        self, output: LogicExtractionOutput, schema: Schema
    ) -> List[str]:
        """Deterministic validation: canonicalize every ON tree, plus
        derived columns' synthetic base-table ON (mirrors how
        constraint_graph.py's _derived_column_to_rich canonicalizes them)."""
        errors: List[str] = []
        for i, c in enumerate(output.constraints):
            result = canonicalize(c.on, schema)
            if isinstance(result, CanonicalizationFailure):
                errors.append(f"Logic[{i}] ON canonicalization failed: {result.reason}")
        for i, dc in enumerate(output.derived):
            result = canonicalize(ONBaseTable(name=dc.target_table), schema)
            if isinstance(result, CanonicalizationFailure):
                errors.append(
                    f"Derived[{i}] target_table canonicalization failed: {result.reason}"
                )
        return errors

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=LogicExtractionOutput,
            query=query,
        )
        assert isinstance(parsed, LogicExtractionOutput)

        # Deterministic validation
        det_errors: List[str] = []
        if self._last_schema is not None:
            det_errors = self._validate_output(parsed, self._last_schema)

        wrapped = LogicOutput(constraints=parsed.constraints, derived=parsed.derived)
        wrapped._det_errors = det_errors  # type: ignore[attr-defined]
        return wrapped, tokens

    def build_context(self, ctx: LoopContext) -> str:
        context_data: Dict[str, Any] = {}
        try:
            context_data = json.loads(ctx.initial_context)
        except json.JSONDecodeError, TypeError, AttributeError:
            logger.warning("[LogicExtractor] Failed to parse initial_context as JSON.")

        schema_raw = context_data.get("schema")
        stub_tables = context_data.get("stub_tables")
        fact_ids = context_data.get("fact_ids", [])
        facts_map = context_data.get("facts_map", {})

        schema: Optional[Schema] = None
        if schema_raw is not None:
            if isinstance(schema_raw, Schema):
                schema = schema_raw
            elif isinstance(schema_raw, dict):
                try:
                    schema = Schema(**schema_raw)
                except Exception:
                    pass

        self._last_schema = schema

        parts: List[str] = []

        if schema is not None:
            parts.append(_schema_to_text(schema, stub_tables))

        if fact_ids and facts_map:
            parts.append(_facts_to_text(fact_ids, facts_map))

        # Prior output
        prior_output: Optional[LogicOutput] = None
        raw_prior = ctx.node_outputs.get("logic_extractor")
        if isinstance(raw_prior, LogicOutput):
            prior_output = raw_prior

        if prior_output is not None and self._errored_ids_history:
            accepted = [
                c
                for c in prior_output.constraints
                if not any(
                    fid in self._errored_ids_history for fid in c.fact_references
                )
            ]
            accepted_derived = [
                dc
                for dc in prior_output.derived
                if not any(
                    fid in self._errored_ids_history for fid in dc.fact_references
                )
            ]
            if accepted or accepted_derived:
                accepted_output = LogicExtractionOutput(
                    constraints=accepted, derived=accepted_derived
                )
                parts.append(
                    "## ACCEPTED OUTPUT (keep these unchanged)\n"
                    + accepted_output.model_dump_json(indent=2)
                )

        # Reconciliation guidance (a conflict-reconciliation pass judged a
        # prior conflict a MISEXTRACTION and pinpointed what this family got
        # wrong -- only present on a targeted re-extraction run, read from
        # initial_context since there is no prior node_outputs entry yet).
        reconciliation_guidance = context_data.get("reconciliation_guidance")
        if reconciliation_guidance:
            parts.append(
                "## RECONCILIATION GUIDANCE (a conflict-reconciliation pass "
                "found a misextraction in a prior round -- fix this)\n"
                + reconciliation_guidance
            )

        if ctx.det_errors:
            feedback = "\n".join(f"- {err}" for err in ctx.det_errors)
            parts.append(
                "## VALIDATION FEEDBACK (correct these issues and re-propose)\n"
                + feedback
            )

        # Semantic audit feedback (logic_auditor re-read the facts against
        # the extraction and found real problems -- distinct from the
        # structural VALIDATION FEEDBACK above, which canonicalize()
        # produces and can't catch things like a wrong derivation formula).
        audit_output = ctx.node_outputs.get("logic_auditor")
        if isinstance(audit_output, AuditReport) and not audit_output.is_valid:
            audit_feedback = "\n".join(f"- {issue}" for issue in audit_output.issues)
            parts.append(
                "## AUDIT FEEDBACK (a second reader found these problems -- fix them)\n"
                + audit_feedback
            )

        parts.append("## TASK\nExtract logic constraints from the facts above.")
        return "\n\n".join(parts)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, LogicOutput)
        n = len(output.constraints) + len(output.derived)

        # Track errored fact IDs
        det_errors = getattr(output, "_det_errors", [])
        if det_errors:
            for c in output.constraints:
                for e in det_errors:
                    if "Logic" in e:
                        self._errored_ids_history.update(c.fact_references)
                        break
            for dc in output.derived:
                for e in det_errors:
                    if "Derived" in e:
                        self._errored_ids_history.update(dc.fact_references)
                        break

        if prior is None:
            return HistoryEntry(
                round=round_num,
                node=node,
                changes_summary=(
                    f"extracted {n} constraints "
                    f"({len(output.constraints)} logic, {len(output.derived)} derived)"
                ),
                was_improvement=None,
            )

        assert isinstance(prior, LogicOutput)
        prior_n = len(prior.constraints) + len(prior.derived)
        delta = n - prior_n

        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=(
                f"{n} constraints ({len(output.constraints)} logic, "
                f"{len(output.derived)} derived) ({delta:+d} vs prior)"
            ),
            was_improvement=(delta != 0),
        )

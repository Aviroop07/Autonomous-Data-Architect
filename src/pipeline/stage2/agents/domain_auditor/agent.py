import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)
from src.util.schema_ops.schema_patch import CritiqueReport

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class DomainAuditorLoopAgent(LoopAgent):
    """LoopAgent for the domain auditor node.

    Absorbs the old _AuditorContextBuilder stateful logic. Error feedback
    arrives via ctx.det_errors (routed from patch_validator via det_error_sources).
    Schema state is updated from the patch_validator output using duck-typing
    to avoid a circular import with utils.py.
    """

    def __init__(
        self,
        schema: Schema,
        intelligence: BaseModel,
        fact_clusters: Dict[str, List[AtomicFact]],
        initial_errors: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> None:
        self.current_schema = schema
        self._intelligence = intelligence
        self._initial_errors: List[str] = initial_errors or []
        self._first_call = True
        self._model = model
        self._agent: Optional[AgentType] = None
        self._fact_pool: Dict[int, str] = {}
        self._entity_fact_ids: Dict[str, List[int]] = {}
        for entity, facts in fact_clusters.items():
            ids: List[int] = []
            for fact in facts:
                if fact.id not in self._fact_pool:
                    self._fact_pool[fact.id] = fact.fact
                ids.append(fact.id)
            if ids:
                self._entity_fact_ids[entity] = sorted(set(ids))

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = get_agent_(
                system_prompt=system_prompt,
                output_structure=CritiqueReport,
                model=self._model,
                name="domain_auditor_stage2",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=CritiqueReport,
            query=query,
        )
        assert isinstance(parsed, CritiqueReport)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        structural_errors: List[str] = list(ctx.det_errors)

        validator_output = ctx.node_outputs.get("patch_validator")
        if validator_output is not None and hasattr(validator_output, "schema_state"):
            self.current_schema = validator_output.schema_state  # type: ignore[union-attr]
            self._first_call = False
        elif self._first_call and self._initial_errors:
            structural_errors = self._initial_errors
            self._first_call = False
        else:
            self._first_call = False

        error_feedback = (
            "\n".join(f"- {e}" for e in structural_errors)
            if structural_errors
            else "None. The previous state was structurally valid."
        )

        fact_lines = []
        for fact_id in sorted(self._fact_pool.keys()):
            fact_lines.append(f"- {fact_id}: {self._fact_pool[fact_id]}")
        fact_pool_block = "\n".join(fact_lines) if fact_lines else "(none)"

        entity_lines = []
        for entity in sorted(self._entity_fact_ids.keys()):
            ids_str = ", ".join(str(fid) for fid in self._entity_fact_ids[entity])
            entity_lines.append(f"- {entity}: [{ids_str}]")
        entity_block = "\n".join(entity_lines) if entity_lines else "(none)"

        intelligence_payload = self._intelligence.model_dump()
        compact_intelligence = {}
        for key in ("domain", "research_summary"):
            if key in intelligence_payload:
                compact_intelligence[key] = intelligence_payload[key]
        if compact_intelligence:
            intelligence_block = json.dumps(compact_intelligence, indent=2)
        else:
            intelligence_block = self._intelligence.model_dump_json(indent=2)

        return (
            f"### GENERATED SCHEMA:\n{self.current_schema}\n\n"
            f"### DOMAIN INTELLIGENCE REPORT:\n{intelligence_block}\n\n"
            f"### FACT POOL (UNIQUE FACTS):\n{fact_pool_block}\n\n"
            f"### ENTITY FACT IDS (HINTS):\n{entity_block}\n\n"
            f"### STRUCTURAL ERRORS (FEEDBACK):\n{error_feedback}"
        )

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, CritiqueReport)
        patch_count = len(output.patches)
        was_improvement = None
        if isinstance(prior, CritiqueReport):
            was_improvement = patch_count <= len(prior.patches)
        changes_summary = f"{patch_count} patches" if patch_count else "no patches"
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=was_improvement,
        )

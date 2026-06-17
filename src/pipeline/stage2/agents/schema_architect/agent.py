from pathlib import Path
from typing import List, Optional

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.models.schema import Schema
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class SchemaArchitectLoopAgent(LoopAgent):
    """LoopAgent for the schema architect node.

    Absorbs the old _ArchitectContextBuilder stateful logic. Error feedback
    arrives via ctx.det_errors (routed from the schema_validator node via
    det_error_sources). Maintains fix_history for caller inspection.
    """

    def __init__(
        self,
        facts: List[AtomicFact],
        model: Optional[str] = None,
    ) -> None:
        self._model = model
        self._agent: Optional[AgentType] = None
        formatted = []
        for fact in facts:
            tag_values = [t.value if hasattr(t, "value") else str(t) for t in fact.tags]
            non_structural = [t for t in tag_values if t != "STRUCTURAL"]
            if non_structural:
                tag_str = ", ".join(non_structural)
                formatted.append(f"- [{tag_str}] {fact.fact}")
            else:
                formatted.append(f"- {fact.fact}")
        self._formatted_facts = "\n".join(formatted)
        self.fix_history: List[FixHistoryStep] = []
        self._attempt: int = 0

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = get_agent_(
                system_prompt=system_prompt,
                output_structure=Schema,
                model=self._model,
                name="schema_architect",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        schema, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=Schema,
            query=query,
        )
        assert isinstance(schema, Schema)
        return schema, tokens

    def build_context(self, ctx: LoopContext) -> str:
        prior_schema = ctx.node_outputs.get("architect")
        errors: List[str] = list(ctx.det_errors)

        if isinstance(prior_schema, Schema) and errors:
            self._attempt += 1
            self.fix_history.append(
                FixHistoryStep(
                    attempt=self._attempt,
                    errors=errors,
                    corrections=[],
                    fixed_schema=str(prior_schema),
                    schema_state=prior_schema.model_copy(deep=True),
                )
            )

        query = (
            "TARGET CHUNK FACTS (default tag: STRUCTURAL; non-structural tags in brackets):\n"
            f"{self._formatted_facts}"
        )
        if isinstance(prior_schema, Schema) and errors:
            query += (
                f"\n\nCURRENT SCHEMA STATE (JSON):\n{prior_schema.model_dump_json(indent=2)}"
                "\n\nCRITICAL STRUCTURAL ERRORS TO FIX:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nMISSION: Repair the schema to resolve all errors while "
                "preserving the intent of the original facts."
            )
        return query

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, Schema)
        changes_summary = (
            f"{len(output.tables)} tables, "
            f"{len(output.relationships or [])} relationships"
        )
        was_improvement = None
        if isinstance(prior, Schema):
            was_improvement = output.model_dump() != prior.model_dump()
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=was_improvement,
        )

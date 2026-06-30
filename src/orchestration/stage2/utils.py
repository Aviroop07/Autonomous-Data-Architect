import json
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.atomic_fact import FactTag
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Schema
from src.util.orchestration.loop import AgentLoop
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
from src.util.orchestration.retry_loop import ErrorRecord, ErrorType, Severity
from src.util.schema_ops.patching_engine import apply_patches

from src.pipeline.stage2.agents.conceptual_extractor.agent import get_agent as get_conceptual_extractor_agent
from src.pipeline.stage2.agents.conceptual_verifier.agent import get_agent as get_conceptual_verifier_agent
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.pipeline.stage2.models.conceptual_critique import ConceptualCritiqueReport


# ---------------------------------------------------------------------------
# Private types
# ---------------------------------------------------------------------------





class ConceptualExtractorLoopAgent(LoopAgent):
    def __init__(self, facts: List[AtomicFact], nl_query: str, model: Optional[str] = None):
        self._facts = facts
        self._nl_query = nl_query
        self._model = model
        self.agent = get_conceptual_extractor_agent(model)

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        from src.util.core.invoke import get_response
        facts_str = "\n".join([f"- [{f.id}] {f.fact}" for f in self._facts])
        full_query = f"## INPUT\nOriginal Description:\n{self._nl_query}\n\nFacts:\n{facts_str}"
        if query:
            full_query += f"\n\n## FEEDBACK\n{query}"
            
        parsed, tokens = await get_response(self.agent, ConceptualModel, full_query)
        return parsed, tokens
        
    def build_context(self, ctx: LoopContext) -> str:
        report = ctx.node_outputs.get("verifier")
        if isinstance(report, ConceptualCritiqueReport):
            return "Please fix the following issues:\n" + "\n".join([f"- {f.description}: {f.rationale}" for f in report.fixes])
        return ""

    def emit_history(self, output: LoopOutputModel, prior: Optional[LoopOutputModel], round_num: int, node: str) -> HistoryEntry:
        return HistoryEntry(round=round_num, node=node, changes_summary="Extracted conceptual model")

class ConceptualVerifierLoopAgent(LoopAgent):
    def __init__(self, facts: List[AtomicFact], model: Optional[str] = None):
        self._facts = facts
        self._model = model
        self.agent = get_conceptual_verifier_agent(model)
        
    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        from src.util.core.invoke import get_response
        parsed, tokens = await get_response(self.agent, ConceptualCritiqueReport, query)
        
        if hasattr(self, "_last_extracted") and isinstance(self._last_extracted, ConceptualModel):
            det_errors = self._last_extracted.get_errors()
            if det_errors:
                parsed.is_valid = False
                from src.pipeline.stage2.models.conceptual_critique import SuggestedFix
                for err in det_errors:
                    parsed.fixes.append(SuggestedFix(description="Deterministic Structural Error", rationale=err))
                    
        return parsed, tokens
        
    def build_context(self, ctx: LoopContext) -> str:
        extracted = ctx.node_outputs.get("extractor")
        self._last_extracted = extracted
        facts_str = "\n".join([f"- [{f.id}] {f.fact}" for f in self._facts])
        if isinstance(extracted, ConceptualModel):
            return f"## INPUT\nFacts:\n{facts_str}\n\nGenerated Conceptual Model:\n{extracted.model_dump_json(indent=2)}"
        return f"## INPUT\nFacts:\n{facts_str}\n\nGenerated Conceptual Model:\n{{}}"
        
    def emit_history(self, output: LoopOutputModel, prior: Optional[LoopOutputModel], round_num: int, node: str) -> HistoryEntry:
        assert isinstance(output, ConceptualCritiqueReport)
        changes_summary = "valid" if output.is_valid else f"{len(output.fixes)} issues found"
        return HistoryEntry(round=round_num, node=node, changes_summary=changes_summary)


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------





async def run_conceptual_extractor_loop(
    facts: List[AtomicFact],
    nl_query: str,
    max_retries: int = 5,
    model: Optional[str] = None,
) -> Tuple[ConceptualModel, List[FixHistoryStep], int]:
    extractor = ConceptualExtractorLoopAgent(facts, nl_query, model)
    verifier = ConceptualVerifierLoopAgent(facts, model)
    
    config = LoopConfig(
        agents={
            "extractor": AgentRoleConfig(agent_factory=lambda: extractor, det_error_sources=["verifier"]),
            "verifier": AgentRoleConfig(agent_factory=lambda: verifier),
        },
        graph={
            "edges": [
                GraphEdge(from_node="extractor", to_node="verifier"),
                GraphEdge(from_node="verifier", to_node="extractor", condition=EdgeCondition(field="is_valid", op="eq", value=False)),
                GraphEdge(from_node="verifier", to_node="end"),
            ]
        },
        start_node="extractor",
        max_iter=max_retries,
    )
    
    result = await AgentLoop(config).run("")
    output = result.node_outputs.get("extractor")
    if not isinstance(output, ConceptualModel):
        output = ConceptualModel(entities=[], relationships=[], functional_dependencies=[])
        
    return output, [], result.total_tokens

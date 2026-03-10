from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.patch import CritiqueReport
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from typing import Tuple, Optional, List

PROMPT_FILE_URL = "g:/Personal Project/Autonomous_Data_Architect/v6/src/pipeline/stage3/agents/shard_refiner/prompt.txt"

def get_agent(style_guide: str, goal: str, model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        template = f.read()
    
    system_prompt = template.format(style_guide=style_guide, goal=goal)
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=CritiqueReport,
        model=model,
        name='Shard Refiner (Stage 3)'
    )

def refine_shard(
    segment: Schema,
    chunk_facts: List[AtomicFact],
    style_guide: str,
    goal: str,
    agent = None,
    model: Optional[str] = None
) -> Tuple[CritiqueReport, int]:
    """
    Audits a single schema segment against its source text chunk.
    """
    if not agent:
        agent = get_agent(style_guide, goal, model)
        
    fact_str = "\n".join([f"- {f.fact}" for f in chunk_facts])
    query = f"SOURCE FACTS FOR THIS SHARD:\n{fact_str}\n\nSCHEMA SEGMENT (JSON):\n{segment.model_dump_json(indent=2)}"
    
    report, tokens = get_response(
        agent=agent,
        output_structure=CritiqueReport,
        query=query
    )
    assert isinstance(report, CritiqueReport)
    
    return report, tokens

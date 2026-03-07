from typing import Tuple, Optional, List
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import SchemaSegment
from src.pipeline.stage3.models.patch import CritiqueReport, SchemaPatch

PROMPT_FILE_URL = "src/pipeline/stage3/agents/shard_auditor/prompt.txt"

def get_agent(style_guide: str, goal: str, model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        template = f.read()
    
    system_prompt = template.format(style_guide=style_guide, goal=goal)
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=CritiqueReport,
        model=model,
        name='Shard Auditor (Stage 3)'
    )

def audit_shard(
    segment: SchemaSegment,
    text_chunk: str,
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
        
    query = f"SOURCE TEXT CHUNK:\n{text_chunk}\n\nSCHEMA SEGMENT (JSON):\n{segment.model_dump_json(indent=2)}"
    
    report, tokens = get_response(
        agent=agent,
        output_structure=CritiqueReport,
        query=query
    )
    
    return report, tokens

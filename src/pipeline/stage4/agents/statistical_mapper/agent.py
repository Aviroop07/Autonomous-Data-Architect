from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import Schema
from src.orchestration.stage4.models import ExplicitCardinality, UnivariateKernel, StatisticalMapping
from pydantic import BaseModel
from typing import List, Optional, Tuple

class MappingExtractionList(BaseModel):
    items: List[StatisticalMapping]

PROMPT_FILE_URL = "g:/Personal Project/Autonomous_Data_Architect/v6/src/pipeline/stage4/agents/statistical_mapper/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=MappingExtractionList,
        model=model,
        name='Statistical Mapper (Stage 4 - Pass 3)'
    )

from src.orchestration.stage4.models import ExplicitCardinality, UnivariateKernel, StateTransition, EventProcess, ConditionalPolicy, StatisticalMapping

def map_distributions(
    global_schema: Schema,
    cardinalities: List[ExplicitCardinality],
    kernels: List[UnivariateKernel],
    transitions: List[StateTransition] = [],
    processes: List[EventProcess] = [],
    policies: List[ConditionalPolicy] = [],
    agent = None,
    model: Optional[str] = None
) -> Tuple[List[StatisticalMapping], int]:
    """
    Maps extracted behavioral and distributional logic to the unified global schema.
    """
    if not agent:
        agent = get_agent(model)
        
    card_str = "\n".join([f"- Fact {c.fact_id} ({c.entity_name}): {c.raw_count_description}" for c in cardinalities])
    kernel_str = "\n".join([f"- Fact {k.fact_id} ({k.kernel_type}): {k.description}" for k in kernels])
    trans_str = "\n".join([f"- Fact {t.fact_id} ({t.entity_name}.{t.target_column}): {t.logic}" for t in transitions])
    proc_str = "\n".join([f"- Fact {p.fact_id} ({p.event_table}): {p.intensity_logic}" for p in processes])
    poly_str = "\n".join([f"- Fact {po.fact_id} ({po.condition_source}): {po.policy_logic}" for po in policies])
    
    query = (
        f"REFINED GLOBAL SCHEMA:\n{global_schema.model_dump_json(indent=2)}\n\n"
        f"EXTRACTED CARDINALITIES:\n{card_str}\n\n"
        f"EXTRACTED KERNELS:\n{kernel_str}\n\n"
        f"EXTRACTED TRANSITIONS:\n{trans_str}\n\n"
        f"EXTRACTED PROCESSES:\n{proc_str}\n\n"
        f"EXTRACTED POLICIES:\n{poly_str}"
    )
    
    result, tokens = get_response(
        agent=agent,
        output_structure=MappingExtractionList,
        query=query
    )
    assert isinstance(result, MappingExtractionList)
    
    return result.items, tokens

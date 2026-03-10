from typing import Tuple, Optional, List
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.util.web_search import get_search_tool
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

PROMPT_FILE_URL = "src/pipeline/stage3/agents/domain_expert/prompt.txt"

def get_agent(domain: str, model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        template = f.read()
    
    system_prompt = template.replace("{domain}", domain)
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=[get_search_tool()],
        output_structure=None, # Returns raw string
        model=model,
        name='Domain Expert (Stage 3)'
    )

def generate_style_guide(
    domain: str,
    facts: List[AtomicFact],
    agent = None,
    model: Optional[str] = None
) -> Tuple[str, int]:
    """
    Researches industry standards and generates a Domain Style Guide.
    Inputs the exact facts required to ensure the style guide includes Analytical Competencies.
    """
    if not agent:
        agent = get_agent(domain, model)
    
    formatted_facts = "\n".join([f"{f.id}. {f.fact} [{f.tag}]" for f in facts])
    
    query = (
        f"Provide a technical data modeling style guide for the {domain} industry, specifically focusing on its core entities and relational patterns.\n\n"
        f"You MUST include a section titled 'Analytical Competencies' that dictates the schema architecture required to answer the following specific domain facts/goals:\n{formatted_facts}"
    )
    
    style_guide, tokens = get_response(
        agent=agent,
        output_structure=None,
        query=query
    )
    assert isinstance(style_guide, str)
    
    return style_guide, tokens


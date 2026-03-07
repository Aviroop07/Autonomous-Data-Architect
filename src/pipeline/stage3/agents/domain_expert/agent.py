from typing import Tuple, Optional
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.util.web_search import get_search_tool

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
    agent = None,
    model: Optional[str] = None
) -> Tuple[str, int]:
    """
    Researches industry standards and generates a Domain Style Guide.
    """
    if not agent:
        agent = get_agent(domain, model)
    
    query = f"Provide a technical data modeling style guide for the {domain} industry, specifically focusing on its core entities and relational patterns."
    
    style_guide, tokens = get_response(
        agent=agent,
        output_structure=None,
        query=query
    )
    
    return style_guide, tokens

import os
from typing import Tuple, Optional, List, Type, TypeVar
from pydantic import BaseModel
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.util.schema_utils import generate_pydantic_description
from .models import PromptStructure, PromptExample

PROMPT_FILE_URL = "src/pipeline/prompt_engineer/prompt.txt"

T = TypeVar("T", bound=BaseModel)

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    # Note: We don't specify output_structure here because it's dynamic in this agent
    return get_agent_(
        system_prompt=system_prompt,
        model=model,
        name='Prompt Engineer'
    )

def generate_system_prompt(
    context: str, 
    meta_agent: Optional[any] = None,
    output_model: Optional[Type[T]] = None,
    examples: Optional[List[PromptExample[T]]] = None,
    model: Optional[str] = None
) -> Tuple[str, int]:
    """
    Core utility to generate a structured system prompt and return it as a formatted string.
    """
    if not meta_agent:
        meta_agent = get_agent(model)

    # Determine the specific structure type (bind the Generic T)
    if output_model:
        # Create a concrete class to avoid Generic name issues with OpenAI (brackets not allowed)
        class PromptStructureConcrete(PromptStructure[output_model]):
            pass
        structure_to_use = PromptStructureConcrete
    else:
        structure_to_use = PromptStructure

    # We need to ensure the agent uses the correct output structure for this specific call
    # In langchain/openai, this is usually passed during invocation or set in the agent
    # Since our get_agent_ helper might have pre-bound it, we might need to be careful.
    # For now, we assume the agent can be used with get_response and a specific structure.
    
    query_parts = [f"### CONTEXT:\n{context}"]
    
    if output_model:
        schema_desc = generate_pydantic_description(output_model)
        query_parts.append(f"### OUTPUT SCHEMA (TECHNICAL METADATA):\n{schema_desc}")

    query = "\n\n".join(query_parts)
    
    parsed, tokens = get_response(
        agent=meta_agent,
        output_structure=structure_to_use,
        query=query
    )
    assert isinstance(parsed, structure_to_use)
    
    # Append manual examples if provided
    if examples:
        if parsed.examples is None:
            parsed.examples = []
        parsed.examples.extend(examples)
    
    return parsed.format_as_text(), tokens



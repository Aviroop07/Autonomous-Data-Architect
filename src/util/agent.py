import os
from typing import TypeVar, Type, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent

# ------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------

def get_model(model: Optional[str] = None) -> Union[ChatOpenAI, ChatOllama]:
    load_dotenv()
    PROVIDER = os.getenv("PROVIDER", "openai")

    if PROVIDER == "openai":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        MODEL = os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
        return ChatOpenAI(
            api_key=SecretStr(OPENAI_API_KEY),
            model=model or MODEL,
        )
    else:
        MODEL = os.getenv("OLLAMA_BASE_MODEL", "llama3")
        return ChatOllama(
            model=MODEL,
            reasoning=False
        )

# ------------------------------------------------------------------
# Generic Agent Factory
# ------------------------------------------------------------------

from src.util.schema_utils import generate_hierarchical_schema_description

T = TypeVar("T", bound=BaseModel)

def get_agent_(
    system_prompt: str,
    tools: Optional[list] = None,
    output_structure: Optional[Type[T]] = None,
    model: Optional[str] = None,
    name: Optional[str] = None,
):
    """
    Create a LangGraph agent. Supports tools and structured output.
    Automatically appends the hierarchical output format to the system prompt.
    """
    llm = get_model(model)

    # Dynamically append the output format if structure is provided
    if output_structure:
        output_format = generate_hierarchical_schema_description(output_structure)
        system_prompt = f"{system_prompt}\n\n## OUTPUT FORMAT\nReturn a JSON object matching this structure:\n{output_format}"

    agent = create_agent(
        model=llm,
        system_prompt=system_prompt,
        tools=tools,
        response_format=output_structure,
        name=name,
    )

    return agent

import os
from typing import TypeVar, Type, Optional, Any, Dict, List, Union
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable

from src.util.schema_utils import generate_hierarchical_schema_description

T = TypeVar("T", bound=BaseModel)

# Public alias used by all agent modules for their get_agent() return type
AgentType = Union["StructuredAgent", Runnable]

# ------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------

def get_model(model: Optional[str] = None, use_responses_api: bool = False) -> ChatOpenAI:
    """Returns a ChatOpenAI instance. Reads config from environment."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in .env or environment.")
    default_model = os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
    return ChatOpenAI(
        api_key=SecretStr(api_key),
        model=model or default_model,
        use_responses_api=use_responses_api,
    )


# ------------------------------------------------------------------
# Lightweight wrapper for structured-output-only agents
# ------------------------------------------------------------------

class StructuredAgent:
    """
    Wraps ChatOpenAI.with_structured_output() to provide the same
    ainvoke({"messages": [...]}) interface as langgraph agents.

    Response format:
        {
            "structured_response": <PydanticModel>,
            "messages": [SystemMessage, HumanMessage, AIMessage],
        }
    """

    def __init__(self, system_prompt: str, llm: ChatOpenAI, output_structure: Type[T]):
        self.system_prompt = system_prompt
        self.output_structure = output_structure
        self.chain = llm.with_structured_output(output_structure, include_raw=True, method="function_calling")

    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(input_dict.get("messages", []))
        result = await self.chain.ainvoke(messages)
        # result = {"raw": AIMessage, "parsed": PydanticModel, "parsing_error": ...}
        if result.get("parsing_error"):
            raise ValueError(
                f"Structured output parsing failed: {result['parsing_error']}"
            )
        return {
            "structured_response": result["parsed"],
            "messages": messages + [result["raw"]],
        }


# ------------------------------------------------------------------
# Generic Agent Factory
# ------------------------------------------------------------------

def get_agent_(
    system_prompt: str,
    tools: Optional[List[Any]] = None,
    output_structure: Optional[Type[T]] = None,
    model: Optional[str] = None,
    name: Optional[str] = None,
    use_responses_api: bool = False,
) -> AgentType:
    """
    Create an agent. Returns an object supporting ainvoke({"messages": [...]}).

    - Without tools: returns a StructuredAgent (lightweight, uses with_structured_output).
    - With tools: returns a langgraph react agent.

    In both cases, the OUTPUT FORMAT section is dynamically appended to the prompt.
    """
    llm = get_model(model, use_responses_api=use_responses_api)

    # Dynamically append the output schema description
    if output_structure:
        output_format = generate_hierarchical_schema_description(output_structure)
        system_prompt = (
            f"{system_prompt}\n\n"
            f"## OUTPUT FORMAT\n"
            f"Return a JSON object matching this structure:\n{output_format}"
        )

    if tools:
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
            response_format=output_structure,
            name=name or "agent",
        )
        return agent
    else:
        assert output_structure is not None, "output_structure is required for StructuredAgent"
        return StructuredAgent(
            system_prompt=system_prompt,
            llm=llm,
            output_structure=output_structure,
        )

from typing import Type, TypeVar, Tuple, Optional, Union
from pydantic import BaseModel
from langchain.messages import HumanMessage

T = TypeVar("T", bound=BaseModel)

def get_response(
    agent,
    output_structure: Optional[Type[T]],
    query: str
) -> Tuple[Union[T, str], int]:
    """
    Standardized caller for agents with optional Pydantic validation and token tracking.
    Returns (parsed_content, total_tokens)
    """
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        }
    )

    # No file IO or prints inside src/

    if output_structure:
        parsed = response.get("structured_response")
        if not isinstance(parsed, output_structure):
            raise TypeError(
                f"Expected {output_structure.__name__}, "
                f"got {type(parsed).__name__}"
            )
    else:
        # Return the content of the last message
        last_msg = response["messages"][-1]
        if isinstance(last_msg.content, list):
            # Find the first text block in the content list
            text_parts = [part["text"] for part in last_msg.content if isinstance(part, dict) and part.get("type") == "text"]
            parsed = "\n\n".join(text_parts)
        else:
            parsed = last_msg.content

    # Extract token usage from all AI messages' metadata
    total_tokens = 0
    for msg in response["messages"]:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total_tokens += msg.usage_metadata.get("total_tokens", 0)
        elif hasattr(msg, "response_metadata"):
            tokens = msg.response_metadata.get("token_usage", {}).get("total_tokens", 0)
            total_tokens += tokens
        elif isinstance(msg, dict):
             usage = msg.get("usage_metadata")
             if usage: total_tokens += usage.get("total_tokens", 0)
             else:
                 tokens = msg.get("response_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
                 total_tokens += tokens

    return parsed, total_tokens


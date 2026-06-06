from typing import Type, TypeVar, Tuple, Optional, Union
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, BaseMessage

T = TypeVar("T", bound=BaseModel)


async def get_response(
    agent,
    output_structure: Optional[Type[T]],
    query: Union[str, list],
) -> Tuple[Union[T, str], int]:
    """
    Standardized async caller for agents with Pydantic validation and token tracking.

    Works with both StructuredAgent (from agent.py) and langgraph agents.
    Both return {"structured_response": model, "messages": [...]}.

    Returns (parsed_content, total_tokens).
    """
    if isinstance(query, list):
        query = "\n".join([str(item) for item in query])

    # Sanitize query to clean UTF-8
    if isinstance(query, str):
        query = query.encode("utf-8", errors="ignore").decode("utf-8")

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )

    # Extract structured response or fallback to last message content
    if output_structure:
        parsed = response.get("structured_response")
        if parsed is None:
            raise TypeError(
                f"Agent did not return structured_response. "
                f"Expected {output_structure.__name__}."
            )
        if not isinstance(parsed, output_structure):
            raise TypeError(
                f"Expected {output_structure.__name__}, "
                f"got {type(parsed).__name__}"
            )
    else:
        last_msg = response["messages"][-1]
        if isinstance(last_msg.content, list):
            text_parts = [
                part["text"]
                for part in last_msg.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            parsed = "\n\n".join(text_parts)
        else:
            parsed = last_msg.content

    # Extract token usage from AI messages' metadata
    total_tokens = _extract_token_usage(response.get("messages", []))

    return parsed, total_tokens


def _extract_token_usage(messages: list) -> int:
    """Sum total_tokens from all messages that carry usage metadata."""
    total = 0
    for msg in messages:
        if isinstance(msg, BaseMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total += usage.get("total_tokens", 0)
            else:
                res_meta = getattr(msg, "response_metadata", {})
                total += res_meta.get("token_usage", {}).get("total_tokens", 0)
        elif isinstance(msg, dict):
            usage = msg.get("usage_metadata")
            if usage:
                total += usage.get("total_tokens", 0)
            else:
                total += (
                    msg.get("response_metadata", {})
                    .get("token_usage", {})
                    .get("total_tokens", 0)
                )
    return total

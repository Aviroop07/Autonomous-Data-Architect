from typing import Type, TypeVar, Tuple, Optional, Union
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, BaseMessage
from src.util.observability.llm_trace import get_active_trace_collector

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

    input_messages = [HumanMessage(content=query)]
    response = await agent.ainvoke({"messages": input_messages})

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
                f"Expected {output_structure.__name__}, got {type(parsed).__name__}"
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

    collector = get_active_trace_collector()
    if collector is not None:
        output_structure_name = (
            output_structure.__name__ if output_structure else "text"
        )
        parsed_response_type = type(parsed).__name__ if parsed is not None else ""
        collector.add_trace(
            agent_name=str(getattr(agent, "name", agent.__class__.__name__)),
            output_structure_name=output_structure_name,
            input_messages=input_messages,
            returned_messages=response.get("messages", []),
            token_usage=total_tokens,
            parsed_response_type=parsed_response_type,
        )

    assert parsed is not None
    return parsed, total_tokens


def _tokens_from_message(msg) -> int:
    if isinstance(msg, BaseMessage):
        usage = getattr(msg, "usage_metadata", None)
        res_meta = getattr(msg, "response_metadata", {})
    else:
        usage = msg.get("usage_metadata")
        res_meta = msg.get("response_metadata", {})
    if usage:
        return usage.get("total_tokens", 0)
    return res_meta.get("token_usage", {}).get("total_tokens", 0)


def _extract_token_usage(messages: list) -> int:
    """Sum total_tokens from all messages that carry usage metadata."""
    return sum(
        _tokens_from_message(msg)
        for msg in messages
        if isinstance(msg, (BaseMessage, dict))
    )

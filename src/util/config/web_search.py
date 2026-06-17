"""
Web search tool for use with langgraph agents.

OpenAI's built-in web_search_preview tool requires the Responses API.
When ChatOpenAI is configured with use_responses_api=True, this tool
spec can be passed directly to create_agent.

Integration point: src/pipeline/stage1/agents/context_enricher/agent.py
passes get_web_search_tool() to get_agent_(tools=[...], use_responses_api=True).
"""


def get_web_search_tool() -> dict:
    """
    Returns the OpenAI built-in web search tool specification.

    Usage with langgraph:
        from src.util.config.web_search import get_web_search_tool
        agent = create_agent(
            model=ChatOpenAI(model="gpt-4o", use_responses_api=True),
            tools=[get_web_search_tool()],
            ...
        )
    """
    return {
        "type": "web_search_preview",
        "search_context_size": "medium",
    }

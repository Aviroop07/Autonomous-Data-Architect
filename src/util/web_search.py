from typing import Tuple

def get_search_tool():
    """
    Returns the built-in web search tool specification for LangChain agents.
    """
    return {"type": "web_search_preview"}

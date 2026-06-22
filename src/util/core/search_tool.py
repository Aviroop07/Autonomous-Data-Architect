import asyncio
import hashlib
import threading
from typing import Optional

from langchain_core.tools import tool  # type: ignore[import]

# Session-level cache: keyed by sha256(query|max_results).
# Populated on first call, returned instantly on subsequent calls.
# Cleared between pipeline runs via clear_search_cache().
_CACHE: dict[str, list[dict]] = {}
_CACHE_LOCK = threading.Lock()


def _search(query: str, max_results: int) -> list[dict]:
    """Execute DDG search with session caching."""
    from ddgs import DDGS  # type: ignore[import]
    from ddgs.exceptions import DDGSException  # type: ignore[import]

    key = hashlib.sha256(f"{query}|{max_results}".encode()).hexdigest()[:16]
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    try:
        results = DDGS().text(query.strip(), max_results=max_results) or []
    except DDGSException:
        results = []
    except Exception:
        results = []

    with _CACHE_LOCK:
        _CACHE[key] = results
    return results


def _format_results(results: list[dict]) -> str:
    """Format results as numbered, source-tagged entries."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    Source: {r['href']}\n    {r['body']}")
    return "\n\n".join(lines)


def clear_search_cache() -> None:
    """Clear the session search cache. Call at pipeline start to avoid stale results."""
    with _CACHE_LOCK:
        _CACHE.clear()


async def prefetch_and_format_searches(
    queries: list[str],
    max_results: int = 5,
) -> str:
    """Run queries concurrently in threads and return a formatted SEARCH RESULTS block.

    Results are cached, so calling this multiple times with the same queries is free.
    Returns an empty string if queries is empty.
    """
    if not queries:
        return ""

    loop = asyncio.get_event_loop()

    async def _one(q: str) -> tuple[str, str]:
        results = await loop.run_in_executor(
            None, lambda: _search(q.strip(), max_results)
        )
        return q, _format_results(results)

    pairs = await asyncio.gather(*[_one(q) for q in queries])

    sections = [f"### Query: {q}\n{formatted}" for q, formatted in pairs]
    return "## SEARCH RESULTS\n\n" + "\n\n".join(sections)


@tool
def ddg_search(query: str, max_results: Optional[int] = None) -> str:
    """Search the web using DuckDuckGo.

    Returns numbered results with title, source URL, and snippet.
    Results are cached per session so retries and repeat queries are free.

    Args:
        query: Search query. Include domain context for better results.
               Good: "MRN medical record number healthcare definition"
               Bad:  "MRN"
        max_results: Number of results (1-10). Defaults to 5.
                     Use 2-3 for precise term definitions.
                     Use 5-8 for broad domain or architecture patterns.
    """
    if not query or not query.strip():
        return "No results: empty query."

    n = max(1, min(10, max_results if max_results is not None else 5))

    try:
        results = _search(query.strip(), n)
        return _format_results(results)
    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"


def get_ddg_search_tool():
    return ddg_search

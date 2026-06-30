"""Custom end-to-end web search engine for Stage 1 context enrichment.

Three concurrent sources per query:
  [W] Wikipedia (via DDG wikipedia backend) -- authoritative domain definitions
  [1] General web results (DDG text search)
  [+] Full-page extraction from top web result (DDGS.extract, if available)

Public API:
  prefetch_and_format_searches(queries, max_results=5) -> str
  clear_search_cache() -> None

Results are session-cached by (query, max_results) hash. Call clear_search_cache()
at pipeline start to avoid stale data across runs.
"""

import asyncio
import hashlib
import threading
from dataclasses import dataclass
from typing import Literal, Optional

# Session-level cache: keyed by sha256(query|max_results)[:16] -> formatted string
_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()

# Max chars from full-page extraction injected per result (keeps context window bounded)
_EXTRACT_MAX_CHARS = 1500


@dataclass
class _Result:
    title: str
    url: str
    snippet: str
    source: Literal["web", "wikipedia"]


def _search_web(query: str, max_results: int) -> list[_Result]:
    """DDG general web search."""
    try:
        from ddgs import DDGS  # type: ignore[import]

        results = DDGS(timeout=6).text(query.strip(), max_results=max_results) or []
        return [
            _Result(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
                source="web",
            )
            for r in results
        ]
    except Exception:
        return []


def _search_wiki(query: str) -> list[_Result]:
    """DDG search scoped to Wikipedia for authoritative domain knowledge."""
    try:
        from ddgs import DDGS  # type: ignore[import]

        wiki_query = f"{query.strip()} site:en.wikipedia.org"
        results = DDGS(timeout=6).text(wiki_query, max_results=2) or []
        return [
            _Result(
                title=r.get("title", "").removesuffix(" - Wikipedia"),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
                source="wikipedia",
            )
            for r in results
        ]
    except Exception:
        return []


def _extract_page(url: str) -> Optional[str]:
    """Extract clean Markdown text from a URL via DDGS.extract().

    Returns None if extraction is unavailable, times out, or produces empty content.
    """
    if not url:
        return None
    try:
        from ddgs import DDGS  # type: ignore[import]

        raw = DDGS(timeout=5).extract(url, fmt="text_markdown")
        if isinstance(raw, dict) and raw.get("content"):
            return raw["content"][:_EXTRACT_MAX_CHARS]
        return None
    except Exception:
        return None


def _format_query_section(
    query: str,
    wiki: list[_Result],
    web: list[_Result],
    extracted: Optional[str],
) -> str:
    """Merge sources into a formatted section for one query."""
    lines = [f"### Query: {query}"]

    for r in wiki:
        lines.append(f"[W] {r.title}")
        lines.append(f"    Source: {r.url}")
        lines.append(f"    {r.snippet}")

    for i, r in enumerate(web, 1):
        snippet = extracted if (i == 1 and extracted) else r.snippet
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    Source: {r.url}")
        lines.append(f"    {snippet}")

    if not wiki and not web:
        lines.append("    No results found.")

    return "\n".join(lines)


def clear_search_cache() -> None:
    """Clear the session search cache. Call at pipeline start to avoid stale results."""
    with _CACHE_LOCK:
        _CACHE.clear()


async def prefetch_and_format_searches(
    queries: list[str],
    max_results: int = 5,
) -> str:
    """Run all queries concurrently and return a formatted '## SEARCH RESULTS' block.

    Sources per query (concurrent):
      - Wikipedia-scoped DDG search (authoritative, labelled [W])
      - General DDG web search (labelled [1], [2], ...)
      - Full-page extraction from top web result (appended to [1] snippet when available)

    Results are session-cached -- repeat calls with the same queries are free.
    Returns an empty string when queries is empty.
    """
    if not queries:
        return ""

    loop = asyncio.get_running_loop()

    async def _one(query: str) -> str:
        q = query.strip()
        if not q:
            return ""

        key = hashlib.sha256(f"{q}|{max_results}".encode()).hexdigest()[:16]
        with _CACHE_LOCK:
            if key in _CACHE:
                return _CACHE[key]

        # Web and Wikipedia searches run concurrently
        wiki_results, web_results = await asyncio.gather(
            loop.run_in_executor(None, _search_wiki, q),
            loop.run_in_executor(None, _search_web, q, max_results),
        )

        # Full-page extraction from top web result (best-effort; skip if slow/unavailable)
        extracted: Optional[str] = None
        if web_results and web_results[0].url:
            extracted = await loop.run_in_executor(
                None, _extract_page, web_results[0].url
            )

        formatted = _format_query_section(q, wiki_results, web_results, extracted)

        with _CACHE_LOCK:
            _CACHE[key] = formatted
        return formatted

    sections = await asyncio.gather(*[_one(q) for q in queries])
    sections = [s for s in sections if s]
    if not sections:
        return ""
    return "## SEARCH RESULTS\n\n" + "\n\n".join(sections)

"""Custom end-to-end web search engine for Stage 1 context enrichment.

Three concurrent sources per query:
  [W] Wikipedia (via DDG wikipedia backend) -- authoritative domain definitions
  [1] General web results (DDG text search)
  [+] Full-page extraction from top web result (DDGS.extract, if available)

Public API:
  EvidenceStore: fetch() and resolve() to get evidence and global tags.
  clear_search_cache() -> None

Results are session-cached by (query, max_results) hash. Call clear_search_cache()
at pipeline start to avoid stale data across runs.
"""

import asyncio
import hashlib
import threading
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict

# Session-level cache: keyed by sha256(query|max_results)[:16] -> tuple of (wiki, web, extracted)
_CACHE: Dict[str, Tuple[List["_Result"], List["_Result"], Optional[str]]] = {}
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


def clear_search_cache() -> None:
    """Clear the session search cache. Call at pipeline start to avoid stale results."""
    with _CACHE_LOCK:
        _CACHE.clear()


@dataclass
class EvidenceSnippet:
    tag: str
    query: str
    title: str
    url: str
    text: str
    source: Literal["web", "wikipedia"]


@dataclass
class FetchResult:
    formatted: str


class EvidenceStore:
    def __init__(self):
        self._by_tag: dict[str, EvidenceSnippet] = {}
        self._tag_counter = 1

    async def fetch(self, queries: list[str], max_results: int = 5) -> FetchResult:
        """Run all queries concurrently and return a formatted '## SEARCH RESULTS' block.
        Also populates the tag->snippet map for later resolution by the auditor.
        """
        if not queries:
            return FetchResult(formatted="")

        loop = asyncio.get_running_loop()
        sections = []

        async def _fetch_one(query: str) -> Optional[Tuple[str, List[_Result], List[_Result], Optional[str]]]:
            q = query.strip()
            if not q:
                return None

            key = hashlib.sha256(f"{q}|{max_results}".encode()).hexdigest()[:16]
            
            with _CACHE_LOCK:
                if key in _CACHE:
                    wiki, web, extracted = _CACHE[key]
                    return q, wiki, web, extracted

            wiki_results, web_results = await asyncio.gather(
                loop.run_in_executor(None, _search_wiki, q),
                loop.run_in_executor(None, _search_web, q, max_results),
            )

            extracted: Optional[str] = None
            if web_results and web_results[0].url:
                extracted = await loop.run_in_executor(
                    None, _extract_page, web_results[0].url
                )

            with _CACHE_LOCK:
                _CACHE[key] = (wiki_results, web_results, extracted)
                
            return q, wiki_results, web_results, extracted

        results = await asyncio.gather(*[_fetch_one(q) for q in queries])

        for res in results:
            if not res:
                continue
            
            q, wiki, web, extracted = res
            lines = [f"### Query: {q}"]
            found = False

            for r in wiki:
                tag = f"E{self._tag_counter}"
                self._tag_counter += 1
                self._by_tag[tag] = EvidenceSnippet(
                    tag=tag, query=q, title=r.title, url=r.url, text=r.snippet, source="wikipedia"
                )
                lines.append(f"[{tag}] {r.title}")
                lines.append(f"    Source: {r.url}")
                lines.append(f"    {r.snippet}")
                found = True

            for i, r in enumerate(web):
                snippet = extracted if (i == 0 and extracted) else r.snippet
                tag = f"E{self._tag_counter}"
                self._tag_counter += 1
                self._by_tag[tag] = EvidenceSnippet(
                    tag=tag, query=q, title=r.title, url=r.url, text=snippet, source="web"
                )
                lines.append(f"[{tag}] {r.title}")
                lines.append(f"    Source: {r.url}")
                lines.append(f"    {snippet}")
                found = True

            if not found:
                lines.append("    No results found.")

            sections.append("\n".join(lines))

        if not sections:
            return FetchResult(formatted="")

        formatted_block = "## SEARCH RESULTS\n\n" + "\n\n".join(sections)
        return FetchResult(formatted=formatted_block)

    def resolve(self, tags: list[str]) -> list[EvidenceSnippet]:
        """Resolve evidence refs to their genuine snippets."""
        return [self._by_tag[t] for t in tags if t in self._by_tag]

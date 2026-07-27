"""Chunk facts by the target model's real context budget.

Stage 2 consumes chunks by sending each one to an ER-extraction agent, so the
only thing a chunk boundary has to guarantee is that the chunk FITS. That makes
chunking a packing problem, not a clustering problem:

    everything fits in one prompt  ->  one chunk, no work
    otherwise                      ->  pack segments until the budget is spent

This replaces a Dirichlet-process Gibbs sampler (bayesian_chunker.py) on the
default path. The sampler was measured returning ONE chunk for every input
tried, at 12,000 fixed sweeps of O(S^2) each, and its degeneracy was not a
tuning problem:

  - lambda was chosen by RANGE rather than discriminative power, so it handed
    68% of the fused weight to an adjacency signal whose median was 0.004;
  - once it merged everything there were no between-cluster pairs left, so the
    "different" Beta could never be re-estimated and it could not split back
    out (the `[BetaEnsemble] 0 informative` warnings);
  - the eigengap of the normalized Laplacian independently said k=1, because
    entity segments drawn from ONE domain document sit in a narrow similarity
    band (0.53-0.84 measured) -- there is no structure at that signal to find.

Crucially, one chunk was the RIGHT answer for the specs measured. This module
reaches the same answer in microseconds, for a reason that can be stated in one
line, and it degrades correctly in both directions: 1 fact needs no packing,
and a spec ten times the context window gets ceil(tokens/budget) chunks.

bayesian_chunker.py is retained and still selectable, since it is the paper's
original method: set AblationConfig.use_bayesian_chunker (or use
AblationConfig.bayesian_chunking()) for a like-for-like comparison. It is no
longer the default path.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.chunk import ChunkedPlan

logger = logging.getLogger(__name__)

# Chars per token. The same crude divisor sharding_ilp.py uses for its own
# budget arithmetic -- deliberately consistent with it rather than independently
# "better", so the two budget calculations cannot silently disagree.
_CHARS_PER_TOKEN = 4.0

# What the ER-extraction prompt costs before any fact is added: the system
# prompt, the schema-so-far, and the output-format block.
_DEFAULT_PROMPT_OVERHEAD_TOKENS = 6000

# Leaves room for the model's own OUTPUT, which shares the context window.
_DEFAULT_SAFETY_MARGIN = 0.6


def estimate_fact_tokens(fact: AtomicFact) -> int:
    """Rough token cost of one fact as it appears in a prompt. Counts the
    rendered text plus a small per-line overhead for the id/tag decoration."""
    text = fact.fact or ""
    return int(len(text) / _CHARS_PER_TOKEN) + 8


def _group_into_segments(
    facts: Sequence[AtomicFact],
) -> List[List[AtomicFact]]:
    """Facts sharing a source span belong together -- splitting mid-sentence
    would hand two ER agents halves of the same statement. Enrichment facts
    (start_char < 0) have no span and each stand alone.

    Ordered by document position so packing preserves reading order, which
    keeps related entities adjacent without needing a similarity model.
    """
    groups: Dict[Tuple[int, int], List[AtomicFact]] = defaultdict(list)
    for f in facts:
        groups[(f.start_char, f.end_char)].append(f)
    return [groups[k] for k in sorted(groups, key=lambda k: (k[0], k[1]))]


class BudgetChunker:
    """Packs facts into the fewest chunks that each fit the model's context.

    `budget_tokens` is normally derived from the target model's live-queried
    context window. It is injectable so tests -- and callers who already know
    their budget -- never have to touch the network.
    """

    def __init__(
        self,
        budget_tokens: Optional[int] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "",
        prompt_overhead_tokens: int = _DEFAULT_PROMPT_OVERHEAD_TOKENS,
        safety_margin: float = _DEFAULT_SAFETY_MARGIN,
    ) -> None:
        self._explicit_budget = budget_tokens
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._overhead = prompt_overhead_tokens
        self._margin = safety_margin

    def _resolve_budget(self) -> Optional[int]:
        """Tokens available for FACTS in one prompt, or None if it cannot be
        determined -- in which case the caller falls back to a single chunk,
        which is what the pipeline did before this module existed."""
        if self._explicit_budget is not None:
            return self._explicit_budget
        try:
            from src.util.core.agent import _detect_provider
            from src.util.core.context_window import get_context_window

            provider, api_key, _base_url, default_model = _detect_provider()
            window = get_context_window(
                self._provider or provider,
                self._model or default_model,
                api_key=self._api_key or api_key,
            )
        except Exception as exc:  # network, missing key, unknown model
            logger.warning(
                "[BudgetChunker] could not determine the context window (%s); "
                "falling back to a single chunk.",
                exc,
            )
            return None
        budget = int(window * self._margin) - self._overhead
        return budget if budget > 0 else None

    def fit(self, facts: List[AtomicFact]) -> ChunkedPlan:
        if len(facts) <= 1:
            return ChunkedPlan(core_modeling_facts=list(facts), chunks=[list(facts)])

        total = sum(estimate_fact_tokens(f) for f in facts)
        budget = self._resolve_budget()

        if budget is None or total <= budget:
            logger.info(
                "[BudgetChunker] %d facts (~%d tokens) fit in one prompt "
                "(budget=%s) -- 1 chunk.",
                len(facts),
                total,
                budget if budget is not None else "unknown",
            )
            return ChunkedPlan(core_modeling_facts=list(facts), chunks=[list(facts)])

        segments = _group_into_segments(facts)
        chunks: List[List[AtomicFact]] = []
        current: List[AtomicFact] = []
        current_tokens = 0

        for segment in segments:
            seg_tokens = sum(estimate_fact_tokens(f) for f in segment)
            # A single segment larger than the whole budget cannot be split
            # without cutting a source span in half, so it goes out oversized
            # and alone rather than being silently truncated.
            if seg_tokens > budget and not current:
                logger.warning(
                    "[BudgetChunker] one segment is ~%d tokens, over the %d "
                    "budget; emitting it alone rather than splitting a span.",
                    seg_tokens,
                    budget,
                )
                chunks.append(list(segment))
                continue
            if current and current_tokens + seg_tokens > budget:
                chunks.append(current)
                current, current_tokens = [], 0
            current.extend(segment)
            current_tokens += seg_tokens

        if current:
            chunks.append(current)

        logger.info(
            "[BudgetChunker] %d facts (~%d tokens) exceed the %d-token budget "
            "-- packed %d segments into %d chunks.",
            len(facts),
            total,
            budget,
            len(segments),
            len(chunks),
        )
        return ChunkedPlan(core_modeling_facts=list(facts), chunks=chunks)

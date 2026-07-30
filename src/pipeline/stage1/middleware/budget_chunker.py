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
import os
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

# How much fact material ONE er_extraction call can faithfully model, which is a
# completely different quantity from what fits in a prompt and is the one that
# actually binds.
#
# MEASURED BY SWEEP. Stage 1 held fixed (the same 121 facts from a 41-table
# cycling-club spec), varying ONLY this number, so chunking is the sole variable.
# Ground truth is 41 tables; one run per point:
#
#   budget  chunks  facts/chunk  tables  unrepresented  Stage 2 tokens
#     2992       1          121       9       86 / 121      (baseline)
#     1800       2       73, 48      41        7 / 121        252,862
#     1100       3   46, 40, 35      32       20 / 121        283,589
#      900       4   39,34,31,17     43        3 / 121        369,921
#      700       5   31,27,26,25,12  41        3 / 121        447,335
#
# ONE effect here is large and unambiguous: putting all 121 facts in a single
# call collapses the schema to 9 tables, and ANY split recovers 32-43. Where
# recall looks complete, it is: every ground-truth table not matched by exact
# name is a pure synonym of a predicted one (AUDIT_LOG/AUDIT_ENTRY,
# COMPONENT/BIKE_COMPONENT, PLANNED_WORKOUT/WORKOUT, KIT_ORDER/ORDER, ...). So
# the er_extractor was never weak -- it was handed a 41-entity domain in one
# call, and one call saturates near 9-16 entities however much material it gets.
#
# The sweep does NOT identify an optimum, and it would be wrong to read one off
# this table. The results are NON-MONOTONIC -- 2 chunks beat 3 on both coverage
# and cost -- which at one run per point means the differences between 2, 3, 4
# and 5 chunks are not separable from run-to-run variance. Independent evidence
# that the variance is large: the SAME spec yielded 15 facts on one Stage 1 run
# and 121 on another.
#
# So 900 is a DEFENSIBLE CHOICE, not a calibrated optimum: it gave the best
# measured coverage (3 unrepresented) and sits mid-range, comfortably clear of
# the one-call failure. Cost is the live tradeoff -- 2 chunks did nearly as well
# for 32% fewer tokens -- and settling that properly needs repeats per point,
# which has not been done. The failure mode of setting this too HIGH is silent
# under-modelling that raises no error whatsoever, so it should move on measured
# evidence only.
#
# THIS NUMBER IS MODEL-SPECIFIC, and that is the important caveat. It was
# measured on gemini-2.5-flash on 2026-07-30. Extraction capacity is a property
# of the MODEL's ability to hold a domain in one structured answer, so a
# different model -- a larger one, a reasoning one, a quantized self-hosted one
# -- will have a different ceiling, and this default will be wrong for it in
# whichever direction.
#
# Deliberately NOT expressed as a fraction of the context window, tempting as
# that is for auto-scaling: capacity is about modelling ability, not window
# size. The measurement above is the counter-example -- gemini-2.5-flash has a
# ~1M-token window and still saturates near 9-16 entities, so window size
# predicts capacity badly and scaling off it would silently reintroduce the exact
# bug this constant fixes.
#
# Three things keep the model-dependence honest rather than hidden:
#   1. it is overridable per deployment via EXTRACTION_CAPACITY_TOKENS;
#   2. the resolved value and its origin are logged on every run;
#   3. Stage 2 warns when the unrepresented-fact fraction is high, which is the
#      model-INDEPENDENT symptom of this value being too large for the model in
#      use (see _compute_uncovered_facts).
# The principled long-term fix is to stop hardcoding it at all and re-chunk
# adaptively on that coverage signal, which needs no per-model calibration.
_MEASURED_ON = "gemini-2.5-flash, 2026-07-30"
_FALLBACK_EXTRACTION_CAPACITY_TOKENS = 900
_CAPACITY_ENV_VAR = "EXTRACTION_CAPACITY_TOKENS"


def _resolve_default_capacity() -> int:
    """The configured per-call extraction capacity, env override winning.

    Read once at import: a run does not change its own environment, and this
    keeps the value a plain int so callers can still pass an explicit one.
    """
    raw = os.getenv(_CAPACITY_ENV_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "[BudgetChunker] %s=%r is not an integer; using the measured "
                "default of %d tokens instead.",
                _CAPACITY_ENV_VAR,
                raw,
                _FALLBACK_EXTRACTION_CAPACITY_TOKENS,
            )
            return _FALLBACK_EXTRACTION_CAPACITY_TOKENS
        if value <= 0:
            logger.info(
                "[BudgetChunker] %s=%d disables the extraction-capacity ceiling; "
                "chunking falls back to the context window alone, which is the "
                "pre-2026-07-30 behaviour and never splits in practice.",
                _CAPACITY_ENV_VAR,
                value,
            )
            return value
        logger.info(
            "[BudgetChunker] extraction capacity overridden to %d tokens via %s "
            "(built-in default %d was measured on %s).",
            value,
            _CAPACITY_ENV_VAR,
            _FALLBACK_EXTRACTION_CAPACITY_TOKENS,
            _MEASURED_ON,
        )
        return value
    return _FALLBACK_EXTRACTION_CAPACITY_TOKENS


_DEFAULT_EXTRACTION_CAPACITY_TOKENS = _resolve_default_capacity()

# Leaves room for the model's own OUTPUT, which shares the context window.
_DEFAULT_SAFETY_MARGIN = 0.6


def estimate_fact_tokens(fact: AtomicFact) -> int:
    """Rough token cost of one fact as it appears in a prompt. Counts the
    rendered text plus a small per-line overhead for the id/tag decoration."""
    text = fact.fact or ""
    return int(len(text) / _CHARS_PER_TOKEN) + 8


def _split_oversized_segment(
    segment: Sequence[AtomicFact], budget: int
) -> List[List[AtomicFact]]:
    """Pack one over-budget segment's facts into budget-sized pieces.

    Only reached when a single source span carries more facts than a prompt can
    hold. Splits at fact boundaries in document order, so each piece is a
    contiguous run of the original segment and no fact is ever divided.

    A single fact larger than the budget on its own goes out alone and
    oversized: that IS genuinely unsplittable, since cutting inside one fact
    would truncate a statement rather than merely separate two.
    """
    pieces: List[List[AtomicFact]] = []
    current: List[AtomicFact] = []
    current_tokens = 0
    for fact in segment:
        fact_tokens = estimate_fact_tokens(fact)
        if current and current_tokens + fact_tokens > budget:
            pieces.append(current)
            current, current_tokens = [], 0
        if fact_tokens > budget and not current:
            logger.warning(
                "[BudgetChunker] a single fact is ~%d tokens, over the %d "
                "budget; emitting it alone rather than truncating it.",
                fact_tokens,
                budget,
            )
            pieces.append([fact])
            continue
        current.append(fact)
        current_tokens += fact_tokens
    if current:
        pieces.append(current)
    return pieces


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
        extraction_capacity_tokens: Optional[int] = _DEFAULT_EXTRACTION_CAPACITY_TOKENS,
    ) -> None:
        self._explicit_budget = budget_tokens
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._overhead = prompt_overhead_tokens
        self._margin = safety_margin
        self._capacity = extraction_capacity_tokens

    def _resolve_budget(self) -> Optional[int]:
        """Tokens available for FACTS in one prompt.

        Two independent ceilings, and the SMALLER one binds:

          context window   -- can this prompt be sent at all?
          extraction capacity -- can one call model this much at once?

        Only the first existed before, and it is the wrong one to chunk by: it
        was measured at 623,145 tokens against a 2,992-token input, so it never
        split anything and the whole shard-and-merge path was unreachable. See
        _DEFAULT_EXTRACTION_CAPACITY_TOKENS for the measurement that motivates
        the second.

        An explicit `budget_tokens` still wins outright and is NOT capped: it is
        how ablations and calibration sweeps address this module directly, and
        silently clamping a caller's stated budget would make those experiments
        measure something other than what they asked for.

        Returns None only when NEITHER ceiling is knowable, in which case the
        caller falls back to a single chunk as it did before this module existed.
        """
        if self._explicit_budget is not None:
            return self._explicit_budget if self._explicit_budget > 0 else None

        context_budget: Optional[int] = None
        try:
            from src.util.core.agent import _detect_provider
            from src.util.core.context_window import get_context_window

            provider, api_key, _base_url, default_model = _detect_provider()
            window = get_context_window(
                self._provider or provider,
                self._model or default_model,
                api_key=self._api_key or api_key,
            )
            candidate = int(window * self._margin) - self._overhead
            context_budget = candidate if candidate > 0 else None
        except Exception as exc:  # network, missing key, unknown model
            # No longer fatal: the capacity ceiling below does not need the
            # network, so an unknown context window degrades to "chunk by
            # capacity" rather than to one unbounded chunk.
            logger.warning(
                "[BudgetChunker] could not determine the context window (%s); "
                "falling back to the extraction-capacity ceiling alone.",
                exc,
            )

        capacity = self._capacity if self._capacity and self._capacity > 0 else None
        candidates = [b for b in (context_budget, capacity) if b is not None]
        if not candidates:
            return None
        budget = min(candidates)
        if capacity is not None and budget == capacity and context_budget is not None:
            logger.debug(
                "[BudgetChunker] extraction capacity (%d tokens) binds, well "
                "inside the context budget (%d).",
                capacity,
                context_budget,
            )
        return budget

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
            # A segment bigger than the whole budget is split at FACT
            # boundaries. The previous behaviour emitted it whole, reasoning
            # that splitting would cut a source span in half -- but a segment
            # is a GROUP OF FACTS, and every fact keeps its own segment_text,
            # start_char and end_char. Splitting the group cuts no span; it
            # only costs co-location, while emitting it whole produces a chunk
            # OVER the model's context budget, which is the single failure
            # chunking exists to prevent.
            #
            # It also had a silent path: the old guard was `if seg_tokens >
            # budget and not current`, so when anything was already pending the
            # warning never fired and the oversized segment still went out
            # whole. Measured -- one small segment followed by a 288-token
            # segment against a 144-token budget produced a 288-token chunk
            # with no warning at all.
            if seg_tokens > budget:
                if current:
                    chunks.append(current)
                    current, current_tokens = [], 0
                pieces = _split_oversized_segment(segment, budget)
                logger.warning(
                    "[BudgetChunker] one segment is ~%d tokens, over the %d "
                    "budget; split into %d piece(s) at fact boundaries. Facts "
                    "from one source span are no longer co-located, which is "
                    "the lesser cost -- the alternative is a chunk that cannot "
                    "fit its prompt.",
                    seg_tokens,
                    budget,
                    len(pieces),
                )
                chunks.extend(pieces)
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

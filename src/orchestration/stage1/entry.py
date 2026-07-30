from typing import List, Optional, Tuple
import logging

from src.orchestration.stage1.loop_config import (
    make_enrichment_loop_config,
    make_stage1_loop_config,
)
from src.orchestration.stage1.models import Output
from src.pipeline.stage1.agents.tagger.agent import tag_facts
from src.pipeline.stage1.middleware.external_context_filter import (
    ExternalFactFilterResult,
    filter_external_facts,
)
from src.pipeline.stage1.middleware.tag_normalization import normalize_stage1_tags
from src.pipeline.stage1.agents.spec_completeness_auditor.agent import (
    audit_completeness,
)
from src.pipeline.stage1.models.coverage_report import SpecGap
from src.pipeline.stage1.models.context_audit import ContextAuditAttempt
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import (
    EnrichedNL,
    RephrasedOutput,
    convert_to_atomic,
)
from src.util.config.ablation import AblationConfig
from src.util.core.agent_provider import AgentProvider
from src.util.core.token_budget import CHARS_PER_TOKEN as _CHARS_PER_TOKEN
from src.util.observability.llm_trace import (
    LLMTraceCollector,
    activate_trace_collector,
    reset_trace_collector,
)
from src.util.core.search_tool import EvidenceStore, clear_search_cache
from src.util.orchestration.loop import AgentLoop
from src.pipeline.stage2.models.chunk import ChunkedPlan

logger = logging.getLogger(__name__)

# Fallback ceiling, used only when the model's real context window cannot be
# resolved (offline, no key, unknown model). Grounded in the largest thing this
# project actually feeds Stage 1 -- the benchmark's longest specification is
# 12,197 characters -- with room to spare, so it rejects a pasted book and
# nothing legitimate.
NL_MAX_CHARS_FALLBACK = 50_000

# Share of the context window the NL itself may occupy. The rest carries the
# system prompt, the tool schema, and the structured output, all of which are
# larger than the input for this stage.
_NL_WINDOW_SHARE = 0.25


def _max_nl_chars(model: Optional[str]) -> int:
    """Resolve the NL length ceiling from the model's real context window.

    This was a bare `NL_MAX_CHARS = 4000` with no comment, and it was wrong by
    roughly two orders of magnitude against the actual constraint: a 12,197-char
    specification is about 3,000 tokens, against a live-queried window of some
    623,000. The consequence was not a slow run but an outright `ValueError` --
    54 of the benchmark's 150 cases exceeded it, INCLUDING ALL 14 of the cases
    above `max_tables_per_shard=18`, which exist specifically to exercise the
    cross-shard path. The dataset built to stress the pipeline could not be fed
    to it.

    Same failure shape as the chunker's token budget: a limit chosen far from the
    quantity it was meant to bound, so it silently governed the wrong thing.
    Deriving it keeps the two in step as models change.
    """
    try:
        from src.util.core.agent import _detect_provider
        from src.util.core.context_window import get_context_window

        provider, api_key, _base_url, default_model = _detect_provider()
        window = get_context_window(provider, model or default_model, api_key=api_key)
    except Exception as exc:  # network, missing key, unknown model
        logger.warning(
            "[Stage 1] could not resolve the context window (%s); falling back to "
            "a %d-character NL ceiling.",
            exc,
            NL_MAX_CHARS_FALLBACK,
        )
        return NL_MAX_CHARS_FALLBACK
    return max(NL_MAX_CHARS_FALLBACK, int(window * _NL_WINDOW_SHARE * _CHARS_PER_TOKEN))


async def orchestrate(
    nl_description: str,
    max_retries: int = 5,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    trace_collector: Optional[LLMTraceCollector] = None,
    provider: Optional[AgentProvider] = None,
    evidence_store: Optional[EvidenceStore] = None,
) -> Tuple[Output, int]:
    """`provider` selects how LLM agents are built (default: live API-backed --
    see util/core/agent_provider.py). `evidence_store` selects where enrichment
    evidence comes from (default: live web search). Both are the run's external
    dependencies; everything else here is internal."""
    limit = _max_nl_chars(model)
    if not nl_description.strip():
        logger.error(
            "[Stage 1] Empty or whitespace-only NL description; returning empty plan."
        )
        empty_plan = ChunkedPlan(core_modeling_facts=[], chunks=[])
        return Output(
            final_facts=[],
            domain="Unknown",
            analytical_goal="General Purpose",
            iterations=[],
            original_nl=nl_description,
            plan=empty_plan,
            token_usage=0,
        ), 0

    if len(nl_description) > limit:
        raise ValueError(
            f"NL description is {len(nl_description)} characters "
            f"(limit: {limit}, derived from the model's context window). "
            "Trim the input before running."
        )
    trace_token = (
        activate_trace_collector(trace_collector)
        if trace_collector is not None
        else None
    )
    try:
        return await _orchestrate_impl(
            nl_description=nl_description,
            model=model,
            ablation_config=ablation_config,
            provider=provider,
            evidence_store=evidence_store,
        )
    finally:
        if trace_token is not None:
            reset_trace_collector(trace_token)


async def _orchestrate_impl(
    nl_description: str,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    provider: Optional[AgentProvider] = None,
    evidence_store: Optional[EvidenceStore] = None,
) -> Tuple[Output, int]:
    clear_search_cache()
    logger.info("[Stage 1] Initializing extraction agent loop...")
    loop_config = make_stage1_loop_config(
        nl_description, model=model, provider=provider
    )
    result = await AgentLoop(loop_config).run(nl_description)
    logger.info(
        f"[Stage 1] Extraction loop completed in {result.iteration_count} iterations. Total tokens so far: {result.total_tokens}"
    )

    raw_extraction = result.node_outputs.get("extractor")
    extraction_output: RephrasedOutput = (
        raw_extraction
        if isinstance(raw_extraction, RephrasedOutput)
        else RephrasedOutput(segments=[])
    )
    if not isinstance(raw_extraction, RephrasedOutput):
        logger.warning(
            f"[Stage 1] Loop exhausted after {result.iteration_count} iterations "
            f"with no accepted extractor output."
        )
    elif result.det_errors_exhausted:
        # The loop produced a RephrasedOutput but gave up while the validator was
        # still reporting errors, so this extraction is structurally invalid --
        # bad verbatim spans, unresolved references. The isinstance check above
        # passes in that case, so without this branch a degraded extraction was
        # indistinguishable from a clean one in both the log and the Output model.
        logger.error(
            "[Stage 1] Extraction loop exhausted its retry budget after %d "
            "iteration(s) with UNRESOLVED validator errors; the %d extracted "
            "fact(s) below are structurally unverified.",
            result.iteration_count,
            len(extraction_output.flat_facts),
        )

    total_tokens = result.total_tokens

    extracted_facts: List[RawFact] = extraction_output.flat_facts
    enrichment_filter_report = ExternalFactFilterResult()
    context_audit_trail: List[ContextAuditAttempt] = []

    search_suggestions: List[str] = []
    raw_report = result.node_outputs.get("verifier")
    last_report = raw_report if isinstance(raw_report, IntegrityReport) else None
    if last_report is not None and last_report.search_suggestions:
        search_suggestions.extend(last_report.search_suggestions)

    coverage_report, t_cov = await audit_completeness(
        extracted_facts,
        domain=extraction_output.domain or "Unknown",
        analytical_goal=extraction_output.analytical_goal or "Unknown",
        verifier_suggestions=search_suggestions,
        model=model,
        provider=provider,
    )
    total_tokens += t_cov

    gate_open = coverage_report.is_underspecified
    enrichment_enabled = ablation_config is None or ablation_config.enable_enrichment

    if enrichment_enabled and gate_open:
        gate_gaps = coverage_report.gaps_for_enrichment()
        logger.info(
            f"[Stage 1] Coverage gate OPEN: {len(gate_gaps)} gap(s) found (minor gaps included). "
            f"Starting context enrichment loop..."
        )
        external_facts, t_enrich = await _run_context_enrichment_loop(
            facts=extracted_facts,
            gaps=gate_gaps,
            model=model,
            audit_trail=context_audit_trail,
            provider=provider,
            evidence_store=evidence_store,
        )
        total_tokens += t_enrich
        enrichment_filter_report = filter_external_facts(
            external_facts, extracted_facts
        )
        external_facts = enrichment_filter_report.accepted_facts
        if enrichment_filter_report.rejected_facts:
            logger.info(
                f"[Stage 1] Filtered {len(enrichment_filter_report.rejected_facts)} "
                "mechanically invalid external facts."
            )
        all_facts: List[RawFact] = extracted_facts + external_facts
        logger.info(f"[Stage 1] Enrichment finished. Total facts now: {len(all_facts)}")
    elif not enrichment_enabled:
        logger.info("[Stage 1] Context enrichment disabled (ablation).")
        all_facts = extracted_facts
    else:
        logger.info(
            "[Stage 1] Coverage gate CLOSED: spec sufficiently complete; skipping enrichment."
        )
        all_facts = extracted_facts

    # Zero-fact guard: no facts means no work for tagger, chunker, or downstream.
    # Emitting a live LLM call on empty input would waste budget and the
    # resulting empty chunk would give Stage 2 a hallucination opportunity.
    if not all_facts:
        logger.error(
            "[Stage 1] No facts were extracted from the NL description; "
            "returning empty plan."
        )
        empty_plan = ChunkedPlan(core_modeling_facts=[], chunks=[])
        output = Output(
            final_facts=[],
            domain="Unknown",
            analytical_goal="General Purpose",
            iterations=[EnrichedNL(extracted_output=extraction_output)],
            original_nl=nl_description,
            plan=empty_plan,
            converged=result.converged,
            token_usage=total_tokens,
        )
        return output, total_tokens

    # Carry each fact's source segment (text + offsets) onto its AtomicFact so the graph
    # chunker can group by segment. External/enrichment facts have no segment and are
    # absent from the lookup (treated as standalone downstream).
    #
    # Built BEFORE tagging on purpose: the tagger documents an `origin` field, but
    # `all_facts` are still RawFact at this point (convert_to_atomic runs below), so
    # reading origin off the fact itself always yielded "(none)". The segment owns
    # that text, so the tagger is handed the lookup explicitly.
    #
    # Segment lookup keys are built with duplicate detection: when the LLM assigns
    # the same ID to facts in different segments, the later entry overwrites the
    # earlier, silently losing segment attribution. A deterministic renumbering
    # pass (below) runs before convert_to_atomic so IDs are globally unique.
    segment_lookup: dict[int, tuple[str, int, int]] = {}
    segment_dup_count = 0
    for s in extraction_output.segments:
        for f in s.facts:
            if f.id in segment_lookup:
                segment_dup_count += 1
            segment_lookup[f.id] = (s.text, s.start_char, s.end_char)
    if segment_dup_count:
        logger.warning(
            "[Stage 1] %d fact ID(s) appeared in multiple segments; "
            "renumbering will deduplicate them.",
            segment_dup_count,
        )

    # Deterministic renumbering: LLMs can emit duplicate fact IDs within or
    # across segments, and external facts from enrichment may collide with
    # extracted facts. A duplicate ID silently overwrites in segment_lookup
    # and the tagger matches on the first collision, so an external fact
    # inherits the wrong segment attribution and tag set. Renumber every fact
    # to a globally-unique ID so membership is never ambiguous.
    seen_ids: set[int] = set()
    for f in all_facts:
        if f.id in seen_ids:
            new_id = max(seen_ids) + 1 if seen_ids else 1
            seen_ids.add(new_id)
            # Update segment_lookup if this fact had a segment entry
            # (enrichment facts won't; they are absent from segment_lookup).
            if f.id in segment_lookup:
                old_entry = segment_lookup.pop(f.id)
                segment_lookup[new_id] = old_entry
            f.id = new_id
        else:
            seen_ids.add(f.id)

    logger.info(f"[Stage 1] Tagging {len(all_facts)} facts...")

    tag_results, t_tag = await tag_facts(
        facts=all_facts,
        model=model,
        provider=provider,
        origins={fid: text for fid, (text, _, _) in segment_lookup.items()},
    )
    total_tokens += t_tag

    tagged_facts = normalize_stage1_tags(
        convert_to_atomic(all_facts, tag_results, segment_lookup)
    )

    # Chunks exist so each one FITS an ER-extraction prompt, so the boundary
    # condition is the token budget, not semantic similarity. See
    # budget_chunker.py for why the Dirichlet-process sampler it replaced was
    # degenerate (it returned 1 chunk for every input measured, at 12,000
    # sweeps a time) and why 1 chunk was nonetheless the right answer.
    if ablation_config is not None and ablation_config.use_bayesian_chunker:
        from src.pipeline.stage1.middleware.bayesian_chunker import BayesianChunker

        logger.info("[Stage 1] Clustering facts via Bayesian partition sampler...")
        plan = BayesianChunker(alpha=0.5, n_burnin=2000, n_samples=2000, thin=5).fit(
            tagged_facts
        )
    else:
        from src.pipeline.stage1.middleware.budget_chunker import BudgetChunker

        # An explicit budget remains an ABLATION knob, not tuning -- but the
        # claim that used to stand here, that a faithful run "always takes the
        # single-chunk path so Stage 2's shard-and-merge is never entered", is no
        # longer true and was the bug rather than a property of the domain. The
        # context-window budget does exceed real fact volume by ~200x, which is
        # exactly why chunking by it never split anything; BudgetChunker now also
        # applies an extraction-CAPACITY ceiling (how much one ER call can model,
        # a far smaller number), so multi-chunk is the normal path for a
        # large-schema spec and forced_multi_chunk() is no longer the only way in.
        budget_override = (
            ablation_config.chunk_budget_tokens if ablation_config is not None else None
        )
        if budget_override is not None:
            logger.info(
                "[Stage 1] Chunking facts to an OVERRIDDEN budget of %d token(s) "
                "-- this is an ablation setting, not the model's real budget.",
                budget_override,
            )
        else:
            logger.info(
                "[Stage 1] Chunking facts to the smaller of the model's context "
                "budget and its per-call extraction capacity..."
            )
        plan = BudgetChunker(budget_tokens=budget_override, model=model).fit(
            tagged_facts
        )
    logger.info(f"[Stage 1] Chunker produced {len(plan.chunks)} chunk(s).")

    output = Output(
        final_facts=tagged_facts,
        domain=extraction_output.domain or "Unknown",
        analytical_goal=extraction_output.analytical_goal or "General Purpose",
        iterations=[EnrichedNL(extracted_output=extraction_output)],
        original_nl=nl_description,
        enrichment_filter_report=enrichment_filter_report,
        context_audit_trail=context_audit_trail,
        plan=plan,
        converged=result.converged,
        token_usage=total_tokens,
    )

    return output, total_tokens


async def _run_context_enrichment_loop(
    facts: List[RawFact],
    gaps: List[SpecGap],
    model: Optional[str],
    audit_trail: List[ContextAuditAttempt],
    provider: Optional[AgentProvider] = None,
    evidence_store: Optional[EvidenceStore] = None,
) -> Tuple[List[RawFact], int]:
    config, enricher_agent, auditor_agent, _filter_agent = make_enrichment_loop_config(
        original_facts=facts,
        gaps=gaps,
        model=model,
        provider=provider,
        evidence_store=evidence_store,
    )
    result = await AgentLoop(config).run("")
    audit_trail.extend(auditor_agent.audit_trail)

    # When the loop ends with is_acceptable=True, the enricher's build_context() is
    # never called for a subsequent round, so accumulated_accepted is never updated
    # with the final accepted set. Merge it here from the terminal node outputs.
    final_auditor = result.node_outputs.get("auditor")
    final_enricher = result.node_outputs.get("enricher")
    if (
        final_auditor is not None
        and getattr(final_auditor, "is_acceptable", False)
        and final_enricher is not None
        and hasattr(final_enricher, "facts")
    ):
        rejected_ids = {
            rf.fact_id for rf in getattr(final_auditor, "rejected_facts", [])
        }
        accepted_ids = set(getattr(final_auditor, "accepted_fact_ids", None) or [])
        if not accepted_ids:
            # Auditor said acceptable but didn't list IDs -- accept all non-rejected.
            accepted_ids = {f.id for f in final_enricher.facts} - rejected_ids  # type: ignore[union-attr]
        existing_ids = {f.id for f in enricher_agent.accumulated_accepted}
        for fact in final_enricher.facts:  # type: ignore[union-attr]
            if fact.id in accepted_ids and fact.id not in existing_ids:
                enricher_agent.accumulated_accepted.append(fact)

    return enricher_agent.accumulated_accepted, result.total_tokens

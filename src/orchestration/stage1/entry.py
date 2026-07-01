from typing import List, Optional, Tuple

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
from src.pipeline.stage1.agents.spec_completeness_auditor.agent import audit_completeness
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
from src.util.observability.llm_trace import (
    LLMTraceCollector,
    activate_trace_collector,
    reset_trace_collector,
)
from src.util.core.search_tool import clear_search_cache
from src.util.orchestration.loop import AgentLoop


NL_MAX_CHARS = 4000


async def orchestrate(
    nl_description: str,
    max_retries: int = 5,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    trace_collector: Optional[LLMTraceCollector] = None,
) -> Tuple[Output, int]:
    if len(nl_description) > NL_MAX_CHARS:
        raise ValueError(
            f"NL description is {len(nl_description)} characters "
            f"(limit: {NL_MAX_CHARS}). Trim the input before running."
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
        )
    finally:
        if trace_token is not None:
            reset_trace_collector(trace_token)


async def _orchestrate_impl(
    nl_description: str,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
) -> Tuple[Output, int]:
    clear_search_cache()
    loop_config = make_stage1_loop_config(nl_description, model=model)
    result = await AgentLoop(loop_config).run(nl_description)

    raw_extraction = result.node_outputs.get("extractor")
    extraction_output: RephrasedOutput = (
        raw_extraction
        if isinstance(raw_extraction, RephrasedOutput)
        else RephrasedOutput(segments=[])
    )
    if not isinstance(raw_extraction, RephrasedOutput):
        print(
            f"[Stage 1] Loop exhausted after {result.iteration_count} iterations "
            f"with no accepted extractor output."
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
    )
    total_tokens += t_cov

    gate_open = coverage_report.is_underspecified
    enrichment_enabled = ablation_config is None or ablation_config.enable_enrichment

    if enrichment_enabled and gate_open:
        gate_gaps = coverage_report.gaps_for_enrichment()
        print(f"[Stage 1] Coverage gate OPEN: {len(gate_gaps)} gaps (minor gaps included for free).")
        external_facts, t_enrich = await _run_context_enrichment_loop(
            facts=extracted_facts,
            gaps=gate_gaps,
            model=model,
            audit_trail=context_audit_trail,
        )
        total_tokens += t_enrich
        enrichment_filter_report = filter_external_facts(
            external_facts, extracted_facts
        )
        external_facts = enrichment_filter_report.accepted_facts
        if enrichment_filter_report.rejected_facts:
            print(
                f"[Stage 1] Filtered {len(enrichment_filter_report.rejected_facts)} "
                "mechanically invalid external facts."
            )
        all_facts: List[RawFact] = extracted_facts + external_facts
    elif not enrichment_enabled:
        print("[Stage 1] Context enrichment disabled (ablation).")
        all_facts = extracted_facts
    else:
        print("[Stage 1] Coverage gate CLOSED: spec sufficiently complete; skipping enrichment.")
        all_facts = extracted_facts

    tag_results, t_tag = await tag_facts(facts=all_facts, model=model)
    total_tokens += t_tag

    # Carry each fact's source segment (text + offsets) onto its AtomicFact so the graph
    # chunker can group by segment. External/enrichment facts have no segment and are
    # absent from the lookup (treated as standalone downstream).
    segment_lookup = {
        f.id: (s.text, s.start_char, s.end_char)
        for s in extraction_output.segments
        for f in s.facts
    }
    tagged_facts = normalize_stage1_tags(
        convert_to_atomic(all_facts, tag_results, segment_lookup)
    )

    from src.util.schema_ops.graph_chunker import run_graph_chunker

    print("[Stage 1] Clustering facts...")
    plan = run_graph_chunker(tagged_facts)
    # the graph chunker is deterministic and zero tokens

    output = Output(
        final_facts=tagged_facts,
        domain=extraction_output.domain or "Unknown",
        analytical_goal=extraction_output.analytical_goal or "General Purpose",
        iterations=[EnrichedNL(extracted_output=extraction_output)],
        original_nl=nl_description,
        enrichment_filter_report=enrichment_filter_report,
        context_audit_trail=context_audit_trail,
        plan=plan,
        token_usage=total_tokens,
    )

    return output, total_tokens


async def _run_context_enrichment_loop(
    facts: List[RawFact],
    gaps: List[SpecGap],
    model: Optional[str],
    audit_trail: List[ContextAuditAttempt],
) -> Tuple[List[RawFact], int]:
    config, enricher_agent, auditor_agent = make_enrichment_loop_config(
        original_facts=facts,
        gaps=gaps,
        model=model,
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

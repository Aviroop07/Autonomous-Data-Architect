from typing import Optional, Tuple, List
from src.orchestration.stage1.models import Output
from src.pipeline.stage1.agents.fact_extractor.agent import get_agent as get_extractor
from src.pipeline.stage1.agents.verifier.agent import verify_integrity
from src.pipeline.stage1.agents.tagger.agent import tag_facts
from src.pipeline.stage1.agents.context_enricher.agent import enrich_context
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput, EnrichedNL, convert_to_atomic
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag
from src.pipeline.stage1.middleware.validation import deterministic_validator
from src.pipeline.stage1.middleware.error_formatter import format_errors_for_stage1
from src.util.retry_loop import RetryLoop, RetryConfig
from src.util.ablation import AblationConfig

async def orchestrate(
    nl_description: str,
    max_retries: int = 5,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None
) -> Tuple[Output, int]:
    config = RetryConfig(max_retries=max_retries)

    async def llm_validator(output, ctx):
        result, _ = await verify_integrity(ctx, output.facts, model=model)
        return result

    error_formatter = lambda errors, iteration, facts: format_errors_for_stage1(
        errors, iteration, facts, nl_description
    )

    loop = RetryLoop(
        agent_getter=lambda: get_extractor(model),
        output_structure=RephrasedOutput,
        llm_validator=llm_validator,  # type: ignore[arg-type]
        error_formatter=error_formatter,
        config=config,
        deterministic_validator=lambda out, ctx: deterministic_validator(out, ctx)
    )

    extraction_output, total_tokens, error_history = await loop.run(
        task="Extract atomic facts from the description",
        context=nl_description
    )

    extracted_facts: List[RawFact] = extraction_output.facts

    if ablation_config is None or ablation_config.enable_enrichment:
        external_facts, t_enrich = await enrich_context(facts=extracted_facts, model=model)
        total_tokens += t_enrich
        all_facts: List[RawFact] = extracted_facts + external_facts
    else:
        print("[Stage 1] Context enrichment disabled (ablation).")
        all_facts = extracted_facts

    tag_results, t_tag = await tag_facts(facts=all_facts, model=model)
    total_tokens += t_tag

    tagged_facts = convert_to_atomic(all_facts, tag_results)

    for fact in tagged_facts:
        if hasattr(fact, 'is_external') and fact.is_external:
            if FactTag.METADATA not in fact.tags:
                fact.tags.append(FactTag.METADATA)

    output = Output(
        final_facts=tagged_facts,
        domain=extraction_output.domain,
        analytical_goal=extraction_output.analytical_goal,
        iterations=[EnrichedNL(extracted_output=extraction_output)],
        original_nl=nl_description,
        token_usage=total_tokens
    )

    return output, total_tokens

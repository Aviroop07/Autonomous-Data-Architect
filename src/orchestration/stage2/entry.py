import asyncio
from typing import Optional, Tuple, List
from src.orchestration.stage2.models import Output
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.orchestration.stage2.utils import (
    run_conceptual_extractor_loop,
)
from src.util.schema_ops.graph_chunker import run_graph_chunker

from src.pipeline.stage2.agents.compliance_certifier.agent import certify_compliance
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.pipeline.stage2.mapper.relational_mapper import map_conceptual_to_relational
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.util.schema_ops.schema_patch import CritiqueReport
from src.util.schema_ops.patching_engine import apply_patches
from src.util.config.ablation import AblationConfig

_REQUIRED_FACT_TAGS = {FactTag.STRUCTURAL, FactTag.LOGICAL, FactTag.STATISTICAL}


def _compute_uncovered_facts(
    facts: List[AtomicFact],
    final_schema: Schema,
    registry: TableFactRegistry,
) -> List[int]:
    required_ids = {
        f.id for f in facts if any(tag in f.tags for tag in _REQUIRED_FACT_TAGS)
    }
    covered: set = set()
    for table in final_schema.tables:
        covered.update(registry.get_facts_for_tables([table.name]))
    uncovered = sorted(required_ids - covered)
    if uncovered:
        print(
            f"  [Stage 2] WARNING: {len(uncovered)} required facts not represented "
            f"in final schema: {uncovered}"
        )
    return uncovered


async def orchestrate(
    plan: ChunkedPlan,
    facts: List[AtomicFact],
    domain: str = "Unknown",
    analytical_goal: str = "General Schema Design",
    retry_count: int = 5,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    enable_audit: bool = True,
    nl_query: str = "",
) -> Tuple[Output, int, TableFactRegistry]:
    total_tokens = 0
    registry = TableFactRegistry()

    if ablation_config is not None and not ablation_config.enable_sharding:
        print("[Stage 2] Sharding disabled — treating all facts as a single chunk.")
        plan = ChunkedPlan(core_modeling_facts=facts, chunks=[facts])

    print(
        f"[Stage 2] Generating {len(plan.chunks)} conceptual model shards in parallel..."
    )
    tasks = [
        run_conceptual_extractor_loop(
            facts=cluster, nl_query=nl_query, max_retries=retry_count, model=model
        )
        for cluster in plan.chunks
    ]
    results = await asyncio.gather(*tasks)

    conceptual_models: List[ConceptualModel] = []
    initial_fix_histories = []

    for i, (cm, fix_hist, t_gen) in enumerate(results):
        total_tokens += t_gen
        conceptual_models.append(cm)
        initial_fix_histories.append(fix_hist)

    print(
        "[Stage 2] Merging conceptual shards via Gale-Shapley deterministic merger..."
    )
    from src.pipeline.stage2.middleware.conceptual_merger import merge_shards

    if not conceptual_models:
        raise ValueError("No conceptual models generated.")

    combined_cm = conceptual_models[0]
    for cm in conceptual_models[1:]:
        combined_cm = await merge_shards(
            combined_cm, cm, domain, analytical_goal, facts, model=model
        )

    print("[Stage 2] Deterministic mapping of conceptual model to relational schema...")
    global_schema = map_conceptual_to_relational(combined_cm)

    from src.pipeline.stage2.models.schema import to_snake_case
    import re

    for table in global_schema.tables:
        t_name_snake = to_snake_case(table.name).lower()
        matched_entity = next(
            (
                e
                for e in combined_cm.entities
                if to_snake_case(e.name).lower() == t_name_snake
            ),
            None,
        )

        matched_rel = None
        for r in combined_cm.relationships:
            r_snake = to_snake_case(r.name).lower()
            if r_snake == t_name_snake:
                matched_rel = r
                break

            parts = sorted(
                list(set(to_snake_case(p.entity).lower() for p in r.participants))
            )
            parts_str = "_".join(parts)

            if t_name_snake == parts_str:
                matched_rel = r
                break

            if t_name_snake == f"{parts_str}_{r_snake}":
                matched_rel = r
                break

            if re.match(rf"^{parts_str}(_{r_snake})?(_\d+)?$", t_name_snake):
                matched_rel = r
                break

        # FK-structural fallback: match junction tables by their FK targets
        if not matched_rel and global_schema.relationships:
            table_fk_targets = sorted(
                {
                    to_snake_case(fk.referred_table).lower()
                    for fk in global_schema.relationships
                    if fk.referencing_table == table.name
                }
            )
            if len(table_fk_targets) >= 2:
                for r in combined_cm.relationships:
                    r_parts = sorted(
                        list(
                            set(to_snake_case(p.entity).lower() for p in r.participants)
                        )
                    )
                    if table_fk_targets == r_parts:
                        matched_rel = r
                        break

        fact_ids = set()
        if matched_entity:
            fact_ids.update(matched_entity.source_fact_ids)
        if matched_rel:
            fact_ids.update(matched_rel.source_fact_ids)

        if not fact_ids:
            for f in facts:
                if table.name.lower() in f.fact.lower():
                    fact_ids.add(f.id)

        registry.register_table_facts(table.name, list(fact_ids))

    print(
        f"  [Merge] Schema generation complete: {len(global_schema.tables)} tables, {len(global_schema.relationships or [])} FKs"
    )
    for r in global_schema.relationships or []:
        print(
            f"  [Merge] FK: {r.referencing_table}.{r.referencing_column} -> {r.referred_table}"
        )

    final_global_schema = global_schema
    cert_report = CritiqueReport(agent_name="certifier_skipped", patches=[])

    if enable_audit:
        print("[Stage 2] Final compliance certification...")
        try:
            cert_report, t_cert = await certify_compliance(
                schema=final_global_schema,
                goal=analytical_goal,
                enriched_nl=facts,
                model=model,
            )
            total_tokens += t_cert
        except Exception as cert_exc:
            print(
                f"  [Stage 2] Certifier failed ({cert_exc}); skipping certification patches."
            )
            cert_report = CritiqueReport(agent_name="certifier_skipped", patches=[])

        if cert_report.patches:
            apply_patches(final_global_schema, cert_report.patches, registry=registry)

    uncovered = _compute_uncovered_facts(facts, final_global_schema, registry)

    # Per-shard relational maps are DIAGNOSTIC output only (Output.segments); the real
    # schema is final_global_schema. A repair failure on one shard must never discard the
    # already-computed global result, so guard each shard mapping and skip on failure.
    mock_segments = []
    for cm in conceptual_models:
        try:
            mock_segments.append(map_conceptual_to_relational(cm))
        except Exception as seg_exc:
            print(
                f"  [Stage 2] Skipping diagnostic segment map (shard failed: {seg_exc})."
            )

    return (
        Output(
            segments=mock_segments,
            plan=plan,
            fix_history=initial_fix_histories,
            merged_schema=global_schema,
            final_global_schema=final_global_schema,
            final_fix_history=[],
            domain_iterations=[],
            token_usage=total_tokens,
            cycles=final_global_schema.detect_cycles(),
            uncovered_fact_ids=uncovered,
            merge_decision_log=None,
            cert_report=cert_report,
        ),
        total_tokens,
        registry,
    )

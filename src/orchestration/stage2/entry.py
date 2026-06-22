import asyncio
from typing import Optional, Tuple, List, Dict
from src.orchestration.stage2.models import Output, RefinementIteration
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.orchestration.stage2.utils import (
    run_architect_self_correction_loop,
    run_auditor_self_correction_loop,
    run_chunker_with_retry,
)

from src.pipeline.stage2.agents.domain_intelligence_extractor.agent import (
    run_domain_intelligence,
)
from src.pipeline.stage2.agents.compliance_certifier.agent import certify_compliance
from src.pipeline.stage2.agents.merge_reviewer.agent import run_merge_review

from src.pipeline.stage2.middleware.schema_merging.merger import SchemaMerger
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.orchestration.stage2.refinement_sharding import (
    get_deterministic_shards,
    absorb_isolated_singletons,
)
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
    facts: List[AtomicFact],
    domain: str = "Unknown",
    analytical_goal: str = "General Schema Design",
    retry_count: int = 5,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    enable_audit: bool = True,
) -> Tuple[Output, int, TableFactRegistry]:
    """
    Orchestrates Stage 2:
    1. Chunker -> Architects (Iterative)
    2. Deterministic Merge (Gale-Shapley + Rule-based FKs)
    3. (Optional) Deterministic Sharding
    4. (Optional) Parallel Auditor (Iterative)
    5. (Optional) Final Deterministic Merge
    6. (Optional) Global Validation & Error-Anchored Repair
    """
    total_tokens = 0
    registry = TableFactRegistry()

    # 1. Domain Intelligence (Passive)
    print(f"[Stage 2] Researching '{domain}'...")
    intelligence, t_res = await run_domain_intelligence(domain, model=model)
    total_tokens += t_res

    # 2. Initial Chunking (skipped when sharding is disabled)
    if ablation_config is not None and not ablation_config.enable_sharding:
        print("[Stage 2] Sharding disabled — treating all facts as a single chunk.")
        plan = ChunkedPlan(core_modeling_facts=facts, chunks=[facts])
        t_plan = 0
    else:
        print("[Stage 2] Clustering facts...")
        plan, t_plan = await run_chunker_with_retry(
            facts=facts,
            max_retries=retry_count,
            model=model,
            domain=domain,
            analytical_goal=analytical_goal,
        )
    total_tokens += t_plan

    # 3. Parallel Shard Generation (Iterative Architect)
    print(f"[Stage 2] Generating {len(plan.chunks)} shards in parallel...")

    # Using asyncio.gather for parallel shard generation
    tasks = [
        run_architect_self_correction_loop(
            facts=cluster, max_retries=retry_count, model=model
        )
        for cluster in plan.chunks
    ]
    results = await asyncio.gather(*tasks)

    shards: List[Schema] = []
    initial_fix_histories: List[List[FixHistoryStep]] = []

    for i, (shard, fix_hist, t_gen) in enumerate(results):
        total_tokens += t_gen
        shards.append(shard)
        initial_fix_histories.append(fix_hist)

        print(f"  [Shard {i + 1}] Tables: {[t.name for t in shard.tables]}")
        print(
            f"  [Shard {i + 1}] FKs:    {[(r.referencing_table + '.' + r.referencing_column + '->' + r.referred_table) for r in (shard.relationships or [])]}"
        )

        # Registry Update
        chunk_fact_ids = [f.id for f in plan.chunks[i]]
        for table in shard.tables:
            registry.register_table_facts(table.name, chunk_fact_ids)

    # 4. Initial Global Merge (Deterministic)
    print("[Stage 2] Initial deterministic merge...")
    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    global_schema, merge_decision_log = merger.merge_segments(shards, registry=registry)

    print(
        f"  [Merge] Post-initial-merge: {len(global_schema.tables)} tables, "
        f"{len(global_schema.relationships or [])} FKs"
    )
    print(f"  [Merge] Tables: {[t.name for t in global_schema.tables]}")
    for r in global_schema.relationships or []:
        print(
            f"  [Merge] FK: {r.referencing_table}.{r.referencing_column} -> {r.referred_table}"
        )

    fact_map = {f.id: f for f in facts}

    # 4b. Merge Review Agent (gated: only on multi-shard with reviewable decisions)
    if (
        enable_audit
        and len(shards) > 1
        and (merge_decision_log.matched_pairs or merge_decision_log.unmatched_tables)
    ):
        print("[Stage 2] Running merge review agent...")
        shard_fact_pairs = [(i, plan.chunks[i]) for i in range(len(plan.chunks))]
        review_report, t_review = await run_merge_review(
            merged_schema=global_schema,
            decision_log=merge_decision_log,
            shard_facts=shard_fact_pairs,
            shard_schemas=shards,
            model=model,
        )
        total_tokens += t_review
        if review_report.patches:
            print(
                f"  [Stage 2] Applying {len(review_report.patches)} merge review patches..."
            )
            apply_patches(global_schema, review_report.patches, registry=registry)
            review_errors = global_schema._validate()
            if review_errors:
                print(
                    f"  [Stage 2] WARNING: {len(review_errors)} validation errors after MRA patches. "
                    f"Auditors will correct in the next step."
                )
                for err in review_errors[:5]:
                    print(f"    - {err}")
        else:
            print("  [Stage 2] Merge review: no patches needed.")
    else:
        print(
            "[Stage 2] Merge review skipped (single shard or no reviewable decisions)."
        )

    if not enable_audit:
        final_global_schema = global_schema
        final_fixes: List[FixHistoryStep] = []
        audit_iterations: List[RefinementIteration] = []
        print("[Stage 2] Final global structural validation (audit disabled)...")
        global_errors = final_global_schema._validate()
        if global_errors:
            print(
                f"  [Stage 2] Found {len(global_errors)} global errors. Skipping critique/repair."
            )
            for err in global_errors[:10]:
                print(f"    - {err}")
            if len(global_errors) > 10:
                print("    - ...")

        uncovered = _compute_uncovered_facts(facts, final_global_schema, registry)
        return (
            Output(
                segments=shards,
                plan=plan,
                fix_history=initial_fix_histories,
                merged_schema=global_schema,
                final_global_schema=final_global_schema,
                final_fix_history=final_fixes,
                domain_iterations=audit_iterations,
                token_usage=total_tokens,
                cycles=final_global_schema.detect_cycles(),
                uncovered_fact_ids=uncovered,
                merge_decision_log=merge_decision_log,
            ),
            total_tokens,
            registry,
        )

    # 5. Deterministic Sharding & Re-allocation
    print("[Stage 2] Deterministic schema sharding...")
    det_shards_table_names = get_deterministic_shards(global_schema)
    det_shards_table_names = absorb_isolated_singletons(
        det_shards_table_names, global_schema
    )

    # Map back to Schema Shards and Fact Clusters
    det_shard_schemas: List[Schema] = []
    det_shard_facts: List[Dict[str, List[AtomicFact]]] = []

    for si, table_names in enumerate(det_shards_table_names):
        print(f"  [DetShard {si + 1}] Tables: {sorted(table_names)}")
        # Create a schema segment for these tables
        segment = Schema(
            tables=[t for t in global_schema.tables if t.name in table_names],
            relationships=[
                r
                for r in (global_schema.relationships or [])
                if r.referencing_table in table_names
                and r.referred_table in table_names
            ],
        )
        det_shard_schemas.append(segment)

        # Get facts for these tables from registry
        cluster = {}
        for t_name in table_names:
            f_ids = registry.get_facts_for_tables([t_name])
            cluster[t_name] = [fact_map[fid] for fid in f_ids if fid in fact_map]
        det_shard_facts.append(cluster)

    # 6. Iterative Parallel Audit
    print(
        f"[Stage 2] Iterative auditing of {len(det_shard_schemas)} shards in parallel..."
    )
    audit_tasks = [
        run_auditor_self_correction_loop(
            shard_schema=shard,
            intelligence=intelligence,
            fact_clusters=fact_cluster,
            max_retries=retry_count,
            registry=registry,
            model=model,
        )
        for shard, fact_cluster in zip(det_shard_schemas, det_shard_facts)
    ]
    audit_results = await asyncio.gather(*audit_tasks)

    audited_shards: List[Schema] = []
    audit_iterations: List[RefinementIteration] = []

    for ai, (audited_shard, fix_hist, t_audit) in enumerate(audit_results):
        total_tokens += t_audit
        audited_shards.append(audited_shard)
        print(
            f"  [PostAudit {ai + 1}] Tables: {[t.name for t in audited_shard.tables]}, "
            f"{len(audited_shard.relationships or [])} FKs"
        )
        for r in audited_shard.relationships or []:
            print(
                f"    FK: {r.referencing_table}.{r.referencing_column} -> {r.referred_table}"
            )

        # Log iterations
        if fix_hist:
            audit_iterations.append(
                RefinementIteration(
                    iteration=len(fix_hist),
                    critique=CritiqueReport(agent_name="auditor", patches=[]),
                    fix_history=fix_hist,
                    schema_state=audited_shard,
                )
            )

    # 7. Final Deterministic Merge (strict: name-exact only — canonical names are stable post-audit)
    print("[Stage 2] Final deterministic merge (strict)...")
    final_global_schema, _ = merger.merge_segments(
        audited_shards, registry=registry, strict=True
    )

    # 8. Global Validation & Error-Anchored Repair
    print("[Stage 2] Final global structural validation...")
    global_errors = final_global_schema._validate()
    final_fixes = []

    if global_errors:
        print(
            f"  [Stage 2] Found {len(global_errors)} global errors. Attempting anchored repair..."
        )
        # Error-anchored sharding: identify tables in error and re-audit
        (
            final_global_schema,
            final_fixes,
            t_global_fix,
        ) = await run_auditor_self_correction_loop(
            shard_schema=final_global_schema,
            intelligence=intelligence,
            fact_clusters={
                t.name: [
                    fact_map[fid]
                    for fid in registry.get_facts_for_tables([t.name])
                    if fid in fact_map
                ]
                for t in final_global_schema.tables
            },
            max_retries=5,
            registry=registry,
            model=model,
            initial_errors=global_errors,
        )
        total_tokens += t_global_fix

    # 9. Final Certification (Agentic)
    print("[Stage 2] Final compliance certification...")
    cert_report, t_cert = await certify_compliance(
        schema=final_global_schema, goal=analytical_goal, enriched_nl=facts, model=model
    )
    total_tokens += t_cert

    if cert_report.patches:
        apply_patches(final_global_schema, cert_report.patches, registry=registry)
        cert_errors = final_global_schema._validate()
        if cert_errors:
            print(
                f"  [Stage 2] Certifier patches introduced {len(cert_errors)} errors. "
                f"Repairing..."
            )
            (
                final_global_schema,
                _cert_repair_history,
                t_cert_repair,
            ) = await run_auditor_self_correction_loop(
                shard_schema=final_global_schema,
                intelligence=intelligence,
                fact_clusters={
                    t.name: [
                        fact_map[fid]
                        for fid in registry.get_facts_for_tables([t.name])
                        if fid in fact_map
                    ]
                    for t in final_global_schema.tables
                },
                max_retries=3,
                registry=registry,
                model=model,
                initial_errors=cert_errors,
            )
            total_tokens += t_cert_repair

    uncovered = _compute_uncovered_facts(facts, final_global_schema, registry)
    return (
        Output(
            segments=shards,
            plan=plan,
            fix_history=initial_fix_histories,
            merged_schema=global_schema,
            final_global_schema=final_global_schema,
            final_fix_history=final_fixes,
            domain_iterations=audit_iterations,
            token_usage=total_tokens,
            cycles=final_global_schema.detect_cycles(),
            uncovered_fact_ids=uncovered,
            merge_decision_log=merge_decision_log,
            cert_report=cert_report,
        ),
        total_tokens,
        registry,
    )

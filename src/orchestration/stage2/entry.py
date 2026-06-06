import asyncio
from typing import Optional, Tuple, List, Dict
from src.orchestration.stage2.models import Output, RefinementIteration
from src.pipeline.stage2.agents.schema_architect.agent import get_agent as get_architect
from src.pipeline.stage2.agents.chunker.agent import run_chunker
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.orchestration.stage2.utils import run_architect_self_correction_loop, run_auditor_self_correction_loop

from src.pipeline.stage2.agents.domain_intelligence_extractor.agent import run_domain_intelligence
from src.pipeline.stage2.agents.domain_auditor.agent import get_agent as get_domain_auditor
from src.pipeline.stage2.agents.compliance_certifier.agent import certify_compliance

from src.pipeline.stage2.middleware.schema_merging.merger import SchemaMerger
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.orchestration.stage2.refinement_sharding import get_deterministic_shards
from src.util.schema_patch import CritiqueReport
from src.util.ablation import AblationConfig

async def orchestrate(
    facts: List[AtomicFact],
    domain: str = "Unknown",
    analytical_goal: str = "General Schema Design",
    retry_count: int = 5,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None
) -> Tuple[Output, int, TableFactRegistry]:
    """
    Orchestrates Stage 2:
    1. Chunker -> Architects (Iterative)
    2. Deterministic Merge (Gale-Shapley + Rule-based FKs)
    3. Deterministic Sharding
    4. Parallel Auditor (Iterative)
    5. Final Deterministic Merge
    6. Global Validation & Error-Anchored Repair
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
        plan, t_plan = await run_chunker(facts, model=model)
    total_tokens += t_plan

    # 3. Parallel Shard Generation (Iterative Architect)
    print(f"[Stage 2] Generating {len(plan.chunks)} shards in parallel...")
    architect = get_architect(model)

    # Using asyncio.gather for parallel shard generation
    tasks = [
        run_architect_self_correction_loop(
            facts=cluster,
            max_retries=retry_count,
            architect=architect,
            model=model
        ) for cluster in plan.chunks
    ]
    results = await asyncio.gather(*tasks)

    shards: List[Schema] = []
    initial_fix_histories: List[List[FixHistoryStep]] = []

    for i, (shard, fix_hist, t_gen) in enumerate(results):
        total_tokens += t_gen
        shards.append(shard)
        initial_fix_histories.append(fix_hist)

        # Registry Update
        chunk_fact_ids = [f.id for f in plan.chunks[i]]
        for table in shard.tables:
            registry.register_table_facts(table.name, chunk_fact_ids)

    # 4. Initial Global Merge (Deterministic)
    print(f"[Stage 2] Initial deterministic merge...")
    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    global_schema = merger.merge_segments(shards, registry=registry)

    # 5. Deterministic Sharding & Re-allocation
    print("[Stage 2] Deterministic schema sharding...")
    det_shards_table_names = get_deterministic_shards(global_schema)

    # Map back to Schema Shards and Fact Clusters
    det_shard_schemas: List[Schema] = []
    det_shard_facts: List[Dict[str, List[AtomicFact]]] = []

    fact_map = {f.id: f for f in facts}

    for table_names in det_shards_table_names:
        # Create a schema segment for these tables
        segment = Schema(
            tables=[t for t in global_schema.tables if t.name in table_names],
            relationships=[r for r in (global_schema.relationships or [])
                          if r.referencing_table in table_names and r.referred_table in table_names]
        )
        det_shard_schemas.append(segment)

        # Get facts for these tables from registry
        cluster = {}
        for t_name in table_names:
            f_ids = registry.get_facts_for_tables([t_name])
            cluster[t_name] = [fact_map[fid] for fid in f_ids if fid in fact_map]
        det_shard_facts.append(cluster)

    # 6. Iterative Parallel Audit
    print(f"[Stage 2] Iterative auditing of {len(det_shard_schemas)} shards in parallel...")
    auditor = get_domain_auditor(model)

    audit_tasks = [
        run_auditor_self_correction_loop(
            shard_schema=shard,
            intelligence=intelligence,
            fact_clusters=fact_cluster,
            max_retries=retry_count,
            registry=registry,
            auditor=auditor,
            model=model
        ) for shard, fact_cluster in zip(det_shard_schemas, det_shard_facts)
    ]
    audit_results = await asyncio.gather(*audit_tasks)

    audited_shards: List[Schema] = []
    audit_iterations: List[RefinementIteration] = []

    for audited_shard, fix_hist, t_audit in audit_results:
        total_tokens += t_audit
        audited_shards.append(audited_shard)

        # Log iterations
        if fix_hist:
            audit_iterations.append(RefinementIteration(
                iteration=len(fix_hist),
                critique=CritiqueReport(agent_name="auditor", patches=[]), # Placeholder
                fix_history=fix_hist,
                schema_state=audited_shard
            ))

    # 7. Final Deterministic Merge
    print("[Stage 2] Final deterministic merge...")
    final_global_schema = merger.merge_segments(audited_shards, registry=registry)

    # 8. Global Validation & Error-Anchored Repair
    print("[Stage 2] Final global structural validation...")
    global_errors = final_global_schema._validate()
    final_fixes = []

    if global_errors:
        print(f"  [Stage 2] Found {len(global_errors)} global errors. Attempting anchored repair...")
        # Error-anchored sharding: identify tables in error and re-audit
        final_global_schema, final_fixes, t_global_fix = await run_auditor_self_correction_loop(
            shard_schema=final_global_schema,
            intelligence=intelligence,
            fact_clusters={t.name: [fact_map[fid] for fid in registry.get_facts_for_tables([t.name]) if fid in fact_map] for t in final_global_schema.tables},
            max_retries=5,
            registry=registry,
            auditor=auditor,
            model=model
        )
        total_tokens += t_global_fix

    # 9. Final Certification (Agentic)
    print("[Stage 2] Final compliance certification...")
    cert_report, t_cert = await certify_compliance(
        schema=final_global_schema,
        goal=analytical_goal,
        enriched_nl=facts,
        model=model
    )
    total_tokens += t_cert

    return Output(
        segments=shards,
        plan=plan,
        fix_history=initial_fix_histories,
        merged_schema=global_schema,
        final_global_schema=final_global_schema,
        final_fix_history=final_fixes,
        domain_iterations=audit_iterations,
        token_usage=total_tokens,
        cycles=final_global_schema.detect_cycles()
    ), total_tokens, registry

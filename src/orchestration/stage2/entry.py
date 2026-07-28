import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import networkx as nx

from src.util.observability.artifact_dump import dump_artifact
from src.orchestration.stage2.models import Output
from src.util.schema_model.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.orchestration.stage2.utils import (
    run_er_extractor_loop,
)
from src.util.orchestration.parallel_loop import run_parallel

from src.pipeline.stage2.agents.compliance_certifier.agent import certify_compliance
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.pipeline.stage2.mapper.relational_mapper import map_conceptual_to_relational
from src.util.schema_model.registry import TableFactRegistry
from src.util.config.ablation import AblationConfig
from src.pipeline.stage2.models.conflicts import ActionType, ConflictFlag

logger = logging.getLogger(__name__)

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
        # Also count the schema's OWN provenance, which is authoritative: the
        # relational mapper stamps source_fact_ids onto every table, column and
        # foreign key as it creates them. The registry is populated afterwards by
        # matching table names back to conceptual entities/relationships, and
        # that reverse-engineering cannot see tables the MAPPER invented --
        # junction tables for M:N and n-ary relationships have no conceptual
        # entity of their own to match against.
        #
        # The consequence was a metric that scaled with junction count rather
        # than with real coverage. A live run reported "37 of 52 required facts
        # not represented" purely because it produced more M:N relationships than
        # its predecessors, while its actual provenance covered MORE facts (49
        # distinct ids against 46) and every one of its 52 facts named a real
        # table in the final schema.
        covered.update(table.source_fact_ids or [])
        for column in table.columns:
            covered.update(column.source_fact_ids or [])
    for fk in final_schema.relationships or []:
        covered.update(fk.source_fact_ids or [])
    uncovered = sorted(required_ids - covered)
    if uncovered:
        logger.warning(
            f"  [Stage 2] WARNING: {len(uncovered)} required facts not represented "
            f"in final schema: {uncovered}"
        )
    return uncovered


def apply_adjudicator_patches(cm: ConceptualModel, patches: List) -> ConceptualModel:
    """Apply the adjudicator's resolutions to a copy of the conceptual model.

    Every patch is checked against ResolutionAction._validate() before it is
    applied, and an invalid one is skipped with a warning rather than being let
    through. That validator already existed but was never called anywhere in
    orchestration, so an action missing an optional-by-typing but
    required-in-practice field went straight into the model: the common case
    was MERGE_ENTITIES without `new_name`, which assigned None onto
    `entity.name` and surfaced much later as an AttributeError inside the
    relational mapper's to_snake_case, with nothing left to point at the
    adjudicator. Omitting an optional field is routine LLM behaviour, and this
    agent is single-shot with no retry loop, so it has to degrade rather than
    fail.
    """
    import copy

    new_cm = copy.deepcopy(cm)
    for p in patches:
        errors = p._validate()
        if errors:
            logger.warning(
                "  [Stage 2] Skipping malformed adjudicator patch %s: %s",
                getattr(p.action_type, "value", p.action_type),
                "; ".join(errors),
            )
            continue
        if p.action_type == ActionType.MERGE_ENTITIES:
            ea = next((e for e in new_cm.entities if e.name == p.entity_a), None)
            eb = next((e for e in new_cm.entities if e.name == p.entity_b), None)
            if ea and eb:
                ea.name = p.new_name
                existing = {a.name for a in ea.attributes}
                for a in eb.attributes:
                    if a.name not in existing:
                        ea.attributes.append(a)
                        existing.add(a.name)
                ea.source_fact_ids.extend(eb.source_fact_ids)
                ea.source_fact_ids = list(set(ea.source_fact_ids))
                new_cm.entities.remove(eb)
                for r in new_cm.relationships:
                    for part in r.participants:
                        if part.entity in [p.entity_a, p.entity_b]:
                            part.entity = p.new_name
            else:
                logger.warning(
                    f"MERGE_ENTITIES patch: could not find entities '{p.entity_a}' or '{p.entity_b}' in model"
                )
        elif p.action_type == ActionType.RENAME_ATTRIBUTE:
            ea = next((e for e in new_cm.entities if e.name == p.entity_a), None)
            if ea:
                found = False
                for a in ea.attributes:
                    if a.name == p.attribute_old:
                        a.name = p.new_name
                        found = True
                if not found:
                    logger.warning(
                        f"RENAME_ATTRIBUTE patch: attribute '{p.attribute_old}' not found on entity '{p.entity_a}'"
                    )
            else:
                logger.warning(
                    f"RENAME_ATTRIBUTE patch: entity '{p.entity_a}' not found in model"
                )
        elif p.action_type == ActionType.RESOLVE_CARDINALITY:
            ra = next(
                (r for r in new_cm.relationships if r.name == p.relationship_name), None
            )
            if ra:
                ra.kind = p.new_cardinality
            else:
                logger.warning(
                    f"RESOLVE_CARDINALITY patch: relationship '{p.relationship_name}' not found in model"
                )
        elif p.action_type == ActionType.RESOLVE_CROSS_CATEGORY:
            ea = next((e for e in new_cm.entities if e.name == p.entity_a), None)
            ra = next(
                (r for r in new_cm.relationships if r.name == p.relationship_name), None
            )
            if ea and ra:
                new_cm.relationships.remove(ra)
                logger.info(
                    f"RESOLVE_CROSS_CATEGORY: removed relationship '{p.relationship_name}' (merged into entity '{p.entity_a}')"
                )
            elif ea and not ra:
                logger.warning(
                    f"RESOLVE_CROSS_CATEGORY: relationship '{p.relationship_name}' not found for entity '{p.entity_a}'"
                )
            else:
                logger.warning(
                    f"RESOLVE_CROSS_CATEGORY: entity '{p.entity_a}' not found for relationship '{p.relationship_name}'"
                )
        elif p.action_type == ActionType.RESOLVE_IDENTIFIER:
            ea = next((e for e in new_cm.entities if e.name == p.entity_a), None)
            if ea:
                ea.identifier_attributes = p.new_identifier_attributes
                logger.info(
                    f"RESOLVE_IDENTIFIER: entity '{p.entity_a}' identifiers set to {p.new_identifier_attributes}"
                )
            else:
                logger.warning(
                    f"RESOLVE_IDENTIFIER patch: entity '{p.entity_a}' not found in model"
                )
    return new_cm


def _build_conflict_graph(
    flags: List[ConflictFlag],
) -> Tuple[nx.Graph, Dict[str, List[ConflictFlag]]]:
    G = nx.Graph()
    flag_map: Dict[str, List[ConflictFlag]] = {}
    for flag in flags:
        for e_name in flag.entities:
            G.add_node(e_name)
            if e_name not in flag_map:
                flag_map[e_name] = []
            flag_map[e_name].append(flag)
        for i in range(len(flag.entities)):
            for j in range(i + 1, len(flag.entities)):
                G.add_edge(flag.entities[i], flag.entities[j])
    return G, flag_map


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
    artifact_dir: Optional[Path] = None,
) -> Tuple[Output, int, TableFactRegistry]:
    total_tokens = 0
    registry = TableFactRegistry()

    if ablation_config is not None and not ablation_config.enable_sharding:
        logger.info(
            "[Stage 2] Sharding disabled. Treating all facts as a single chunk."
        )
        plan = ChunkedPlan(core_modeling_facts=facts, chunks=[facts])

    logger.info(
        f"[Stage 2] Generating {len(plan.chunks)} conceptual model shards in parallel..."
    )
    tasks = [
        run_er_extractor_loop(
            facts=cluster, nl_query=nl_query, max_retries=retry_count, model=model
        )
        for cluster in plan.chunks
    ]
    results = await run_parallel(
        tasks, labels=[f"chunk-{i}" for i in range(len(tasks))]
    )

    conceptual_models: List[ConceptualModel] = []

    for i, result in enumerate(results):
        if result is None:
            logger.warning(
                f"[Stage 2] shard {i + 1} ER-extraction failed entirely (isolated -- "
                f"siblings unaffected) -- treated as an empty conceptual model rather "
                f"than crashing the whole run."
            )
            result = (
                ConceptualModel(
                    entities=[], relationships=[], functional_dependencies=[]
                ),
                [],
                0,
            )
        cm, fix_hist, t_gen = result
        total_tokens += t_gen
        conceptual_models.append(cm)
        logger.info(
            f"[Stage 2] Shard {i + 1} completed. Entities: {len(cm.entities)}, Rels: {len(cm.relationships)}"
        )

    dump_artifact(artifact_dir, "01_shard_conceptual_models", conceptual_models)

    logger.info("[Stage 2] Global Merging via Multi-Verse Correlation Clustering...")
    from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards

    if not conceptual_models:
        raise ValueError("No conceptual models generated.")

    combined_cm, flags = merge_all_shards(conceptual_models, facts)
    logger.info(
        f"[Stage 2] Multi-Verse Merging completed. Total entities: {len(combined_cm.entities)}"
    )
    dump_artifact(artifact_dir, "02_merged_conceptual_model", combined_cm)

    if flags:
        logger.info(
            f"[Stage 2] Mathematical Engine raised {len(flags)} tension flags. Building Conflict Graph..."
        )

        G, flag_map = _build_conflict_graph(flags)

        from src.pipeline.stage2.agents.adjudicator.agent import resolve_conflicts

        adjudicator_tasks = []
        components = list(nx.connected_components(G))

        fact_dict = {f.id: f.fact for f in facts}

        for comp in components:
            comp_ents = list(comp)
            comp_flag_set: List[str] = []
            comp_facts: set = set()
            subgraph_str = ""

            for e_name in comp_ents:
                for f in flag_map.get(e_name, []):
                    comp_flag_set.append(str(f))

                ent = next((e for e in combined_cm.entities if e.name == e_name), None)
                if ent:
                    subgraph_str += (
                        f"ENTITY {ent.name}: {[a.name for a in ent.attributes]}\n"
                    )
                    for fid in ent.source_fact_ids:
                        if fid in fact_dict:
                            comp_facts.add(fact_dict[fid])

            adjudicator_tasks.append(
                resolve_conflicts(
                    subgraph=subgraph_str,
                    facts="\n".join(comp_facts),
                    flags="\n".join(comp_flag_set),
                    model=model,
                )
            )

        logger.info(
            f"[Stage 2] Dispatching {len(adjudicator_tasks)} parallel Adjudicator queries..."
        )
        adj_results = await run_parallel(
            adjudicator_tasks,
            labels=[f"component-{i}" for i in range(len(adjudicator_tasks))],
        )

        all_patches = []
        for res_pair in adj_results:
            if res_pair is None:
                logger.warning(
                    "[Stage 2] an Adjudicator component query failed entirely "
                    "(isolated -- siblings unaffected) -- its conflicts stay "
                    "unresolved this pass."
                )
                continue
            res, tokens = res_pair
            total_tokens += tokens
            all_patches.extend(res.resolutions)

        logger.info(
            f"[Stage 2] Adjudicator returned {len(all_patches)} resolution patches. Applying..."
        )
        combined_cm = apply_adjudicator_patches(combined_cm, all_patches)

        # Post-merge relationship deduplication
        seen_rel_keys: set = set()
        deduped_rels = []
        for r in combined_cm.relationships:
            p_keys = tuple(sorted([p.entity for p in r.participants]))
            key = (r.name, p_keys)
            if key not in seen_rel_keys:
                seen_rel_keys.add(key)
                deduped_rels.append(r)
        if len(deduped_rels) != len(combined_cm.relationships):
            logger.info(
                f"[Stage 2] Post-adjudication dedup: removed {len(combined_cm.relationships) - len(deduped_rels)} duplicate relationships"
            )
            combined_cm.relationships = deduped_rels

    # Dumped unconditionally, right before the call that can raise (e.g. a false-
    # positive FK cycle) -- this is the artifact that would have let us diagnose
    # the 2026-07-08 spurious reversed-FK bug without a live re-run.
    dump_artifact(artifact_dir, "03_final_conceptual_model_before_mapping", combined_cm)

    logger.info(
        "[Stage 2] Deterministic mapping of conceptual model to relational schema..."
    )
    global_schema = map_conceptual_to_relational(combined_cm)
    dump_artifact(artifact_dir, "04_global_schema", global_schema)

    # ... (Rest of existing FK structural fallback code)
    from src.util.schema_model.schema import to_snake_case
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

        if not matched_rel and global_schema.relationships:
            table_fk_targets = sorted(
                {
                    to_snake_case(fk.referred_table).lower()
                    for fk in global_schema.relationships
                    if fk.referencing_table == table.name
                }
            )
            for r in combined_cm.relationships:
                r_parts = sorted(
                    {to_snake_case(p.entity).lower() for p in r.participants}
                )
                if table_fk_targets and set(r_parts).issubset(set(table_fk_targets)):
                    matched_rel = r
                    break

        # Seed from the schema's own provenance FIRST, unconditionally. The
        # mapper stamped these on as it built the table, so they are authoritative
        # and available even for tables no conceptual entity corresponds to --
        # notably the junction tables the mapper synthesizes for M:N and n-ary
        # relationships, which the name/FK matching below structurally cannot
        # find. Stage 3's sharding reads this registry, so leaving junctions
        # unregistered under-reported their facts downstream too, not just in the
        # coverage warning.
        own_provenance: set[int] = set(table.source_fact_ids or [])
        for column in table.columns:
            own_provenance.update(column.source_fact_ids or [])
        for fk in global_schema.relationships or []:
            if fk.referencing_table == table.name:
                own_provenance.update(fk.source_fact_ids or [])
        if own_provenance:
            registry.register_table_facts(table.name, sorted(own_provenance))

        # The conceptual-model match then ADDS anything the mapper did not stamp
        # directly onto the table (e.g. facts attached to the entity as a whole).
        if matched_entity:
            registry.register_table_facts(table.name, matched_entity.source_fact_ids)
        elif matched_rel:
            registry.register_table_facts(table.name, matched_rel.source_fact_ids)
        elif not own_provenance:
            logger.warning(
                f"[Stage 2] Table '{table.name}' has NO fact provenance from either "
                f"the schema itself or a conceptual-model match; facts contributing "
                f"to it will be reported as uncovered."
            )

    uncovered = _compute_uncovered_facts(facts, global_schema, registry)

    cert_report = None
    if enable_audit:
        logger.info("[Stage 2] Running Schema Certifier (Rule compliance check)...")
        cert_report, t_cert = await certify_compliance(
            schema=global_schema,
            goal=analytical_goal,
            enriched_nl=facts,
            model=model,
        )
        total_tokens += t_cert
        dump_artifact(artifact_dir, "05_cert_report", cert_report)

        if cert_report.patches:
            logger.info(
                f"[Stage 2] Schema Certifier found {len(cert_report.patches)} violations. Applying rigid patches..."
            )
            from src.util.schema_ops.patching_engine import apply_patches

            apply_patches(global_schema, cert_report.patches, registry=registry)
            logger.info(f"[Stage 2] Applied {len(cert_report.patches)} patches.")

            # Patches can add tables the provenance pass above never saw, since
            # it ran before certification. Report them rather than leaving the
            # empty `if ...: pass` that used to stand here.
            for table in global_schema.tables:
                if not registry.get_facts_for_tables([table.name]):
                    logger.warning(
                        f"[Stage 2] Table '{table.name}' exists after certifier "
                        f"patches but has no registered fact provenance."
                    )
        else:
            logger.info("[Stage 2] Schema Certifier: PERFECT compliance.")
    else:
        logger.info(
            "[Stage 2] Skipping Schema Certifier (Audit disabled via ablation)."
        )

    segments_relational = []
    for cm in conceptual_models:
        try:
            segments_relational.append(map_conceptual_to_relational(cm))
        except ValueError as e:
            # A shard with no entities (e.g. a chunk of purely descriptive/non-schema
            # facts) is a legitimate degenerate case for this reporting-only field --
            # it must not abort a Stage 2 run whose real merged schema already succeeded.
            logger.warning(
                f"[Stage 2] Segment mapping produced an empty/invalid schema, "
                f"represented as empty for documentation purposes: {e}"
            )
            segments_relational.append(Schema(tables=[], relationships=[]))

    return (
        Output(
            segments=segments_relational,
            plan=plan,
            final_global_schema=global_schema,
            uncovered_fact_ids=uncovered,
            cert_report=cert_report if enable_audit else None,
            token_usage=total_tokens,
        ),
        total_tokens,
        registry,
    )

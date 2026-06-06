import asyncio
import json
from typing import Any, List, Dict, Optional, Tuple, Set

from src.pipeline.stage3.models import (
    BaseTableNode, JoinNode, AggNode, ColumnNode, ConstNode, LogicalNode, IfNode, ConditionPair,
    TableConstraintManifest, AlgebraicManifest, UnivariateDist,
    RelationalCycleError, NumericRange, DateRange
)
from src.pipeline.stage3.models.sql_models import LLMResponse, SQLGroundedConstraint
from src.pipeline.stage3.utils.constraint_builder import build_from_llm_input
from src.pipeline.stage3.models.distributions import (
    NormalDist, PoissonDist, ZipfDist, CategoricalDist
)
from src.pipeline.stage3.agents.metadata_extractor.agent import extract_metadata, get_agent
from src.orchestration.stage3.models import ShardMetadata, Output, RetryStep, RawSQLRule, HealingAttempt
from src.orchestration.stage2.refinement_sharding import get_deterministic_shards
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage3.middleware.fact_allocation import allocate_facts_to_shards
from src.util.ablation import AblationConfig

async def orchestrate(
    global_schema: Schema,
    registry: TableFactRegistry,
    all_facts: List[AtomicFact],
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None
) -> Tuple[Output, int]:
    """
    Orchestrates Stage 3: Algebraic Metadata Extraction with Self-Healing.
    """
    total_tokens = 0
    fact_map = {f.id: f for f in all_facts}

    # 1. Deterministic Sharding
    print("[Stage 3] Partitioning schema into FK clusters...")
    shard_table_sets = get_deterministic_shards(global_schema)

    global_table_manifests: Dict[str, TableConstraintManifest] = {}
    shard_metadata_list: List[ShardMetadata] = []
    extractor = get_agent(model)

    # 1.5. Advanced Fact Allocation (Similarity + Recovery)
    print(f"[Stage 3] Allocating {len(all_facts)} global facts to {len(shard_table_sets)} shards...")
    shard_allocations = allocate_facts_to_shards(all_facts, [set(s) for s in shard_table_sets], registry)

    # 2. Initial Extraction Pass (Parallelized)
    print(f"[Stage 3] Processing {len(shard_table_sets)} shards in parallel...")
    tasks = [
        _extract_shard_with_retry(
            i, list(shard_table_sets[i]), global_schema, registry, fact_map, extractor, model,
            allocated_fids=shard_allocations[i]
        ) for i, table_names in enumerate(shard_table_sets)
    ]
    results = await asyncio.gather(*tasks)

    for i, (manifest_map, raw_rules, tokens, history, f_ids) in enumerate(results):
        total_tokens += tokens
        global_table_manifests.update(manifest_map)
        shard_metadata_list.append(ShardMetadata(
            shard_index=i,
            table_names=shard_table_sets[i],
            allocated_fact_ids=f_ids,
            manifests=manifest_map,
            raw_sql_rules=raw_rules,
            retry_history=history,
            token_usage=tokens
        ))

    # 3. Validation & Healing Loop (Max 3 Attempts)
    healing_history = []
    final_rules = []
    for attempt in range(3):
        print(f"[Stage 3] Validation & Healing Attempt {attempt + 1}...")
        unify_cache = {}
        root_rules_map = {}
        errors_found = []

        try:
            # Try to unify and find errors
            for t_name, manifest in global_table_manifests.items():
                vm_errors = manifest._validate(global_schema)
                if vm_errors:
                    errors_found.append((t_name, "\n".join(vm_errors)))

                for rule in manifest.logical_rules:
                    # Relational Unification
                    unified = rule.unify(unify_cache, global_schema)
                    # Deduplicate by signature
                    root_rules_map[unified.get_signature(global_schema)] = unified

            if not errors_found:
                print("  [Stage 3] Validation Successful.")
                final_rules = list(root_rules_map.values())
                healing_history.append(HealingAttempt(attempt=attempt + 1, success=True, errors=[]))
                break

        except RelationalCycleError as e:
            print(f"  [Stage 3] Cycle Detected: {e}")
            errors_found.append(("CYCLE", str(e)))
        except Exception as e:
            print(f"  [Stage 3] unexpected Unification Error: {e}")
            errors_found.append(("ERROR", str(e)))

        healing_history.append(HealingAttempt(attempt=attempt + 1, success=False, errors=[str(e) for e in errors_found]))

        # Healing: Re-extract shards involved in errors (Serial for now but could be parallel)
        if errors_found:
            for culprit, err_msg in errors_found:
                # Find which shard contains the culprit
                for i, table_names in enumerate(shard_table_sets):
                    if culprit == "CYCLE" or culprit == "ERROR" or culprit in table_names:
                        print(f"  [Stage 3] Healing Shard {i+1} due to: {culprit}")
                        manifest_map, raw_rules, tokens, history, f_ids = await _extract_shard_with_retry(
                            i, table_names, global_schema, registry, fact_map, extractor, model, feedback=err_msg
                        )
                        total_tokens += tokens
                        global_table_manifests.update(manifest_map)
                        # Update shard metadata
                        shard_metadata_list[i].manifests.update(manifest_map)
                        shard_metadata_list[i].raw_sql_rules = raw_rules
                        shard_metadata_list[i].retry_history.extend(history)
                        shard_metadata_list[i].token_usage += tokens
        else:
            final_rules = list(root_rules_map.values())
            break

    # 4. Final Manifest Assembly
    final_manifest = AlgebraicManifest(
        table_manifests=global_table_manifests,
        global_rules=final_rules
    )

    return Output(
        global_manifest=final_manifest,
        shard_results=shard_metadata_list,
        total_tokens=total_tokens,
        execution_success=True,
        healing_history=healing_history
    ), total_tokens

from pydantic import ValidationError
from src.pipeline.stage3.utils.logic_parser import LogicParsingError

async def _extract_shard_with_retry(
    shard_idx: int,
    table_names: List[str],
    global_schema: Schema,
    registry: TableFactRegistry,
    fact_map: Dict[int, AtomicFact],
    extractor,
    model: Optional[str],
    feedback: Optional[str] = None,
    allocated_fids: Optional[List[int]] = None
) -> Tuple[Dict[str, TableConstraintManifest], List[RawSQLRule], int, List[RetryStep], List[int]]:
    print(f"  [Stage 3] Processing Shard {shard_idx + 1} (Tables: {', '.join(table_names)})...")

    shard_schema = Schema(
        tables=[t for t in global_schema.tables if t.name in table_names],
        relationships=[r for r in (global_schema.relationships or [])
                      if r.referencing_table in table_names and r.referred_table in table_names]
    )

    if allocated_fids is not None:
        fact_ids: List[int] = list(allocated_fids)
    else:
        fact_ids = list(registry.get_facts_for_tables(table_names))

    grounded_fact_texts = [f"[ID {fid}] {fact_map[fid].fact}" for fid in fact_ids if fid in fact_map]

    max_retries = 3
    current_feedback = feedback
    total_tokens = 0
    tokens = 0
    history: List[RetryStep] = []

    for attempt in range(max_retries):
        try:
            sql_shard_manifest, tokens = await extract_metadata(
                table_name=", ".join(table_names),
                shard_schema_json=shard_schema.model_dump_json(),
                grounded_facts=grounded_fact_texts,
                validator_report=current_feedback,
                extractor=extractor,
                model=model
            )
            total_tokens += tokens

            # 1. First, validate the LLMResponse items against the SHARD SCHEMA
            # This triggers a retry if the LLM hallucinated tables/columns or produced invalid SQL.
            for dist in sql_shard_manifest.distributions:
                v_errors = dist._validate(shard_schema)
                if v_errors:
                     raise LogicParsingError(f"Distribution validation error: {v_errors[0]}", hint="Check table and column names in the shard schema.")

            for constraint in sql_shard_manifest.logical_constraints:
                v_errors = constraint._validate(shard_schema)
                if v_errors:
                     raise LogicParsingError(f"Logical constraint validation error: {v_errors[0]}", fragment=f"ON: {constraint.on}, COND: {constraint.condition}")

            # 2. Bridge validated SQL format to Algebraic format
            manifest_map: Dict[str, TableConstraintManifest] = {t: TableConstraintManifest(table_name=t) for t in table_names}
            raw_rules_captured = []

            # 2a. Map Distributions and Ranges
            for dist in sql_shard_manifest.distributions:
                t_name = dist.table_name.upper()
                if t_name not in manifest_map: continue # Should be caught by validation, but safety first

                # Check if it's a NumericRange/DateRange vs a Probability Distribution
                d_val = dist.distribution
                if isinstance(d_val, (NumericRange, DateRange)):
                    # Distribute to numeric_bounds (normalized to NumericRange for Stage 4)
                    if isinstance(d_val, (NumericRange, DateRange)):
                        # PRESERVE FACT REFERENCES
                        d_val.fact_references = dist.distribution_ref

                        if isinstance(d_val, NumericRange):
                            manifest_map[t_name].numeric_bounds[dist.column_name] = d_val
                    # TODO: Add DateRange support to compiler if needed
                else:
                    # Probability Distribution
                    manifest_map[t_name].distributions[dist.column_name] = dist

            # 2b. Map Logical Rules using the Bridge Parser
            for r_sql in sql_shard_manifest.logical_constraints:
                # CAPTURE RAW RULE FOR VISIBILITY
                raw_rules_captured.append(RawSQLRule(
                    on=r_sql.on,
                    condition=r_sql.condition,
                    fact_references=r_sql.fact_references
                ))

                logic_node, err = build_from_llm_input(r_sql.on, r_sql.condition, shard_schema)
                if logic_node:
                    if isinstance(logic_node, IfNode):
                        node = logic_node
                    else:
                        node = IfNode(
                            anchor_table=logic_node.table,
                            on=r_sql.on, # Capture SQL Scope
                            pairs=[ConditionPair(
                                condition=LogicalNode(
                                    operator="ALWAYS",
                                    table=logic_node.table,
                                    column_1=ColumnNode(name="id"),
                                    column_2=ColumnNode(name="id")
                                ),
                                result=logic_node
                            )]
                        )

                    # Ensure on is set even if logic_node was already IfNode
                    node.on = r_sql.on
                    node.fact_references = r_sql.fact_references

                    # Resolve physical anchor table
                    t_anchor = None
                    anchor_node = node.anchor_table

                    # 1. Direct Base Table
                    if hasattr(anchor_node, "name") and not hasattr(anchor_node, "l_table"):
                        t_anchor = anchor_node.name.upper()
                    # 2. Join Node (Heuristic: Use Backbone Side)
                    elif hasattr(anchor_node, "get_backbone_side"):
                        backbone = anchor_node.get_backbone_side(global_schema)  # type: ignore[union-attr]
                        if hasattr(backbone, "name"):
                            t_anchor = backbone.name.upper()
                    # 3. Fallback to first table origin in pairs
                    elif hasattr(anchor_node, "table") and isinstance(anchor_node.table, str):  # type: ignore[union-attr]
                        t_anchor = anchor_node.table.upper()  # type: ignore[union-attr]

                    # NORMALIZATION CHECK
                    found_anchor = False
                    if t_anchor:
                        for m_tname in manifest_map.keys():
                            if m_tname.upper() == t_anchor:
                                manifest_map[m_tname].logical_rules.append(node)
                                found_anchor = True

                                # FLAG NULLABLE COLUMNS
                                null_cols = _extract_nullable_columns(node)
                                for nc in null_cols:
                                    if nc not in manifest_map[m_tname].nullable_columns:
                                        manifest_map[m_tname].nullable_columns.append(nc)
                                break

                    if not found_anchor:
                        # LOG TO SHARD BUT DONT FAIL - Fall back to global bucket later
                        print(f"  [Stage 3] Warning: Rule anchor '{t_anchor}' not in shard. Adding to shard-level temp rules.")
                        # Create a dummy Entry in manifest map if needed? No, just add to a general list
                        # We will move these to global_rules in the next part
                        if "__GLOBAL__" not in manifest_map:
                            manifest_map["__GLOBAL__"] = TableConstraintManifest(table_name="__GLOBAL__")
                        manifest_map["__GLOBAL__"].logical_rules.append(node)
                else:
                    raise LogicParsingError(f"SQL Bridge Error: {err}", fragment=f"ON: {r_sql.on}, COND: {r_sql.condition}")

            history.append(RetryStep(
                attempt=attempt + 1,
                feedback=current_feedback,
                token_usage=tokens
            ))

            return manifest_map, raw_rules_captured, total_tokens, history, fact_ids

        except (ValidationError, LogicParsingError) as e:
            err_type = "SCHEMA_VALIDATION" if isinstance(e, ValidationError) else "LOGIC_PARSING"
            print(f"  [Stage 3] {err_type} Error in attempt {attempt + 1}: {e}")

            # Generate Repair Hint
            repair_hint = f"### MATHEMATICAL VALIDATION REPORT\n"
            if isinstance(e, LogicParsingError):
                repair_hint += f"RELATIONAL ERROR: {e.message}\n"
                if e.fragment: repair_hint += f"FRAGILE CODE: {e.fragment}\n"
                if e.hint: repair_hint += f"HINT: {e.hint}\n"
            else:
                 repair_hint += f"SCHEMA VALIDATION ERROR: {str(e)}\n"

            repair_hint += "Please correct the output to match the expected algebraic schema."

            history.append(RetryStep(
                attempt=attempt + 1,
                feedback=current_feedback,
                error=str(e),
                token_usage=tokens if 'tokens' in locals() else 0
            ))

            current_feedback = repair_hint

            if attempt == max_retries - 1:
                print(f"  [Stage 3] Critical: Failed to extract shard {shard_idx + 1} after {max_retries} attempts.")
                return {}, [], total_tokens, history, fact_ids
        except Exception as e:
            print(f"  [Stage 3] Unexpected Error in shard extraction: {e}")
            history.append(RetryStep(
                attempt=attempt + 1,
                feedback=current_feedback,
                error=str(e),
                token_usage=tokens if 'tokens' in locals() else 0
            ))
            if attempt == max_retries - 1:
                return {}, [], total_tokens, history, fact_ids
            current_feedback = f"### MATHEMATICAL VALIDATION REPORT\nUnexpected Error: {str(e)}"

    return {}, [], total_tokens, history, fact_ids

def _extract_nullable_columns(node: Any) -> Set[str]:
    """Recursively finds columns involved in NULL-related constraints."""
    cols = set()
    # 1. Handle LogicalNode
    if hasattr(node, "operator"):
        op = getattr(node, "operator", "")
        if op in ["IS_NULL", "IS_NOT_NULL"]:
            c1 = getattr(node, "column_1", None)
            if c1 and hasattr(c1, "name"):
                cols.add(c1.name)

        # Recurse for CombinationNode
        operands = getattr(node, "operands", [])
        for o in operands:
            cols.update(_extract_nullable_columns(o))

    # 2. Recurse for IfNode
    pairs = getattr(node, "pairs", [])
    for pair in pairs:
        cols.update(_extract_nullable_columns(pair.condition))
        cols.update(_extract_nullable_columns(pair.result))

    return cols

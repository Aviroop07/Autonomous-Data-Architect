import asyncio
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel

from src.pipeline.stage3.models import TableConstraintManifest, AlgebraicManifest
from src.pipeline.stage3.middleware.satisfiability import (
    check_structural_satisfiability,
    check_state_constraint_satisfiability,
    format_satisfiability_issues_for_retry,
)
from src.pipeline.stage3.models.sql_models import CardinalityConstraint, FanoutConstraint, LLMResponse, SQLGroundedConstraint
from src.pipeline.stage3.agents.metadata_extractor.agent import extract_metadata, get_agent
from src.orchestration.stage3.models import ShardMetadata, Output, RetryStep, RawSQLRule, HealingAttempt

from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage3.middleware.fact_allocation import allocate_facts_to_shards
from src.util.config.ablation import AblationConfig


class StateConstraintEntry(BaseModel):
    signature: str
    source_table: str
    constraint: SQLGroundedConstraint


class CardinalityConstraintEntry(BaseModel):
    signature: str
    source_table: str
    constraint: CardinalityConstraint


class FanoutConstraintEntry(BaseModel):
    signature: str
    source_table: str
    constraint: FanoutConstraint

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
    # 1. Deterministic Sharding
    print("[Stage 3] Partitioning schema into FK clusters...")
    shard_table_sets = get_deterministic_shards(global_schema)

    global_table_manifests: List[TableConstraintManifest] = []
    shard_metadata_list: List[ShardMetadata] = []
    extractor = get_agent(model)

    # 1.5. Advanced Fact Allocation (Similarity + Recovery)
    print(f"[Stage 3] Allocating {len(all_facts)} global facts to {len(shard_table_sets)} shards...")
    shard_allocations = allocate_facts_to_shards(all_facts, [set(s) for s in shard_table_sets], registry)

    # 2. Initial Extraction Pass (Parallelized)
    print(f"[Stage 3] Processing {len(shard_table_sets)} shards in parallel...")
    tasks = [
        _extract_shard_with_retry(
            i, list(shard_table_sets[i]), global_schema, registry, all_facts, extractor, model,
            allocated_fids=shard_allocations[i],
        ) for i, table_names in enumerate(shard_table_sets)
    ]
    results = await asyncio.gather(*tasks)

    for i, (manifests, raw_rules, tokens, history, f_ids) in enumerate(results):
        total_tokens += tokens
        shard_metadata_list.append(ShardMetadata(
            shard_index=i,
            table_names=shard_table_sets[i],
            allocated_fact_ids=f_ids,
            manifests=manifests,
            raw_sql_rules=raw_rules,
            retry_history=history,
            token_usage=tokens
        ))
    global_table_manifests = _build_global_table_manifests(shard_metadata_list)

    # 3. Validation & Healing Loop (Max 3 Attempts)
    healing_history = []
    final_state_constraints = []
    final_cardinality_constraints = []
    final_fanout_constraints = []
    final_tunable_knobs = []
    for attempt in range(3):
        print(f"[Stage 3] Validation & Healing Attempt {attempt + 1}...")
        state_constraint_entries: List[StateConstraintEntry] = []
        cardinality_constraint_entries: List[CardinalityConstraintEntry] = []
        fanout_constraint_entries: List[FanoutConstraintEntry] = []
        errors_found = []

        try:
            # Try to unify and find errors
            for manifest in global_table_manifests:
                t_name = manifest.table_name
                vm_errors = manifest._validate(global_schema)
                if vm_errors:
                    errors_found.append((t_name, "\n".join(vm_errors)))

                for constraint in manifest.state_constraints:
                    _add_state_constraint_entry(state_constraint_entries, t_name, constraint)

                for constraint in manifest.cardinality_constraints:
                    _add_cardinality_constraint_entry(cardinality_constraint_entries, t_name, constraint)

                for constraint in manifest.fanout_constraints:
                    _add_fanout_constraint_entry(fanout_constraint_entries, t_name, constraint)

            if not errors_found:
                state_constraints = [entry.constraint for entry in state_constraint_entries]
                satisfiability_issues = check_state_constraint_satisfiability(state_constraints, global_schema)
                for issue in satisfiability_issues:
                    culprit = _find_satisfiability_issue_culprit(issue.target, state_constraint_entries)
                    errors_found.append((culprit, format_satisfiability_issues_for_retry([issue])))

            if not errors_found:
                cardinality_constraints = [entry.constraint for entry in cardinality_constraint_entries]
                fanout_constraints = [entry.constraint for entry in fanout_constraint_entries]
                structural_issues, tunable_knobs = check_structural_satisfiability(
                    cardinality_constraints,
                    fanout_constraints,
                    global_schema,
                )
                for issue in structural_issues:
                    culprit = _find_structural_issue_culprit(
                        issue.target,
                        cardinality_constraint_entries,
                        fanout_constraint_entries,
                    )
                    errors_found.append((culprit, format_satisfiability_issues_for_retry([issue])))

            if not errors_found:
                print("  [Stage 3] Validation Successful.")
                final_state_constraints = state_constraints
                final_cardinality_constraints = cardinality_constraints
                final_fanout_constraints = fanout_constraints
                final_tunable_knobs = tunable_knobs
                healing_history.append(HealingAttempt(attempt=attempt + 1, success=True, errors=[]))
                break

        except Exception as e:
            print(f"  [Stage 3] unexpected validation error: {e}")
            errors_found.append(("ERROR", str(e)))

        healing_history.append(HealingAttempt(attempt=attempt + 1, success=False, errors=[str(e) for e in errors_found]))

        # Healing: Re-extract shards involved in errors (Serial for now but could be parallel)
        if errors_found:
            for culprit, err_msg in errors_found:
                # Find which shard contains the culprit
                for i, table_names in enumerate(shard_table_sets):
                    if culprit == "CYCLE" or culprit == "ERROR" or culprit in table_names:
                        print(f"  [Stage 3] Healing Shard {i+1} due to: {culprit}")
                        manifests, raw_rules, tokens, history, f_ids = await _extract_shard_with_retry(
                            i, table_names, global_schema, registry, all_facts, extractor, model,
                            feedback=err_msg,
                        )
                        total_tokens += tokens
                        # Update shard metadata
                        shard_metadata_list[i].manifests = manifests
                        shard_metadata_list[i].raw_sql_rules = raw_rules
                        shard_metadata_list[i].retry_history.extend(history)
                        shard_metadata_list[i].token_usage += tokens
                        global_table_manifests = _build_global_table_manifests(shard_metadata_list)
        else:
            final_state_constraints = [entry.constraint for entry in state_constraint_entries]
            final_cardinality_constraints = [entry.constraint for entry in cardinality_constraint_entries]
            final_fanout_constraints = [entry.constraint for entry in fanout_constraint_entries]
            break

    # 4. Final Manifest Assembly
    final_manifest = AlgebraicManifest(
        table_manifests=global_table_manifests,
        global_state_constraints=final_state_constraints,
        global_cardinality_constraints=final_cardinality_constraints,
        global_fanout_constraints=final_fanout_constraints,
        tunable_knobs=final_tunable_knobs,
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


def _build_global_table_manifests(shard_metadata_list: List[ShardMetadata]) -> List[TableConstraintManifest]:
    merged: List[TableConstraintManifest] = []
    for shard_metadata in shard_metadata_list:
        for manifest in shard_metadata.manifests:
            target = _get_or_create_table_manifest(merged, manifest.table_name)
            target.state_constraints.extend(manifest.state_constraints)
            target.cardinality_constraints.extend(manifest.cardinality_constraints)
            target.fanout_constraints.extend(manifest.fanout_constraints)
            for nullable_column in manifest.nullable_columns:
                if nullable_column not in target.nullable_columns:
                    target.nullable_columns.append(nullable_column)

    return merged


def _get_or_create_table_manifest(
    manifests: List[TableConstraintManifest],
    table_name: str,
) -> TableConstraintManifest:
    for manifest in manifests:
        if manifest.table_name.upper() == table_name.upper():
            return manifest
    manifest = TableConstraintManifest(table_name=table_name)
    manifests.append(manifest)
    return manifest


def _add_state_constraint_entry(
    entries: List[StateConstraintEntry],
    source_table: str,
    constraint: SQLGroundedConstraint,
) -> None:
    signature = constraint.get_signature()
    if not any(entry.signature == signature for entry in entries):
        entries.append(StateConstraintEntry(
            signature=signature,
            source_table=source_table,
            constraint=constraint,
        ))


def _add_cardinality_constraint_entry(
    entries: List[CardinalityConstraintEntry],
    source_table: str,
    constraint: CardinalityConstraint,
) -> None:
    signature = constraint.get_signature()
    if not any(entry.signature == signature for entry in entries):
        entries.append(CardinalityConstraintEntry(
            signature=signature,
            source_table=source_table,
            constraint=constraint,
        ))


def _add_fanout_constraint_entry(
    entries: List[FanoutConstraintEntry],
    source_table: str,
    constraint: FanoutConstraint,
) -> None:
    signature = constraint.get_signature()
    if not any(entry.signature == signature for entry in entries):
        entries.append(FanoutConstraintEntry(
            signature=signature,
            source_table=source_table,
            constraint=constraint,
        ))


def _format_grounded_facts(fact_ids: List[int], all_facts: List[AtomicFact]) -> List[str]:
    grounded: List[str] = []
    for fact_id in fact_ids:
        for fact in all_facts:
            if fact.id == fact_id:
                grounded.append(f"[ID {fact_id}] {fact.fact}")
                break
    return grounded


def _validate_sql_response_against_schema(sql_response: LLMResponse, shard_schema: Schema) -> None:
    for constraint in sql_response.logical_constraints:
        v_errors = constraint._validate(shard_schema)
        if v_errors:
            raise LogicParsingError(
                f"Logical constraint validation error: {v_errors[0]}",
                fragment=(
                    f"STATE_QUERY: {constraint.state_query}, "
                    f"PREDICATE: {constraint.left_operand} {constraint.operator} {constraint.right_operand}"
                ),
            )

    for constraint in sql_response.cardinality_constraints:
        v_errors = constraint._validate(shard_schema)
        if v_errors:
            raise LogicParsingError(
                f"Cardinality constraint validation error: {v_errors[0]}",
                fragment=f"TABLE: {constraint.table_name}, BOUNDS: {constraint.bounds()}",
            )

    for constraint in sql_response.fanout_constraints:
        v_errors = constraint._validate(shard_schema)
        if v_errors:
            raise LogicParsingError(
                f"Fanout constraint validation error: {v_errors[0]}",
                fragment=(
                    f"PARENT: {constraint.parent_table}, CHILD: {constraint.child_table}, "
                    f"MIN: {constraint.min_children_per_parent}, MAX: {constraint.max_children_per_parent}"
                ),
            )


async def _extract_shard_with_retry(
    shard_idx: int,
    table_names: List[str],
    global_schema: Schema,
    registry: TableFactRegistry,
    all_facts: List[AtomicFact],
    extractor,
    model: Optional[str],
    feedback: Optional[str] = None,
    allocated_fids: Optional[List[int]] = None,
) -> Tuple[List[TableConstraintManifest], List[RawSQLRule], int, List[RetryStep], List[int]]:
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

    grounded_fact_texts = _format_grounded_facts(fact_ids, all_facts)

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

            # 1. First, validate the LLMResponse items against the SHARD SCHEMA.
            # This triggers a retry if the LLM hallucinated tables/columns or produced invalid SQL.
            _validate_sql_response_against_schema(sql_shard_manifest, shard_schema)

            # 2. Store validated SQL state-table constraints.
            manifests: List[TableConstraintManifest] = [TableConstraintManifest(table_name=t) for t in table_names]
            raw_rules_captured = []

            for r_sql in sql_shard_manifest.logical_constraints:
                raw_rules_captured.append(RawSQLRule(
                    state_query=r_sql.state_query,
                    left_operand=r_sql.left_operand,
                    operator=r_sql.operator,
                    right_operand=r_sql.right_operand,
                    fact_references=r_sql.fact_references
                ))

                anchor_table = _find_constraint_anchor(r_sql, table_names)
                _get_or_create_table_manifest(manifests, anchor_table).state_constraints.append(r_sql)

            for cardinality_constraint in sql_shard_manifest.cardinality_constraints:
                anchor_table = _find_cardinality_anchor(cardinality_constraint, table_names)
                _get_or_create_table_manifest(manifests, anchor_table).cardinality_constraints.append(cardinality_constraint)

            for fanout_constraint in sql_shard_manifest.fanout_constraints:
                anchor_table = _find_fanout_anchor(fanout_constraint, table_names)
                _get_or_create_table_manifest(manifests, anchor_table).fanout_constraints.append(fanout_constraint)

            history.append(RetryStep(
                attempt=attempt + 1,
                feedback=current_feedback,
                token_usage=tokens
            ))

            return manifests, raw_rules_captured, total_tokens, history, fact_ids

        except (ValidationError, LogicParsingError) as e:
            err_type = "SCHEMA_VALIDATION" if isinstance(e, ValidationError) else "LOGIC_PARSING"
            print(f"  [Stage 3] {err_type} Error in attempt {attempt + 1}: {e}")

            # Generate Repair Hint
            repair_hint = f"### FEASIBILITY VALIDATION REPORT\n"
            if isinstance(e, LogicParsingError):
                repair_hint += f"RELATIONAL ERROR: {e.message}\n"
                if e.fragment: repair_hint += f"FRAGILE CODE: {e.fragment}\n"
                if e.hint: repair_hint += f"HINT: {e.hint}\n"
            else:
                 repair_hint += f"SCHEMA VALIDATION ERROR: {str(e)}\n"

            repair_hint += "Please correct the output to match the expected state-table constraint schema."

            history.append(RetryStep(
                attempt=attempt + 1,
                feedback=current_feedback,
                error=str(e),
                token_usage=tokens if 'tokens' in locals() else 0
            ))

            current_feedback = repair_hint

            if attempt == max_retries - 1:
                print(f"  [Stage 3] Critical: Failed to extract shard {shard_idx + 1} after {max_retries} attempts.")
                return [], [], total_tokens, history, fact_ids
        except Exception as e:
            print(f"  [Stage 3] Unexpected Error in shard extraction: {e}")
            history.append(RetryStep(
                attempt=attempt + 1,
                feedback=current_feedback,
                error=str(e),
                token_usage=tokens if 'tokens' in locals() else 0
            ))
            if attempt == max_retries - 1:
                return [], [], total_tokens, history, fact_ids
            current_feedback = f"### FEASIBILITY VALIDATION REPORT\nUnexpected Error: {str(e)}"

    return [], [], total_tokens, history, fact_ids


def _find_constraint_anchor(constraint: SQLGroundedConstraint, table_names: List[str]) -> str:
    query_upper = constraint.state_query.upper()
    mentioned_tables = re.findall(r"\b(?:FROM|JOIN)\s+([A-Z][A-Z0-9_]*)", query_upper)
    for mentioned in mentioned_tables:
        for table_name in table_names:
            if mentioned == table_name.upper():
                return table_name
    return table_names[0]


def _find_cardinality_anchor(constraint: CardinalityConstraint, table_names: List[str]) -> str:
    for table_name in table_names:
        if constraint.table_name.upper() == table_name.upper():
            return table_name
    return table_names[0]


def _find_fanout_anchor(constraint: FanoutConstraint, table_names: List[str]) -> str:
    for candidate in (constraint.child_table, constraint.parent_table):
        for table_name in table_names:
            if candidate.upper() == table_name.upper():
                return table_name
    return table_names[0]


def _find_satisfiability_issue_culprit(
    issue_target: str,
    state_constraint_entries: List[StateConstraintEntry],
) -> str:
    match = re.search(r"state_constraints\[(\d+)\]", issue_target)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(state_constraint_entries):
            return state_constraint_entries[idx].source_table
    return "ERROR"


def _find_structural_issue_culprit(
    issue_target: str,
    cardinality_constraint_entries: List[CardinalityConstraintEntry],
    fanout_constraint_entries: List[FanoutConstraintEntry],
) -> str:
    cardinality_match = re.search(r"cardinality_constraints\[(\d+)\]", issue_target)
    if cardinality_match:
        idx = int(cardinality_match.group(1))
        if 0 <= idx < len(cardinality_constraint_entries):
            return cardinality_constraint_entries[idx].source_table

    fanout_match = re.search(r"fanout_constraints\[(\d+)\]", issue_target)
    if fanout_match:
        idx = int(fanout_match.group(1))
        if 0 <= idx < len(fanout_constraint_entries):
            return fanout_constraint_entries[idx].source_table

    if fanout_constraint_entries:
        return fanout_constraint_entries[0].source_table
    if cardinality_constraint_entries:
        return cardinality_constraint_entries[0].source_table
    return "ERROR"

from typing import List, Optional, Tuple
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.agents.schema_architect.agent import run_schema_architect
from src.pipeline.stage2.agents.domain_auditor.agent import audit_domain
from src.util.patching_engine import apply_patches

async def run_architect_self_correction_loop(
    facts: List[AtomicFact],
    max_retries: int,
    architect = None,
    model: Optional[str] = None
) -> Tuple[Schema, List[FixHistoryStep], int]:
    """
    Invokes SchemaArchitect with iterative self-correction.
    """
    total_tokens = 0
    fix_history: List[FixHistoryStep] = []

    # 1. Initial Generation
    schema, tokens = await run_schema_architect(chunk_facts=facts, architect=architect, model=model)
    total_tokens += tokens

    # 2. Self-Correction Loop
    for attempt in range(max_retries):
        errors = schema._validate()
        if not errors:
            break

        print(f"    [Architect Self-Correction] Attempt {attempt+1}: {len(errors)} errors.")

        # Save history
        fix_history.append(FixHistoryStep(
            attempt=attempt + 1,
            errors=errors,
            corrections=[], # Legacy field, can be empty or used for reasoning
            fixed_schema=str(schema),
            schema_state=schema.model_copy(deep=True)
        ))

        # Re-invoke architect with current state and errors
        schema, tokens = await run_schema_architect(
            chunk_facts=facts,
            base_schema=schema,
            errors=errors,
            architect=architect,
            model=model
        )
        total_tokens += tokens

    return schema, fix_history, total_tokens

async def run_auditor_self_correction_loop(
    shard_schema: Schema,
    intelligence,
    fact_clusters,
    max_retries: int,
    registry: Optional[TableFactRegistry] = None,
    auditor = None,
    model: Optional[str] = None
) -> Tuple[Schema, List[FixHistoryStep], int]:
    """
    Invokes DomainAuditor with iterative patch-repair feedback.
    """
    total_tokens = 0
    fix_history: List[FixHistoryStep] = []
    structural_errors = []

    for attempt in range(max_retries):
        # 1. Audit
        report, t_audit = await audit_domain(
            schema=shard_schema,
            intelligence=intelligence,
            fact_clusters=fact_clusters,
            structural_errors=structural_errors,
            agent=auditor,
            model=model
        )
        total_tokens += t_audit

        if not report.patches:
            break

        # 2. Apply Patches
        temp_schema = shard_schema.model_copy(deep=True)
        # Gather all fact IDs involved in this audit to ground any newly added tables
        owner_fact_ids = []
        if fact_clusters:
            for facts in fact_clusters.values():
                owner_fact_ids.extend([f.id for f in facts])
            owner_fact_ids = list(set(owner_fact_ids))

        apply_patches(temp_schema, report.patches, registry=registry, owner_fact_ids=owner_fact_ids)

        # 3. Deterministic Validation
        structural_errors = temp_schema._validate()

        # If any patches failed to apply or if they introduced errors
        if not structural_errors:
            # Success!
            shard_schema.tables = temp_schema.tables
            shard_schema.relationships = temp_schema.relationships
            break
        else:
            # Feedback to Auditor in next iteration
            print(f"    [Auditor Self-Correction] Attempt {attempt+1}: Patches introduced {len(structural_errors)} errors.")
            fix_history.append(FixHistoryStep(
                attempt=attempt + 1,
                errors=structural_errors,
                corrections=[],
                fixed_schema=str(temp_schema),
                schema_state=temp_schema
            ))
            # Next iteration will pass these structural_errors back to the auditor

    return shard_schema, fix_history, total_tokens

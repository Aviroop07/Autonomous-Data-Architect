from typing import List, Optional, Tuple
from src.orchestration.stage3.models import Output, PatchRepairStep
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage3.agents.domain_expert.agent import generate_style_guide
from src.pipeline.stage3.agents.shard_refiner.agent import refine_shard, get_agent
from src.pipeline.stage3.agents.global_linker.agent import link_global, get_agent as get_linker
from src.pipeline.stage3.agents.compliance_certifier.agent import certify_compliance, get_agent as get_certifier_agent
from src.pipeline.stage3.agents.patch_repair.agent import repair_patches, get_agent as get_repair_agent
from src.pipeline.stage2.agents.schema_corrector.agent import fix_schema_step, get_agent as get_corrector
from src.pipeline.stage3.middleware.patcher import apply_patches
from src.pipeline.stage2.middleware.schema_merging.merger import SchemaMerger
from src.pipeline.stage3.models.patch import CritiqueReport, SchemaPatch, PatchValidationError
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.util.documentation import render_schema_to_base64

def _iterative_patch_apply(
    schema: Schema, 
    patches: List[SchemaPatch]
) -> Tuple[Schema, bool, List[PatchValidationError]]:
    """
    Iteratively applies patches to a DEEP COPY of the schema.
    If a patch is valid, it's applied immediately, allowing dependent patches to potentially become valid.
    Returns (updated_schema_copy, all_applied_success, remaining_errors).
    """
    temp_schema = schema.model_copy(deep=True)
    remaining = list(enumerate(patches)) # (original_index, patch)
    
    any_progress = True
    while any_progress and remaining:
        any_progress = False
        next_remaining = []
        for idx, patch in remaining:
            # Validate against the current state of temp_schema
            errors = patch._validate(temp_schema)
            if not errors:
                # Apply this single patch to temp_schema
                apply_patches(temp_schema, [patch])
                any_progress = True
            else:
                next_remaining.append((idx, patch))
        remaining = next_remaining
        
    if not remaining:
        return temp_schema, True, []
        
    # Collate final errors for patches that never became valid
    final_errors = []
    for idx, patch in remaining:
        # Re-run validation to get the specific errors against the final state
        final_errors.append(PatchValidationError(
            patch_index=idx,
            action=patch.action,
            errors=patch._validate(temp_schema)
        ))
        
    return temp_schema, False, final_errors

def _apply_patches_with_retry(
    schema: Schema, 
    report: CritiqueReport, 
    repair_agent = None,
    model: Optional[str] = None
) -> Tuple[CritiqueReport, List[PatchRepairStep]]:
    """
    Orchestration helper: Validates patches iteratively, repairs them if necessary, and applies them.
    Returns (final_report, repair_history).
    """
    repair_history: List[PatchRepairStep] = []
    
    # 1. Iterative Validation & Partial Application
    temp_schema, success, patch_errors = _iterative_patch_apply(schema, report.patches)
    
    if success:
        # All succeeded. Update original schema in-place.
        schema.tables = temp_schema.tables
        schema.relationships = temp_schema.relationships
        return report, []
    
    # 2. Repair Remaining
    # We pass the PARTIALLY PATCHED schema to the repair agent
    repaired_report, _ = repair_patches(
        schema=temp_schema, 
        report=report,
        errors=patch_errors,
        repair_agent=repair_agent,
        model=model
    )
    
    repair_history.append(PatchRepairStep(
        original_errors=patch_errors,
        repaired_patches=[p.model_dump() for p in repaired_report.patches]
    ))
    
    # 3. Final Step: Apply Repaired Patches (also iteratively to be safe)
    # Start from the temp_schema (which has the valid ones applied)
    final_schema, final_success, final_errors = _iterative_patch_apply(temp_schema, repaired_report.patches)
    
    # Update the original schema reference
    schema.tables = final_schema.tables
    schema.relationships = final_schema.relationships
    
    return repaired_report, repair_history

from src.orchestration.stage2.utils import run_correction_loop

from src.orchestration.stage3.context_filter import filter_distributional_facts

def orchestrate(
    segments: List[Schema],
    facts: List[AtomicFact],
    modeling_facts: List[AtomicFact],
    plan: ChunkedPlan,
    domain: str,
    analytical_goal: str,
    retry_count: int = 5,
    model: Optional[str] = None
) -> Output:
    """
    Orchestrates Stage 3: Distributed Precision 2.0 with Visual Tracking.
    """
    import os
    from src.orchestration.stage3.models import ShardStep
    
    # 1. Initialize Agents & Strategy
    corrector = get_corrector(model)
    repair_agent = get_repair_agent(model)
    modeling_fact_ids = {f.id for f in modeling_facts}
    style_guide, _ = generate_style_guide(domain, modeling_facts, model=model)
    
    shard_steps: List[ShardStep] = []
    
    # Pre-merging Shard Refinement
    for i, (seg, chunk_facts) in enumerate(zip(segments, plan.chunks)):
        # --- PASS A: MINIMALISM (Strict Fact Fidelity) ---
        # Note: No style_guide or workload context passed to ensure minimalism
        min_refiner = get_agent("", analytical_goal, model=model) 
        report_a, _ = refine_shard(seg, chunk_facts, "", analytical_goal, agent=min_refiner, model=model)
        final_report_a, _ = _apply_patches_with_retry(seg, report_a, repair_agent=repair_agent, model=model)
        run_correction_loop(seg, corrector, retry_count, model=model)
        
        # Capture Pass A State
        pass_a_schema = seg.model_copy(deep=True)
        img_a_uri = render_schema_to_base64(pass_a_schema)
        
        # --- PASS B: REALISM & WORKLOAD (Style Guide + Dist Facts) ---
        workload_facts = filter_distributional_facts(seg, facts, modeling_fact_ids, model=model)
        combined_facts = chunk_facts + workload_facts
        real_refiner = get_agent(style_guide, analytical_goal, model=model)
        
        report_b, _ = refine_shard(seg, combined_facts, style_guide, analytical_goal, agent=real_refiner, model=model)
        final_report_b, _ = _apply_patches_with_retry(seg, report_b, repair_agent=repair_agent, model=model)
        run_correction_loop(seg, corrector, retry_count, model=model)
        
        # Capture Pass B State
        pass_b_schema = seg.model_copy(deep=True)
        img_b_uri = render_schema_to_base64(pass_b_schema)

        shard_steps.append(ShardStep(
            chunk_index=i,
            pass_a_schema=pass_a_schema,
            pass_a_report=final_report_a,
            pass_a_image_uri=img_a_uri,
            pass_b_schema=pass_b_schema,
            pass_b_report=final_report_b,
            pass_b_image_uri=img_b_uri
        ))

    # 4. Global Stitching (Authoritative Election)
    import re
    master_entities = []
    me_match = re.search(r"## Master Entities\n(.*?)\n##", style_guide + "\n##", re.DOTALL)
    if me_match:
        master_entities = re.findall(r"[*+-]?\s*[`]?([A-Z_]+)[`]?", me_match.group(1))

    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    global_schema = merger.merge_segments(segments, authoritative_dimensions=master_entities)
    run_correction_loop(global_schema, corrector, retry_count, model=model)
    
    merge_uri = render_schema_to_base64(global_schema)

    # 5. Global Linker Iteration
    linker = get_linker(style_guide, modeling_facts, model=model)
    for _ in range(retry_count):
        recon_report, _ = link_global(global_schema, style_guide, modeling_facts, agent=linker, model=model)
        if not recon_report.patches:
            break
        _apply_patches_with_retry(global_schema, recon_report, repair_agent=repair_agent, model=model)
        run_correction_loop(global_schema, corrector, retry_count, model=model)

    final_uri = render_schema_to_base64(global_schema)

    return Output(
        global_schema=global_schema,
        style_guide=style_guide,
        shard_steps=shard_steps,
        merge_image_uri=merge_uri,
        final_image_uri=final_uri
    )

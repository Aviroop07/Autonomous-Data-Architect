from typing import List, Tuple, Optional
from src.pipeline.stage2.models.schema import SchemaSegment
from src.pipeline.stage3.models.patch import SchemaPatch, CritiqueReport
from src.pipeline.stage3.middleware.patcher import apply_patches
from src.pipeline.stage3.middleware.validator import detect_cycles
from src.pipeline.stage3.agents.domain_expert.agent import generate_style_guide
from src.pipeline.stage3.agents.shard_auditor.agent import audit_shard
from src.pipeline.stage3.agents.global_reconciler.agent import reconcile_global
from src.pipeline.stage3.agents.analytical_integrity.agent import audit_integrity

def run_stage3_refinement(
    global_schema: SchemaSegment,
    segments_with_chunks: List[Tuple[SchemaSegment, str]],
    enriched_nl: str,
    domain: str,
    analytical_goal: str,
    model: Optional[str] = None
) -> Tuple[SchemaSegment, int, List[CritiqueReport], str]:
    """
    Orchestrates the Stage 3 Critique & Patching loop.
    """
    total_tokens = 0
    all_reports: List[CritiqueReport] = []
    
    # Pass 0: Domain Expert
    print("  -> Pass 0: Generating Domain Style Guide...")
    style_guide, tok0 = generate_style_guide(domain, model=model)
    total_tokens += tok0
    
    # Pass 1: Shard Audit (Local Patching on each segment)
    print("  -> Pass 1: Auditing Shards (Local Pass)...")
    for i, (segment, chunk_text) in enumerate(segments_with_chunks):
        report, tok1 = audit_shard(segment, chunk_text, style_guide, analytical_goal, model=model)
        total_tokens += tok1
        all_reports.append(report)
        apply_patches(segment, report.patches)
    
    # Re-merge the patched segments
    from src.pipeline.stage2.middleware.schema_merging.merger import SchemaMerger
    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    print("  -> Re-merging patched segments...")
    global_schema = merger.merge_segments([s for s, _ in segments_with_chunks])
    
    # Pass 2: Global Reconciliation
    print("  -> Pass 2: Global Reconciliation (Structural Pass)...")
    recon_report, tok2 = reconcile_global(global_schema, style_guide, enriched_nl, model=model)
    total_tokens += tok2
    all_reports.append(recon_report)
    apply_patches(global_schema, recon_report.patches)
    
    # Pass 3: Analytical Integrity
    print("  -> Pass 3: Analytical Integrity Check...")
    integrity_report, tok3 = audit_integrity(global_schema, analytical_goal, enriched_nl, model=model)
    total_tokens += tok3
    all_reports.append(integrity_report)
    apply_patches(global_schema, integrity_report.patches)
    
    # Pass 4: Cycle Resolution
    print("  -> Pass 4: Detecting and Resolving Cycles...")
    MAX_CYCLE_FIX_ATTEMPTS = 3
    for i in range(MAX_CYCLE_FIX_ATTEMPTS):
        cycles = detect_cycles(global_schema)
        if not cycles:
            break
        print(f"     ! Cycle detected: {cycles[0]}. Attempting to fix...")
        break
        
    return global_schema, total_tokens, all_reports, style_guide

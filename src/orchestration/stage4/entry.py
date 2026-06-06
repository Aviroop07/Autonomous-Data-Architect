import json
import re
from typing import Optional, Tuple, List, Dict, Set
import pandas as pd
import numpy as np

from src.pipeline.stage3.models import AlgebraicManifest
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage4.models import SynthesisResult, ParameterManifest
from src.pipeline.stage4.agents.parameter_agent.agent import derive_parameters, get_agent as get_parameter_agent
from src.pipeline.stage4.agents.semantic_agent.agent import infill_semantics, get_agent as get_semantic_agent
from src.pipeline.stage4.compiler import MinimalCompiler
from src.pipeline.stage4.smoke_test import run_smoke_test
from src.util.ablation import AblationConfig

async def orchestrate(
    global_schema: Schema,
    manifest: AlgebraicManifest,
    business_facts: List[AtomicFact],
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None
) -> Tuple[SynthesisResult, int]:
    """
    Orchestrates Stage 4: Hardened Synthesis.
    """
    total_tokens = 0
    schema_json = global_schema.model_dump_json()
    fact_texts = [f.fact for f in business_facts]

    # Extract nullable columns from manifest to pass to parameter agent
    nullable_map = {
        t_name: m.nullable_columns
        for t_name, m in manifest.table_manifests.items()
        if m.nullable_columns
    }

    # 1. Derive Parameters (Scales + Sparsity)
    print("[Stage 4] Deriving synthesis parameters (scales + sparsity)...")
    parameters, t_param = await derive_parameters(
        schema_json=schema_json,
        business_facts=fact_texts,
        nullable_columns_map=nullable_map,
        model=model
    )
    total_tokens += t_param
    param_map = {p.table_name: p for p in parameters.parameters}

    # 2. Generate Deterministic Skeleton
    print("[Stage 4] Compiling algebraic skeleton...")
    enable_logical = ablation_config.enable_logical_constraints if ablation_config else True
    compiler = MinimalCompiler(
        manifest=manifest,
        parameters=parameters,
        enable_logical_constraints=enable_logical,
    )
    skeleton_code = compiler.compile(schema_json)

    # 3. Infill Semantics (Probabilistic Flesh)
    print("[Stage 4] Infilling semantic descriptions...")
    _infill_raw, t_semantic = await infill_semantics(
        skeleton_code=skeleton_code,
        schema_json=schema_json,
        model=model
    )
    table_infill: Dict[str, str] = _infill_raw  # type: ignore[assignment]
    total_tokens += t_semantic

    # 4. Supplemental Column Infilling
    print("[Stage 4] Checking for missed columns...")
    missing_columns = _find_missing_columns(global_schema, skeleton_code + "\n".join(table_infill.values()))
    if missing_columns:
        print(f"  [Stage 4] Supplementing {len(missing_columns)} missing columns...")
        _supp_raw, t_supp = await infill_semantics(
            skeleton_code=f"# MISSING COLUMNS TO INFILL: {missing_columns}\n# DO NOT RE-GENERATE PREVIOUS COLUMNS.",
            schema_json=schema_json,
            model=model
        )
        supp_infill: Dict[str, str] = _supp_raw  # type: ignore[assignment]
        for t_name, s_code in supp_infill.items():
            if t_name in table_infill:
                table_infill[t_name] += f"\n{s_code}"
            else:
                table_infill[t_name] = s_code
        total_tokens += t_supp

    # 5. Final Assembly (Phased Infill Injection)
    final_code = skeleton_code
    for t_name, infill_code in table_infill.items():
        placeholder = f"# --- SEMANTIC_INFILL_PLACEHOLDER_{t_name} ---"
        if placeholder in final_code:
            final_code = final_code.replace(placeholder, f"# --- SEMANTIC INFILL: {t_name} ---\n{infill_code}")
    final_code = re.sub(r"# --- SEMANTIC_INFILL_PLACEHOLDER_.*? ---\n", "", final_code)

    # 6. Smoke Test (10% scale, subprocess isolation, 1 GB memory cap)
    print("[Stage 4] Running smoke test (scale=0.1)...")
    smoke_ok, _smoke_dfs, smoke_logs = run_smoke_test(final_code, scale_factor=0.1)
    verification_status = "PASSED" if smoke_ok else "FAILED"

    # 7. Coverage Metric
    total_cols = sum(len(t.columns) for t in global_schema.tables)
    missed_count = len(_find_missing_columns(global_schema, final_code))
    coverage = (total_cols - missed_count) / total_cols if total_cols > 0 else 1.0

    return SynthesisResult(
        generated_code=final_code,
        token_usage=total_tokens,
        success=smoke_ok,
        verification_status=verification_status,
        verification_logs=smoke_logs,
        column_coverage=coverage
    ), total_tokens

def _find_missing_columns(schema: Schema, code: str) -> List[str]:
    """Identifies columns in schema not touched by the generated code."""
    missing = []
    for table in schema.tables:
        t_name = table.name
        # Look for assignments like BUFFERS['table']['column'] = ...
        for col in table.columns:
            # Skip specialized PK/FK/Internal if needed, but here we want full coverage
            pattern = rf"BUFFERS\['{re.escape(t_name)}'\]\['{re.escape(col.name)}'\]\s*="
            if not re.search(pattern, code):
                missing.append(f"{t_name}.{col.name}")
    return missing

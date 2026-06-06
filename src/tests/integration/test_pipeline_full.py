"""Integration tests for the full 4-stage pipeline chain.

LIVE -- calls the real OpenAI API. Auto-skipped unless OPENAI_API_KEY is set.
Run with: pytest -m integration src/tests/integration/test_pipeline_full.py -v

These tests run the full NL -> Schema -> Constraints -> Code chain on a
short, self-contained NL description. They are slow and cost money; do not
run them in CI unless explicitly opted in.
"""
from __future__ import annotations

import asyncio

import pytest

from src.orchestration.stage1.entry import orchestrate as stage1
from src.orchestration.stage2.entry import orchestrate as stage2
from src.orchestration.stage3.entry import orchestrate as stage3
from src.orchestration.stage4.entry import orchestrate as stage4
from src.pipeline.stage3.models.manifest import AlgebraicManifest
from src.pipeline.stage4.models import SynthesisResult
from src.util.ablation import AblationConfig

SHORT_NL = (
    "Build a simple employee database. "
    "Each employee has an ID, name, salary (between $30k and $200k), and department. "
    "Departments have a name and a manager employee. "
    "An employee belongs to exactly one department."
)


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_stage1_to_stage4():
    """Smoke test: NL all the way to generated Python code."""
    # Stage 1
    s1_output, s1_tokens = asyncio.run(stage1(SHORT_NL))
    assert s1_tokens > 0
    assert len(s1_output.final_facts) >= 3

    # Stage 2
    s2_output, s2_tokens, registry = asyncio.run(
        stage2(
            s1_output.final_facts,
            domain=s1_output.domain,
            analytical_goal=s1_output.analytical_goal,
        )
    )
    assert s2_tokens > 0
    schema = s2_output.final_global_schema or s2_output.merged_schema
    assert schema is not None
    assert len(schema.tables) >= 1

    # Stage 3
    s3_output, s3_tokens = asyncio.run(
        stage3(
            global_schema=schema,
            registry=registry,
            all_facts=s1_output.final_facts,
        )
    )
    assert s3_tokens > 0
    assert isinstance(s3_output.global_manifest, AlgebraicManifest)

    # Stage 4
    s4_result, s4_tokens = asyncio.run(
        stage4(
            global_schema=schema,
            manifest=s3_output.global_manifest,
            business_facts=s1_output.final_facts,
        )
    )
    assert s4_tokens > 0
    assert isinstance(s4_result, SynthesisResult)
    assert len(s4_result.generated_code) > 100


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_total_tokens_positive():
    s1_out, t1 = asyncio.run(stage1(SHORT_NL))
    schema = (s1_out,)  # just checking stage 1 token here; full chain tested above
    assert t1 > 0


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_no_enrichment_ablation():
    config = AblationConfig.no_enrichment()
    s1_output, _ = asyncio.run(stage1(SHORT_NL, ablation_config=config))
    assert len(s1_output.final_facts) >= 2

    s2_output, _, registry = asyncio.run(
        stage2(
            s1_output.final_facts,
            domain=s1_output.domain,
            analytical_goal=s1_output.analytical_goal,
        )
    )
    schema = s2_output.final_global_schema or s2_output.merged_schema
    assert schema is not None
    assert len(schema.tables) >= 1

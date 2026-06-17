"""Integration tests for Stage 2 (Schema Generation).

LIVE -- calls the real OpenAI API. Auto-skipped unless OPENAI_API_KEY is set.
Run with: pytest -m integration src/tests/integration/test_pipeline_stage2.py -v

These tests verify that:
  - orchestrate() produces a valid Schema with at least 1 table.
  - Token tracking is positive.
  - The output TableFactRegistry is populated.
  - Ablation (no-sharding) still produces a schema.
  - Detected cycles in the output are empty for a simple domain.
"""
from __future__ import annotations

import asyncio

import pytest

from src.orchestration.stage2.entry import orchestrate
from src.orchestration.stage2.models import Output
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Schema
from src.tests.fixtures import sample_data
from src.util.config.ablation import AblationConfig

_FACTS = sample_data.fintech_facts()
_DOMAIN = sample_data.FINTECH_DOMAIN
_GOAL = sample_data.FINTECH_GOAL


@pytest.mark.integration
def test_stage2_orchestrate_returns_output():
    output, tokens, registry = asyncio.run(
        orchestrate(_FACTS, domain=_DOMAIN, analytical_goal=_GOAL)
    )
    assert isinstance(output, Output)
    assert tokens > 0
    assert isinstance(registry, TableFactRegistry)


@pytest.mark.integration
def test_stage2_output_contains_schema_with_tables():
    output, _, _ = asyncio.run(
        orchestrate(_FACTS, domain=_DOMAIN, analytical_goal=_GOAL)
    )
    schema = output.final_global_schema or output.merged_schema
    assert isinstance(schema, Schema)
    assert len(schema.tables) >= 1


@pytest.mark.integration
def test_stage2_output_schema_tables_are_valid():
    output, _, _ = asyncio.run(
        orchestrate(_FACTS, domain=_DOMAIN, analytical_goal=_GOAL)
    )
    schema = output.final_global_schema or output.merged_schema
    assert schema is not None
    for table in schema.tables:
        assert table.name and table.name == table.name.upper()
        assert table.pk and len(table.columns) >= 1


@pytest.mark.integration
def test_stage2_no_cycles_for_simple_domain():
    output, _, _ = asyncio.run(
        orchestrate(_FACTS, domain=_DOMAIN, analytical_goal=_GOAL)
    )
    assert output.cycles == []


@pytest.mark.integration
def test_stage2_registry_populated():
    _, _, registry = asyncio.run(
        orchestrate(_FACTS, domain=_DOMAIN, analytical_goal=_GOAL)
    )
    assert len(registry.table_to_facts) >= 1


@pytest.mark.integration
def test_stage2_ablation_no_sharding_produces_schema():
    config = AblationConfig.no_sharding()
    output, tokens, _ = asyncio.run(
        orchestrate(_FACTS, domain=_DOMAIN, analytical_goal=_GOAL,
                    ablation_config=config)
    )
    assert tokens > 0
    schema = output.final_global_schema or output.merged_schema
    assert schema is not None
    assert len(schema.tables) >= 1

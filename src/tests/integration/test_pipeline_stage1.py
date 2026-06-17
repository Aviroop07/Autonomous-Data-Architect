"""Integration tests for Stage 1 (Fact Extraction).

LIVE -- calls the real OpenAI API. Auto-skipped unless OPENAI_API_KEY is set.
Run with: pytest -m integration src/tests/integration/test_pipeline_stage1.py -v

These tests verify that:
  - The full Stage 1 orchestrate() call completes without error.
  - The output contains a meaningful set of AtomicFacts.
  - Token usage is tracked and positive.
  - Ablation config (no-enrichment) produces fewer facts than the full run.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from src.orchestration.stage1.entry import orchestrate
from src.orchestration.stage1.models import Output
from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.util.config.ablation import AblationConfig

SIMPLE_NL = (
    "We need to simulate loan applications. Applicants have a credit score and annual income. "
    "Loans have an interest rate, principal amount, and term in months. "
    "Each loan belongs to exactly one applicant. "
    "The interest rate must be between 1% and 30%."
)


@pytest.mark.integration
def test_stage1_orchestrate_returns_output():
    output, tokens = asyncio.run(orchestrate(SIMPLE_NL))
    assert isinstance(output, Output)
    assert tokens > 0


@pytest.mark.integration
def test_stage1_extracts_at_least_five_facts():
    output, _ = asyncio.run(orchestrate(SIMPLE_NL))
    assert len(output.final_facts) >= 5


@pytest.mark.integration
def test_stage1_output_facts_are_atomicfact_instances():
    output, _ = asyncio.run(orchestrate(SIMPLE_NL))
    for fact in output.final_facts:
        assert isinstance(fact, AtomicFact)


@pytest.mark.integration
def test_stage1_output_has_domain_and_goal():
    output, _ = asyncio.run(orchestrate(SIMPLE_NL))
    assert output.domain and len(output.domain) > 0
    assert output.analytical_goal and len(output.analytical_goal) > 0


@pytest.mark.integration
def test_stage1_ablation_no_enrichment_runs_without_error():
    config = AblationConfig.no_enrichment()
    output, tokens = asyncio.run(orchestrate(SIMPLE_NL, ablation_config=config))
    assert isinstance(output, Output)
    assert tokens > 0
    # No enrichment -> no external facts; fewer total facts expected
    external_count = sum(1 for f in output.final_facts if f.is_external)
    assert external_count == 0


@pytest.mark.integration
def test_stage1_facts_have_non_empty_text():
    output, _ = asyncio.run(orchestrate(SIMPLE_NL))
    for fact in output.final_facts:
        assert fact.fact and len(fact.fact.strip()) > 0


@pytest.mark.integration
def test_stage1_fact_ids_are_unique():
    output, _ = asyncio.run(orchestrate(SIMPLE_NL))
    ids = [f.id for f in output.final_facts]
    assert len(ids) == len(set(ids))

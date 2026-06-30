from __future__ import annotations

from src.pipeline.stage1.agents.fact_extractor.agent import PROMPT_PATH


def test_fact_extractor_prompt_requires_explicit_relationship_facts():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    assert "Relationship Extraction Rules" in prompt
    assert "Preserve relationship semantics as first-class facts" in prompt
    # Rule 3B requires aggressive explicit relationship extraction (generalized from the
    # earlier brittle literal phrase-list of "routes to"/"is assigned to"/etc.).
    assert "aggressively extract explicit semantic relationships" in prompt
    assert (
        "Do not rely on downstream stages to infer relationships from column names alone"
        in prompt
    )


def test_fact_extractor_prompt_handles_routing_entities():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    assert "For routing or bridge entities" in prompt
    assert "VM instances are associated with tenants" in prompt
    assert "VM instances are assigned to compute nodes" in prompt

"""The fact_extractor prompt must keep its relationship-extraction rules.

These assertions used to pin the literal cloud-tenancy sentences from one live
run -- "VM instances are associated with tenants", "VM instances are assigned
to compute nodes". That made the test the enforcement mechanism for exactly the
brittleness it should have been guarding against: the prompt could not be
de-domained without going red, and one specification's vocabulary was locked
into the suite.

They now assert the RULES are present, in wording general enough to survive a
rephrase but specific enough to fail if a rule is deleted.
"""

from __future__ import annotations

from src.pipeline.stage1.agents.fact_extractor.agent import PROMPT_PATH


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").lower()


def test_relationships_are_extracted_as_standalone_facts():
    prompt = _prompt()
    assert "relationship extraction" in prompt
    assert "first-class facts" in prompt


def test_a_relationship_is_never_left_implicit_in_a_column_name():
    """The rule that motivated the original test: an identifier-shaped
    attribute is not a substitute for stating the link."""
    prompt = _prompt()
    assert "identifier-style attribute" in prompt
    assert "downstream stage" in prompt and "naming alone" in prompt


def test_scoping_phrasing_yields_both_a_relationship_and_a_cardinality_fact():
    prompt = _prompt()
    assert "per" in prompt
    assert "cardinality fact" in prompt


def test_a_clause_with_several_links_is_split_per_link():
    """Replaces the old routing/bridge-entity test, which asserted one
    domain's sentences verbatim."""
    prompt = _prompt()
    assert "one fact per directional link" in prompt


def test_enumeration_members_are_kept_in_one_fact():
    """Caught by a live run. The de-compounding rules ("one fact per idea")
    overrode the enumeration rule and split a value set per member, turning
    one closed claim -- "the permitted values are A and B" -- into two open
    ones. Stage 3 then correctly declined to build an IN-set from either half,
    and the enumerated-value constraint was lost.

    The value set has to be the stated exception to splitting."""
    prompt = _prompt()
    assert "the set is the claim" in prompt
    assert "single fact" in prompt

"""Conventions every agent prompt must satisfy, checked mechanically.

Two rules are enforced here rather than by review, because both were violated
repeatedly and neither is a judgement call:

1. The section structure from CLAUDE.md -- ROLE, TASK, INPUT, GUIDELINES,
   RESTRICTIONS, in that order, with NO hand-written OUTPUT section (that block
   is appended at runtime by get_agent_, and a hand-written one either
   duplicates or contradicts it).

2. No domain-specific examples. Prompts are meant to carry detailed general
   guidelines; a worked example naming a business domain biases the model
   toward that domain and rewards pattern-matching over reasoning. Every prompt
   in this repo had accumulated them, and one live run's vocabulary had been
   copied into four separate prompts.

Rule 2 is necessarily a keyword heuristic. It is deliberately tuned to the
vocabulary that actually appeared -- it cannot catch a novel domain example,
but it does stop the removed ones from creeping back, which is the failure mode
that already happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPTS = sorted(
    Path("src/pipeline").glob("stage[123]/agents/*/prompt.txt"),
)

REQUIRED_SECTIONS = [
    "## ROLE",
    "## TASK",
    "## INPUT",
    "## GUIDELINES",
    "## RESTRICTIONS",
]

# Domain nouns previously used in worked examples. `order` and `book` are
# deliberately absent: both are ordinary English ("in order to", "meaningful
# order") and would make this check unusable through false positives.
BANNED_DOMAIN_TERMS = [
    "platinum",
    "mrn",
    "medical record number",
    "credit score",
    "ohlcv",
    "gics",
    "tenant_id",
    "node_id",
    "region_id",
    "department_id",
    "doctor_name",
    "sick_person",
    "vm instance",
    "compute node",
    "billing ledger",
    "loyalty_tier",
    "shipping_cost",
    "order_item",
    "isbn",
]


def test_at_least_one_prompt_was_discovered():
    """Guards the glob: a silently-empty parametrisation would make every test
    below vacuously pass."""
    assert len(PROMPTS) >= 13


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.parent.name)
class TestStructure:
    def test_has_every_required_section(self, path):
        text = path.read_text(encoding="utf-8")
        missing = [s for s in REQUIRED_SECTIONS if s not in text]
        assert not missing, f"{path.parent.name} is missing {missing}"

    def test_sections_appear_in_the_conventional_order(self, path):
        text = path.read_text(encoding="utf-8")
        positions = [text.index(s) for s in REQUIRED_SECTIONS if s in text]
        assert positions == sorted(positions), (
            f"{path.parent.name} has its sections out of order"
        )

    def test_has_no_handwritten_output_section(self, path):
        text = path.read_text(encoding="utf-8")
        assert "## OUTPUT" not in text, (
            f"{path.parent.name} hand-writes an OUTPUT section; it is appended "
            f"at runtime by get_agent_"
        )

    def test_has_no_unexpected_top_level_sections(self, path):
        text = path.read_text(encoding="utf-8")
        headers = [ln.strip() for ln in text.splitlines() if ln.startswith("## ")]
        unexpected = [h for h in headers if h not in REQUIRED_SECTIONS]
        assert not unexpected, (
            f"{path.parent.name} has extra top-level sections {unexpected}; "
            f"put the content under GUIDELINES or RESTRICTIONS"
        )


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.parent.name)
def test_contains_no_domain_specific_example_vocabulary(path):
    text = path.read_text(encoding="utf-8").lower()
    found = [term for term in BANNED_DOMAIN_TERMS if term in text]
    assert not found, (
        f"{path.parent.name} reintroduces domain-specific example vocabulary "
        f"{found}. State the rule generally, or use a <PLACEHOLDER> name."
    )

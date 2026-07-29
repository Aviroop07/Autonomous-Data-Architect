"""Information-capacity evaluation: can the schema hold what the spec states?

The classical criterion for comparing two schemas is relative information
capacity -- whether they can represent the same set of database states -- not
whether they chose the same identifiers. This module measures the practical
proxy: every fact the specification states should have a home in the schema, and
every part of the schema should trace to some fact.

Why this replaces table and attribute F1 rather than supplementing them:

  - It is immune to naming. Nothing here reads a table or column name.
  - It is immune to normalisation. A junction decomposed two different ways
    carries the same facts either way. A live hospital pair produced 12 tables in
    one run and 11 in another with IDENTICAL 100% coverage; name-set F1 called
    that a regression, which it was not.
  - It is not all-or-nothing. `Table Acc` was 1.0 only on a perfect set match,
    which is why it read 0.000 while F1 read 0.75.

Recall needs no ground-truth schema at all -- it compares the predicted schema
against the FACTS, which is what the schema is supposed to represent. That makes
it usable on inputs with no authored ground truth, including specs written by a
user, where every name-based metric is inapplicable by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

from src.util.schema_model.schema import Schema

# Facts that carry modelling obligations. A fact tagged only as commentary places
# no requirement on the schema, so counting it against recall would penalise a
# correct schema for ignoring something it should ignore.
LOAD_BEARING_TAGS = ("STRUCTURAL", "LOGICAL", "STATISTICAL")


@dataclass
class CapacityResult:
    ic_recall: float
    ic_precision: float
    uncovered_fact_ids: List[int] = field(default_factory=list)
    unsupported_elements: List[str] = field(default_factory=list)
    n_required_facts: int = 0
    n_elements: int = 0

    @property
    def ic_f1(self) -> float:
        p, r = self.ic_precision, self.ic_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def as_dict(self) -> Dict[str, float]:
        # n_required_facts travels WITH the score, for the same reason KDC
        # reports n_checked: with no facts to check, recall is vacuously 1.0, and
        # a vacuous 1.0 must never be mistakable for an earned one. The harness
        # can pass facts=None whenever Stage 1's output is missing.
        return {
            "ic_f1": self.ic_f1,
            "ic_recall": self.ic_recall,
            "ic_precision": self.ic_precision,
            "ic_n_required_facts": float(self.n_required_facts),
            "ic_n_elements": float(self.n_elements),
        }


def _fact_id(fact: object) -> int | None:
    fid = getattr(fact, "id", None)
    if fid is None and isinstance(fact, dict):
        fid = fact.get("id")
    return fid if isinstance(fid, int) else None


def _fact_tags(fact: object) -> Sequence[str]:
    tags = getattr(fact, "tags", None)
    if tags is None and isinstance(fact, dict):
        tags = fact.get("tags")
    out: List[str] = []
    for t in tags or []:
        # Tags may be a str enum; compare on the value either way.
        out.append(getattr(t, "value", None) or str(t))
    return out


def required_fact_ids(facts: Iterable[object]) -> Set[int]:
    """Fact IDs the schema is obliged to represent."""
    required: Set[int] = set()
    for fact in facts:
        fid = _fact_id(fact)
        if fid is None:
            continue
        tags = _fact_tags(fact)
        if not tags or any(t in LOAD_BEARING_TAGS for t in tags):
            # An untagged fact is treated as load-bearing: silently exempting it
            # would let a tagging failure inflate the score.
            required.add(fid)
    return required


def evaluate_capacity(schema: Schema, facts: Sequence[object]) -> CapacityResult:
    """Score how much of the specification the schema can hold, and vice versa.

    An "element" is a table, a column or a foreign key -- the units the mapper
    stamps provenance onto as it builds them.
    """
    required = required_fact_ids(facts)

    cited: Set[int] = set()
    elements = 0
    unsupported: List[str] = []

    for table in schema.tables:
        elements += 1
        t_ids = set(table.source_fact_ids or [])
        cited |= t_ids
        table_supported = bool(t_ids)

        for col in table.columns:
            elements += 1
            c_ids = set(col.source_fact_ids or [])
            cited |= c_ids
            if not c_ids:
                # A synthesized surrogate key legitimately traces to no fact --
                # the mapper invents it -- so a column that IS the primary key of
                # a table that is otherwise supported is not a hallucination.
                is_surrogate_pk = col.name in (table.primary_key or [])
                if not (is_surrogate_pk and table_supported):
                    unsupported.append(f"{table.name}.{col.name}")
        if not table_supported:
            unsupported.append(table.name)

    for fk in schema.relationships or []:
        elements += 1
        f_ids = set(fk.source_fact_ids or [])
        cited |= f_ids
        if not f_ids:
            unsupported.append(
                f"FK {fk.referencing_table}.{fk.referencing_column}"
                f" -> {fk.referred_table}"
            )

    covered = required & cited
    recall = len(covered) / len(required) if required else 1.0
    precision = (elements - len(unsupported)) / elements if elements else 0.0

    return CapacityResult(
        ic_recall=recall,
        ic_precision=precision,
        uncovered_fact_ids=sorted(required - cited),
        unsupported_elements=sorted(unsupported),
        n_required_facts=len(required),
        n_elements=elements,
    )

"""Information capacity must be blind to names and to normalisation.

These are the two properties that made the name-based metrics unsound: renaming
a table changed the score, and decomposing a junction differently read as a
regression even when the schema carried exactly the same facts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.evaluation.schema_level.capacity_eval import (
    evaluate_capacity,
    required_fact_ids,
)
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table


def _facts(n: int, tags: Optional[List[FactTag]] = None) -> List[AtomicFact]:
    out = []
    for i in range(1, n + 1):
        kwargs = {"id": i, "fact": f"fact {i}"}
        if tags is not None:
            kwargs["tags"] = list(tags)
        out.append(AtomicFact(**kwargs))
    return out


def _col(name: str, facts: Sequence[int], dt: DataType = DataType.VARCHAR) -> Column:
    return Column(name=name, data_type=dt, source_fact_ids=list(facts))


def _table(name: str, cols: Sequence[Column], facts: Sequence[int], pk: str) -> Table:
    return Table(
        name=name, columns=list(cols), primary_key=[pk], source_fact_ids=list(facts)
    )


def test_full_coverage_scores_one() -> None:
    schema = Schema(
        tables=[
            _table(
                "CUSTOMER",
                [_col("customer_id", [1], DataType.INTEGER), _col("full_name", [2])],
                [1],
                "customer_id",
            )
        ],
        relationships=[],
    )
    r = evaluate_capacity(schema, _facts(2))
    assert r.ic_recall == 1.0
    assert r.ic_precision == 1.0
    assert r.ic_f1 == 1.0


def test_a_lost_fact_costs_recall() -> None:
    schema = Schema(
        tables=[
            _table(
                "CUSTOMER",
                [_col("customer_id", [1], DataType.INTEGER)],
                [1],
                "customer_id",
            )
        ],
        relationships=[],
    )
    r = evaluate_capacity(schema, _facts(4))
    assert r.ic_recall == 0.25
    assert r.uncovered_fact_ids == [2, 3, 4]


def test_renaming_everything_changes_nothing() -> None:
    """The property no name-based metric can have."""
    a = Schema(
        tables=[
            _table(
                "CUSTOMER",
                [_col("customer_id", [1], DataType.INTEGER), _col("nm", [2])],
                [1],
                "customer_id",
            )
        ],
        relationships=[],
    )
    b = Schema(
        tables=[
            _table(
                "PATRON",
                [_col("patron_key", [1], DataType.INTEGER), _col("moniker", [2])],
                [1],
                "patron_key",
            )
        ],
        relationships=[],
    )
    facts = _facts(2)
    assert (
        evaluate_capacity(a, facts).as_dict() == evaluate_capacity(b, facts).as_dict()
    )


def test_two_normalisations_of_the_same_facts_score_the_same() -> None:
    """The hospital 12-vs-11-table case, in miniature.

    One schema keeps a junction, the other folds the same facts into a parent.
    Both represent facts 1-4, so both must score identically -- name-set F1 read
    this shape as a regression.
    """
    with_junction = Schema(
        tables=[
            _table("ORDER", [_col("order_id", [1], DataType.INTEGER)], [1], "order_id"),
            _table("ITEM", [_col("item_id", [2], DataType.INTEGER)], [2], "item_id"),
            _table(
                "ORDER_ITEM",
                [
                    _col("order_id", [3], DataType.INTEGER),
                    _col("item_id", [3], DataType.INTEGER),
                    _col("qty", [4], DataType.INTEGER),
                ],
                [3],
                "order_id",
            ),
        ],
        relationships=[],
    )
    folded = Schema(
        tables=[
            _table("ORDER", [_col("order_id", [1], DataType.INTEGER)], [1], "order_id"),
            _table(
                "ITEM",
                [
                    _col("item_id", [2], DataType.INTEGER),
                    _col("order_id", [3], DataType.INTEGER),
                    _col("qty", [4], DataType.INTEGER),
                ],
                [2],
                "item_id",
            ),
        ],
        relationships=[],
    )
    facts = _facts(4)
    assert evaluate_capacity(with_junction, facts).ic_recall == 1.0
    assert evaluate_capacity(folded, facts).ic_recall == 1.0


def test_an_invented_table_costs_precision() -> None:
    schema = Schema(
        tables=[
            _table(
                "CUSTOMER",
                [_col("customer_id", [1], DataType.INTEGER)],
                [1],
                "customer_id",
            ),
            _table(
                "AUDIT_LOG", [_col("audit_id", [], DataType.INTEGER)], [], "audit_id"
            ),
        ],
        relationships=[],
    )
    r = evaluate_capacity(schema, _facts(1))
    assert r.ic_recall == 1.0
    assert r.ic_precision < 1.0
    assert "AUDIT_LOG" in r.unsupported_elements


def test_a_synthesized_surrogate_key_is_not_a_hallucination() -> None:
    """The mapper invents surrogate keys, so they trace to no fact by design."""
    schema = Schema(
        tables=[
            _table(
                "CUSTOMER",
                [_col("customer_id", [], DataType.INTEGER), _col("full_name", [1])],
                [1],
                "customer_id",
            )
        ],
        relationships=[],
    )
    r = evaluate_capacity(schema, _facts(1))
    assert r.ic_precision == 1.0, r.unsupported_elements


def test_an_unsupported_foreign_key_costs_precision() -> None:
    schema = Schema(
        tables=[
            _table("A", [_col("a_id", [1], DataType.INTEGER)], [1], "a_id"),
            _table(
                "B",
                [
                    _col("b_id", [2], DataType.INTEGER),
                    _col("a_id", [2], DataType.INTEGER),
                ],
                [2],
                "b_id",
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="B", referencing_column="a_id", referred_table="A"
            )
        ],
    )
    r = evaluate_capacity(schema, _facts(2))
    assert any("FK B.a_id" in e for e in r.unsupported_elements)
    assert r.ic_precision < 1.0


def test_only_load_bearing_facts_are_required() -> None:
    """A fact with no modelling obligation must not count against recall."""
    structural = _facts(1, tags=[FactTag.STRUCTURAL])
    commentary = [AtomicFact(id=2, fact="fact 2", tags=[FactTag.METADATA])]
    req = required_fact_ids(structural + commentary)
    assert req == {1}


def test_untagged_facts_are_treated_as_required() -> None:
    """A tagging failure must not silently inflate the score."""
    assert required_fact_ids(_facts(3)) == {1, 2, 3}


def test_no_facts_means_recall_is_vacuously_one() -> None:
    schema = Schema(
        tables=[_table("A", [_col("a_id", [], DataType.INTEGER)], [], "a_id")],
        relationships=[],
    )
    assert evaluate_capacity(schema, []).ic_recall == 1.0

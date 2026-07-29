"""Structural schema evaluation must be blind to names and to renaming.

The name-based metrics it sits beside cannot be: every one of them is keyed on
matching names, propped up by a fuzzy matcher whose similarity ranges for
synonyms (0.235-0.839) and for unrelated words (0.216-0.608) overlap so heavily
that no threshold separates them. These tests pin the properties that make the
structural metric a different kind of measurement rather than a re-tuning of the
same one.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.evaluation.schema_level.structural_eval import (
    align_tables,
    evaluate_structural,
    structural_similarity,
    _signatures,
)
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table


def _t(
    name: str,
    cols: Sequence[tuple[str, DataType]],
    pk: Optional[List[str]] = None,
) -> Table:
    return Table(
        name=name,
        columns=[Column(name=c, data_type=d) for c, d in cols],
        primary_key=list(pk) if pk else [cols[0][0]],
    )


def _fk(rt: str, rc: str, dt: str, _dc: str = "") -> ForeignKey:
    # ForeignKey carries no referred_column -- it always targets the referred
    # table's primary key. The unused arg keeps call sites readable.
    return ForeignKey(referencing_table=rt, referencing_column=rc, referred_table=dt)


def _customer_order_schema(customer: str = "CUSTOMER", order: str = "ORDER") -> Schema:
    """Two tables, one FK: order -> customer."""
    return Schema(
        tables=[
            _t(
                customer,
                [("customer_id", DataType.INTEGER), ("full_name", DataType.VARCHAR)],
            ),
            _t(
                order,
                [
                    ("order_id", DataType.INTEGER),
                    ("customer_id", DataType.INTEGER),
                    ("placed_at", DataType.DATE),
                    ("total", DataType.DECIMAL),
                ],
            ),
        ],
        relationships=[_fk(order, "customer_id", customer, "customer_id")],
    )


def test_identical_schemas_score_perfectly() -> None:
    s = _customer_order_schema()
    r = evaluate_structural(s, s)
    assert r.fk_topology_f1 == 1.0
    assert r.table_structural_recall == 1.0
    assert r.column_type_agreement == 1.0
    assert r.unaligned_gt == [] and r.unaligned_pred == []


def test_renaming_every_table_changes_nothing() -> None:
    """The defining property. A name-based metric would score this near zero."""
    gt = _customer_order_schema("CUSTOMER", "ORDER")
    pred = _customer_order_schema("PATRON", "PURCHASE")

    r = evaluate_structural(pred, gt)
    assert r.fk_topology_f1 == 1.0, r.as_dict()
    assert r.table_structural_recall == 1.0
    # And the alignment found the right correspondence despite the names.
    mapping, _pairs = align_tables(pred, gt)
    assert mapping["CUSTOMER"] == "PATRON"
    assert mapping["ORDER"] == "PURCHASE"


def test_a_reversed_foreign_key_is_penalised() -> None:
    """Reversing the only FK makes the schema its own MIRROR IMAGE.

    The optimal alignment therefore swaps the two tables and topology F1 alone
    reports a perfect 1.0 -- correctly, since the graphs are isomorphic. What
    exposes the swap is that the paired tables do not resemble each other, so
    the headline structural_score must fall even though topology does not.
    """
    gt = _customer_order_schema()
    pred = _customer_order_schema()
    pred.relationships = [_fk("CUSTOMER", "customer_id", "ORDER", "order_id")]

    r = evaluate_structural(pred, gt)
    assert r.structural_score < 0.5, r.as_dict()
    assert r.column_type_agreement < 0.5, "the swap should show up here"


def test_the_headline_score_is_perfect_only_for_a_true_match() -> None:
    gt = _customer_order_schema()
    assert evaluate_structural(gt, gt).structural_score == 1.0
    # Renaming is still free.
    assert (
        evaluate_structural(
            _customer_order_schema("PATRON", "PURCHASE"), gt
        ).structural_score
        == 1.0
    )


def test_a_missing_foreign_key_costs_recall_not_precision() -> None:
    gt = _customer_order_schema()
    pred = _customer_order_schema()
    pred.relationships = []

    r = evaluate_structural(pred, gt)
    assert r.fk_topology_recall == 0.0
    # No predicted edges at all, so there is nothing wrong to have predicted.
    assert r.fk_topology_f1 == 0.0


def test_an_invented_table_shows_up_as_unaligned_predicted() -> None:
    gt = _customer_order_schema()
    pred = _customer_order_schema()
    pred.tables.append(
        _t("SPURIOUS", [("spurious_id", DataType.INTEGER), ("note", DataType.TEXT)])
    )

    r = evaluate_structural(pred, gt)
    # Ground truth is still fully recovered ...
    assert r.table_structural_recall == 1.0
    # ... but the extra table is reported rather than ignored.
    assert "SPURIOUS" in r.unaligned_pred


def test_a_missing_table_costs_structural_recall() -> None:
    """Recall is SOFT, so the surviving table contributes its own similarity
    rather than a flat 1. Dropping ORDER also strips the foreign key, so even
    CUSTOMER no longer matches perfectly -- hence a figure below the 0.5 a
    count-based recall would have given, which is the more honest reading."""
    gt = _customer_order_schema()
    pred = Schema(tables=[gt.tables[0]], relationships=[])

    r = evaluate_structural(pred, gt)
    assert 0.0 < r.table_structural_recall < 0.5, r.as_dict()
    assert "ORDER" in r.unaligned_gt


def test_alignment_is_deterministic_and_order_independent() -> None:
    """Optimal assignment, not a greedy pass over whatever order dicts yield."""
    gt = _customer_order_schema()
    pred = _customer_order_schema("PATRON", "PURCHASE")
    reversed_pred = Schema(
        tables=list(reversed(pred.tables)), relationships=list(pred.relationships or [])
    )

    a, _ = align_tables(pred, gt)
    b, _ = align_tables(reversed_pred, gt)
    assert a == b


def test_structural_similarity_separates_differently_shaped_tables() -> None:
    """A hub table and a leaf lookup table must not look alike."""
    schema = Schema(
        tables=[
            _t(
                "HUB",
                [
                    ("hub_id", DataType.INTEGER),
                    ("a", DataType.INTEGER),
                    ("b", DataType.INTEGER),
                    ("c", DataType.INTEGER),
                ],
            ),
            _t("LEAF", [("leaf_id", DataType.INTEGER), ("label", DataType.VARCHAR)]),
        ],
        relationships=[
            _fk("HUB", "a", "LEAF", "leaf_id"),
            _fk("HUB", "b", "LEAF", "leaf_id"),
        ],
    )
    sigs = _signatures(schema)
    assert sigs["HUB"].out_degree == 2
    assert sigs["LEAF"].in_degree == 2
    assert structural_similarity(sigs["HUB"], sigs["LEAF"]) < 0.75


def test_composite_primary_key_is_part_of_the_signature() -> None:
    """A junction table is structurally distinct from an entity table."""
    junction = _t(
        "ENROLMENT",
        [("student_id", DataType.INTEGER), ("course_id", DataType.INTEGER)],
        pk=["student_id", "course_id"],
    )
    entity = _t(
        "STUDENT",
        [("student_id", DataType.INTEGER), ("name", DataType.VARCHAR)],
        pk=["student_id"],
    )
    schema = Schema(tables=[junction, entity], relationships=[])
    sigs = _signatures(schema)
    assert sigs["ENROLMENT"].pk_arity == 2
    assert sigs["STUDENT"].pk_arity == 1
    assert structural_similarity(sigs["ENROLMENT"], sigs["STUDENT"]) < 1.0


def test_empty_schemas_do_not_raise() -> None:
    empty = Schema(tables=[], relationships=[])
    r = evaluate_structural(empty, empty)
    assert r.table_structural_recall == 0.0
    assert r.as_dict()["fk_topology_f1"] in (0.0, 1.0)


def test_reversal_is_best_effort_and_never_loses_the_defect() -> None:
    """What is actually true, established by measurement rather than hoped for.

    Reversing a foreign key changes the in/out degrees of BOTH endpoints -- and
    those degrees are exactly what the name-blind alignment uses. So the act of
    reversing tends to move the alignment that would be needed to recognise it.
    Four different schema shapes were tried (two tables, a three-table chain, a
    hub with four spokes, and a chain with distinctive column profiles) and in
    every one the alignment re-mapped, so the reversal surfaced as a missing edge
    plus a spurious one rather than as a reversal.

    That is an inherent tension in a metric that refuses to read names, not a
    coding fault, and `reversed_fks` is therefore BEST EFFORT: it fires when the
    alignment happens to hold, or when a missing edge's exact opposite was
    predicted. What must always hold is the weaker, more important property
    asserted here -- the defect is reported under SOME heading and never silently
    disappears.
    """
    def chain(reverse_last: bool) -> Schema:
        rels = [_fk("A", "b_id", "B"), _fk("B", "c_id", "C")]
        if reverse_last:
            rels[1] = _fk("C", "b_id", "B")
        return Schema(
            tables=[
                _t("A", [("a_id", DataType.INTEGER), ("b_id", DataType.INTEGER),
                         ("name", DataType.VARCHAR)]),
                _t("B", [("b_id", DataType.INTEGER), ("c_id", DataType.INTEGER),
                         ("when", DataType.DATE)]),
                _t("C", [("c_id", DataType.INTEGER), ("b_id", DataType.INTEGER),
                         ("amount", DataType.DECIMAL)]),
            ],
            relationships=rels,
        )

    r = evaluate_structural(chain(reverse_last=True), chain(reverse_last=False))
    counted = len(r.reversed_fks) + len(r.missing_fks) + len(r.spurious_fks)
    assert counted > 0, f"the reversal vanished entirely: {r.as_dict()}"
    # And it costs the headline score, whichever heading it landed under.
    assert r.structural_score < 1.0, r.as_dict()


def test_the_three_fk_counts_are_complete_even_when_attribution_degrades() -> None:
    """The documented limit: with structurally identical tables the alignment may
    permute them, so a reversal can stay split across missing and spurious. What
    must never happen is a defect vanishing entirely."""
    def hub(reverse_one: bool) -> Schema:
        rels = [_fk("HUB", f"leaf{i}_id", f"LEAF{i}") for i in range(1, 5)]
        if reverse_one:
            rels[0] = _fk("LEAF1", "hub_id", "HUB")
        return Schema(
            tables=[
                _t("HUB", [("hub_id", DataType.INTEGER)]
                   + [(f"leaf{i}_id", DataType.INTEGER) for i in range(1, 5)]),
                *[_t(f"LEAF{i}", [(f"leaf{i}_id", DataType.INTEGER),
                                  ("hub_id", DataType.INTEGER),
                                  ("label", DataType.VARCHAR)]) for i in range(1, 5)],
            ],
            relationships=rels,
        )

    r = evaluate_structural(hub(True), hub(False))
    assert len(r.reversed_fks) + len(r.missing_fks) + len(r.spurious_fks) > 0, (
        "the defect must be reported under SOME heading, never dropped"
    )


def test_a_genuinely_absent_fk_is_reported_as_missing() -> None:
    gt = _customer_order_schema()
    pred = _customer_order_schema()
    pred.relationships = []

    r = evaluate_structural(pred, gt)
    assert r.missing_fks, r.as_dict()
    assert not r.reversed_fks
    assert not r.spurious_fks


def test_an_invented_fk_is_reported_as_spurious() -> None:
    gt = _customer_order_schema()
    pred = _customer_order_schema()
    pred.tables.append(
        _t("PROMO", [("promo_id", DataType.INTEGER), ("code", DataType.VARCHAR)])
    )
    pred.relationships = list(pred.relationships or []) + [
        _fk("PROMO", "promo_id", "CUSTOMER")
    ]

    r = evaluate_structural(pred, gt)
    assert r.spurious_fks, r.as_dict()


def test_a_perfect_schema_reports_no_fk_defects_of_any_kind() -> None:
    s = _customer_order_schema()
    r = evaluate_structural(s, s)
    assert (r.reversed_fks, r.missing_fks, r.spurious_fks) == ([], [], [])

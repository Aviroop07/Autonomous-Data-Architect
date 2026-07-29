"""The cardinality contract that decides where foreign keys land.

`relational_mapper.py`'s 1:N branch reads `cardinality_max` alone to choose
which side receives the synthesized foreign key. Until these checks existed the
invariant was asserted only in the er_extractor and er_auditor prompts, so a
violation did not fail -- it produced a structurally valid schema with a
REVERSED foreign key, and nothing downstream could detect it, because a
reversed edge is still a legal edge.

The convention the mapper implements is LOOK-ACROSS: `cardinality_max=1` marks
the side of which at most one instance is associated with each instance of the
other participant. The competing ER convention, LOOK-HERE (Chen's participation
counts), produces the identical JSON shape with the opposite meaning -- two
saved runs of the same specification disagreed exactly this way, one emitting
`Customer(0, null) / Order(1, 1)` and the other `Customer(1, 1) /
Order(0, null)` for the same relationship over the same source facts.
"""

from __future__ import annotations

from typing import Literal

from src.pipeline.stage2.mapper.conceptual_model import (
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
)


def _model(rel: Relationship, *names: str) -> ConceptualModel:
    return ConceptualModel(
        entities=[Entity(name=n, attributes=[]) for n in names],
        relationships=[rel],
    )


def _rel(
    name: str,
    kind: Literal["1:1", "1:N", "M:N"],
    a: tuple[str, int | None, int | None],
    b: tuple[str, int | None, int | None],
    degree: Literal["binary", "n-ary"] = "binary",
) -> Relationship:
    return Relationship(
        name=name,
        kind=kind,
        degree=degree,
        participants=[
            Participant(entity=a[0], cardinality_min=a[1], cardinality_max=a[2]),
            Participant(entity=b[0], cardinality_min=b[1], cardinality_max=b[2]),
        ],
    )


def test_a_well_formed_one_to_many_is_accepted() -> None:
    m = _model(
        _rel("placement", "1:N", ("Customer", 1, 1), ("Order", 0, None)),
        "Customer",
        "Order",
    )
    assert m.get_errors() == []


def test_the_population_count_that_leaked_into_a_real_run_is_rejected() -> None:
    """The observed defect, verbatim: a saved conceptual model carried
    `Warehouse(min=50, max=5000)` -- "each warehouse handles 50-5000 orders",
    which is a population estimate under the LOOK-HERE reading, in a field the
    mapper interprets structurally under LOOK-ACROSS."""
    m = _model(
        _rel("fulfillment", "1:N", ("Order", 1, 1), ("Warehouse", 50, 5000)),
        "Order",
        "Warehouse",
    )
    errors = m.get_errors()
    assert errors, "the out-of-domain pair must be reported"
    joined = " ".join(errors)
    assert "cardinality_max=5000" in joined
    assert "cardinality_min=50" in joined
    # And the message must say what the field is FOR, not merely that it is wrong.
    assert "STRUCTURAL" in joined


def test_a_one_to_many_with_no_one_side_is_rejected() -> None:
    """The live hazard. `p1.cardinality_max != 1` is true for BOTH participants,
    so the mapper makes p1 the child and the foreign key lands wherever
    participant order happens to put it -- an arbitrary direction."""
    m = _model(
        _rel("placement", "1:N", ("Customer", 0, None), ("Order", 0, None)),
        "Customer",
        "Order",
    )
    errors = m.get_errors()
    assert errors
    assert "listed first" in " ".join(errors), (
        "the feedback must explain the consequence, since the model has to fix it"
    )


def test_a_one_to_many_with_two_one_sides_is_rejected_as_really_one_to_one() -> None:
    m = _model(
        _rel("payment", "1:N", ("Order", 1, 1), ("Payment", 1, 1)),
        "Order",
        "Payment",
    )
    errors = m.get_errors()
    assert errors
    assert "1:1" in " ".join(errors)


def test_one_to_one_must_have_both_sides_single() -> None:
    ok = _model(
        _rel("payment", "1:1", ("Order", 1, 1), ("Payment", 1, 1)), "Order", "Payment"
    )
    assert ok.get_errors() == []

    bad = _model(
        _rel("payment", "1:1", ("Order", 1, 1), ("Payment", 0, None)),
        "Order",
        "Payment",
    )
    assert bad.get_errors()


def test_many_to_many_must_have_neither_side_single() -> None:
    ok = _model(
        _rel("order_item", "M:N", ("Order", 1, None), ("Product", 0, None)),
        "Order",
        "Product",
    )
    assert ok.get_errors() == []

    bad = _model(
        _rel("order_item", "M:N", ("Order", 1, 1), ("Product", 0, None)),
        "Order",
        "Product",
    )
    errors = bad.get_errors()
    assert errors
    assert "Order" in " ".join(errors), "the offending side must be named"


def test_null_cardinalities_are_still_allowed_everywhere() -> None:
    """Unspecified cardinality is legitimate -- a spec may simply not state it,
    and the mapper has a documented safe default. Only 1:N is shape-checked,
    because only 1:N uses the value to choose a direction."""
    m = _model(
        _rel("association", "M:N", ("A", None, None), ("B", None, None)), "A", "B"
    )
    assert m.get_errors() == []


def test_n_ary_relationships_are_not_shape_checked() -> None:
    """An n-ary relationship always becomes a junction table, so no side
    receives a foreign key by cardinality and there is no direction to get
    wrong. The value domain still applies."""
    rel = Relationship(
        name="review",
        kind="M:N",
        degree="n-ary",
        participants=[
            Participant(entity=e, cardinality_min=0, cardinality_max=None)
            for e in ("Customer", "Product", "Order")
        ],
    )
    m = _model(rel, "Customer", "Product", "Order")
    assert m.get_errors() == []


def test_the_domain_check_still_applies_to_n_ary_participants() -> None:
    rel = Relationship(
        name="review",
        kind="M:N",
        degree="n-ary",
        participants=[
            Participant(entity="Customer", cardinality_min=0, cardinality_max=None),
            Participant(entity="Product", cardinality_min=0, cardinality_max=900),
            Participant(entity="Order", cardinality_min=0, cardinality_max=None),
        ],
    )
    m = _model(rel, "Customer", "Product", "Order")
    assert m.get_errors()


def test_both_conventions_cannot_both_validate_the_same_relationship() -> None:
    """The two runs that disagreed both PASS the shape check -- which is the
    honest limit of a deterministic guard here. Inverting a 1:N pair yields
    another well-formed 1:N pair, so no validator can choose between them; only
    the prompt's stated convention can. This test pins that limit so it is not
    mistaken for coverage the guard does not have."""
    look_across = _model(
        _rel("placement", "1:N", ("Customer", 1, 1), ("Order", 0, None)),
        "Customer",
        "Order",
    )
    look_here = _model(
        _rel("placement", "1:N", ("Customer", 0, None), ("Order", 1, 1)),
        "Customer",
        "Order",
    )
    assert look_across.get_errors() == []
    assert look_here.get_errors() == []

"""Every metric, computed over one fabricated case, with the numbers asserted.

The per-metric test files check each metric in isolation. This one exists for a
different reason: to run the evaluation code the way the harness runs it -- a
predicted schema, a ground-truth schema, facts, and ground-truth functional
dependencies, all at once -- so the whole path is exercised and the actual figures
are visible in one place rather than inferred from unit tests of the parts.

The fabricated case is deliberately imperfect. A predicted schema that matched
perfectly would make every metric 1.0 and prove only that nothing crashed. This
one gets some things right and some wrong, in specific ways, so each number has a
reason:

  ground truth                      prediction
  ------------------------------    ---------------------------------------
  CUSTOMER(customer_id, email,      SHOPPER(shopper_id, email_address,
           city)                            city)          <- renamed only
  ORDER(order_id, customer_id,      PURCHASE(purchase_id, shopper_id,
        placed_at, total)                    placed_at, total)
  ORDER_LINE(order_id, line_no,     PURCHASE_LINE(purchase_id, line_no,
             sku, qty)                            sku, qty)
  SUPPLIER(supplier_id, name)       -- MISSING
  --                                PROMO_BANNER(banner_id, image_url)
                                       <- INVENTED, no fact supports it

So: renaming is free, one real table is missing, one table is hallucinated, and
the FK topology is right for what survived.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pytest

from src.evaluation.schema_level.capacity_eval import evaluate_capacity
from src.evaluation.schema_level.kdc_eval import evaluate_kdc
from src.evaluation.schema_level.structural_eval import evaluate_structural
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table

# --------------------------------------------------------------------------
# the fabricated case
# --------------------------------------------------------------------------

# 1 customer, 2 email, 3 city, 4 order, 5 placed_at, 6 total,
# 7 order line, 8 sku, 9 qty, 10 supplier
FACTS: List[AtomicFact] = [
    AtomicFact(id=i, fact=f"fact {i}", tags=[FactTag.STRUCTURAL]) for i in range(1, 11)
]


def _col(name: str, dt: DataType, facts: Sequence[int] = ()) -> Column:
    return Column(name=name, data_type=dt, source_fact_ids=list(facts))


def _table(
    name: str, cols: Sequence[Column], pk: Sequence[str], facts: Sequence[int] = ()
) -> Table:
    return Table(
        name=name, columns=list(cols), primary_key=list(pk), source_fact_ids=list(facts)
    )


def _fk(rt: str, rc: str, dt: str, facts: Sequence[int] = ()) -> ForeignKey:
    return ForeignKey(
        referencing_table=rt,
        referencing_column=rc,
        referred_table=dt,
        source_fact_ids=list(facts),
    )


def ground_truth() -> Schema:
    return Schema(
        tables=[
            _table(
                "CUSTOMER",
                [
                    _col("customer_id", DataType.INTEGER),
                    _col("email", DataType.VARCHAR),
                    _col("city", DataType.VARCHAR),
                ],
                ["customer_id"],
            ),
            _table(
                "ORDER",
                [
                    _col("order_id", DataType.INTEGER),
                    _col("customer_id", DataType.INTEGER),
                    _col("placed_at", DataType.DATE),
                    _col("total", DataType.DECIMAL),
                ],
                ["order_id"],
            ),
            _table(
                "ORDER_LINE",
                [
                    _col("order_id", DataType.INTEGER),
                    _col("line_no", DataType.INTEGER),
                    _col("sku", DataType.VARCHAR),
                    _col("qty", DataType.INTEGER),
                ],
                ["order_id", "line_no"],
            ),
            _table(
                "SUPPLIER",
                [
                    _col("supplier_id", DataType.INTEGER),
                    _col("name", DataType.VARCHAR),
                ],
                ["supplier_id"],
            ),
        ],
        relationships=[
            _fk("ORDER", "customer_id", "CUSTOMER"),
            _fk("ORDER_LINE", "order_id", "ORDER"),
        ],
    )


def prediction() -> Schema:
    """Renamed throughout, SUPPLIER missing, PROMO_BANNER invented."""
    return Schema(
        tables=[
            _table(
                "SHOPPER",
                [
                    _col("shopper_id", DataType.INTEGER),
                    _col("email_address", DataType.VARCHAR, [2]),
                    _col("city", DataType.VARCHAR, [3]),
                ],
                ["shopper_id"],
                [1],
            ),
            _table(
                "PURCHASE",
                [
                    _col("purchase_id", DataType.INTEGER),
                    _col("shopper_id", DataType.INTEGER, [4]),
                    _col("placed_at", DataType.DATE, [5]),
                    _col("total", DataType.DECIMAL, [6]),
                ],
                ["purchase_id"],
                [4],
            ),
            _table(
                "PURCHASE_LINE",
                [
                    _col("purchase_id", DataType.INTEGER, [7]),
                    _col("line_no", DataType.INTEGER, [7]),
                    _col("sku", DataType.VARCHAR, [8]),
                    _col("qty", DataType.INTEGER, [9]),
                ],
                ["purchase_id", "line_no"],
                [7],
            ),
            # Nothing in the specification asks for this.
            _table(
                "PROMO_BANNER",
                [
                    _col("banner_id", DataType.INTEGER),
                    _col("image_url", DataType.TEXT),
                ],
                ["banner_id"],
            ),
        ],
        relationships=[
            _fk("PURCHASE", "shopper_id", "SHOPPER", [4]),
            _fk("PURCHASE_LINE", "purchase_id", "PURCHASE", [7]),
        ],
    )


# Authored from the specification, NOT taken from the pipeline -- which is the
# whole point: dependencies the pipeline derived itself cannot score it.
GT_FDS: List[Dict[str, List[str]]] = [
    {
        "determinant": ["PURCHASE_LINE.purchase_id", "PURCHASE_LINE.line_no"],
        "dependent": ["PURCHASE_LINE.qty"],
    },
    {"determinant": ["SHOPPER.shopper_id"], "dependent": ["SHOPPER.email_address"]},
]


# --------------------------------------------------------------------------
# the whole suite over that one case
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scores() -> Dict[str, float]:
    pred, gt = prediction(), ground_truth()
    out: Dict[str, float] = {}
    out.update(evaluate_structural(pred, gt).as_dict())
    out.update(evaluate_capacity(pred, FACTS).as_dict())
    out.update(evaluate_kdc(pred, GT_FDS, source="ground_truth").as_dict())
    return out


def test_every_metric_is_produced(scores: Dict[str, float]) -> None:
    """The harness reads these keys; a rename that breaks one must fail here."""
    for key in (
        "ic_f1",
        "ic_recall",
        "ic_precision",
        "structural_score",
        "fk_topology_f1",
        "table_structural_recall",
        "column_type_agreement",
        "kdc",
    ):
        assert key in scores, f"{key} missing from the reported metrics"
        assert 0.0 <= scores[key] <= 1.0, f"{key}={scores[key]} out of range"


def test_ic_recall_reflects_the_one_lost_table(scores: Dict[str, float]) -> None:
    """9 of 10 facts have a home; fact 10 (SUPPLIER) does not."""
    capacity = evaluate_capacity(prediction(), FACTS)
    assert capacity.uncovered_fact_ids == [10]
    assert scores["ic_recall"] == pytest.approx(0.9)


def test_ic_precision_reflects_the_invented_table(scores: Dict[str, float]) -> None:
    """PROMO_BANNER and its non-key column trace to no fact."""
    capacity = evaluate_capacity(prediction(), FACTS)
    assert "PROMO_BANNER" in capacity.unsupported_elements
    assert "PROMO_BANNER.image_url" in capacity.unsupported_elements
    # Its surrogate key is NOT exempt here, and that is correct: the exemption
    # applies only to a surrogate on an otherwise-supported table. When the whole
    # table is a hallucination, its key is part of that hallucination.
    assert "PROMO_BANNER.banner_id" in capacity.unsupported_elements
    assert scores["ic_precision"] == pytest.approx(16 / 19)


def test_structural_recall_is_fooled_by_a_same_shaped_invented_table(
    scores: Dict[str, float],
) -> None:
    """A KNOWN LIMITATION, pinned here so it cannot regress silently.

    SUPPLIER is missing and PROMO_BANNER is invented, but both are two-column
    tables with a single-column key, no foreign keys, and -- once VARCHAR and
    TEXT coarsen to the same family -- identical type multisets. They are
    structurally indistinguishable, so the optimal assignment pairs them and
    structural recall reports a perfect 1.0 for a schema that lost a table.

    This is the same ambiguity that pairs MEDICAL_PROCEDURE with MEDICATION on
    real data: structure alone cannot separate unrelated tables of like shape.

    It is also the argument for reporting several metrics instead of one --
    information capacity catches exactly what this misses, on the same case:
    IC-Recall 0.9 flags the lost fact, IC-Precision flags the invention. Neither
    metric is sufficient alone.
    """
    assert scores["table_structural_recall"] == pytest.approx(1.0)
    # ... while capacity is not fooled:
    assert scores["ic_recall"] < 1.0
    assert scores["ic_precision"] < 1.0


def test_fk_topology_is_credited_for_what_survived(scores: Dict[str, float]) -> None:
    """Both ground-truth foreign keys are present, under different names."""
    assert scores["fk_topology_f1"] == pytest.approx(1.0)


def test_kdc_passes_because_both_dependencies_are_on_keys(
    scores: Dict[str, float],
) -> None:
    """Each determinant is the primary key of its table, so both are enforced."""
    assert scores["kdc"] == pytest.approx(1.0)
    assert scores["kdc_n_checked"] == 2.0


def test_kdc_catches_a_denormalised_prediction() -> None:
    """The same case, but the prediction folds a dependent column into the
    junction: sku now depends on purchase_id alone, a 2NF violation."""
    pred = prediction()
    line = next(t for t in pred.tables if t.name == "PURCHASE_LINE")
    line.columns.append(_col("purchase_note", DataType.VARCHAR, [7]))
    fds = GT_FDS + [
        {
            "determinant": ["PURCHASE_LINE.purchase_id"],
            "dependent": ["PURCHASE_LINE.purchase_note"],
        }
    ]
    r = evaluate_kdc(pred, fds, source="ground_truth")
    assert r.partial_2nf, r.as_dict()
    assert r.kdc < 1.0


def test_a_circular_score_is_never_reported_as_kdc() -> None:
    """Pipeline-derived dependencies must land under a different key, so a
    self-comparison cannot be mistaken for a measurement."""
    diagnostic = evaluate_kdc(prediction(), GT_FDS, source="pipeline").as_dict()
    assert "kdc" not in diagnostic
    assert "internal_fd_consistency" in diagnostic


def test_a_perfect_prediction_scores_one_everywhere() -> None:
    """The control. Without it, a metric stuck near zero would look plausible."""
    gt = ground_truth()
    # Give the ground truth the provenance a real pipeline output would carry.
    for i, table in enumerate(gt.tables, start=1):
        table.source_fact_ids = [i]
        for col in table.columns:
            col.source_fact_ids = [i]
    for fk in gt.relationships or []:
        fk.source_fact_ids = [1]
    facts = [
        AtomicFact(id=i, fact=f"f{i}", tags=[FactTag.STRUCTURAL]) for i in range(1, 5)
    ]

    structural = evaluate_structural(gt, gt)
    capacity = evaluate_capacity(gt, facts)
    assert structural.structural_score == pytest.approx(1.0)
    assert capacity.ic_recall == pytest.approx(1.0)
    assert capacity.ic_precision == pytest.approx(1.0)


def test_print_the_scorecard(scores: Dict[str, float]) -> None:
    """Not an assertion so much as a window: `pytest -s` shows the whole
    scorecard for this case, which is how the numbers were checked by hand."""
    lines = ["", "  fabricated case -- full scorecard"]
    for key in sorted(scores):
        lines.append(f"    {key:28} {scores[key]:.4f}")
    print("\n".join(lines))
    assert scores  # keeps the test meaningful if printing is ever removed

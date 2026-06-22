"""
Unit tests for the MRA (Merge Review Agent) false-merge detection and table restoration.

These tests verify that:
1. _format_shard_schemas() produces the correct text representation.
2. run_merge_review() includes the ORIGINAL SHARD SCHEMAS section in its query when
   shard_schemas is provided.
3. The MRA is backward-compatible when shard_schemas is omitted.
4. The full false-merge detection + restoration flow works end-to-end:
   GS collapse -> MRA returns ADD_TABLE/DELETE_RELATIONSHIP/ADD_RELATIONSHIP patches
   -> apply_patches() restores the absorbed table with correct FKs.

All tests use a FakeAgent stub -- no LLM calls are made.
"""

from __future__ import annotations

import asyncio
from typing import List

from src.pipeline.stage2.agents.merge_reviewer.agent import (
    _format_shard_schemas,
    run_merge_review,
)
from src.pipeline.stage2.middleware.schema_merging.merger import SchemaMerger
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.util.schema_ops.patching_engine import apply_patches
from src.util.schema_ops.schema_patch import (
    AddRelationshipPatch,
    AddTablePatch,
    CritiqueReport,
    DeleteRelationshipPatch,
    RelationshipDefinition,
    SimplifiedColumn,
    SimplifiedTable,
)


# ---------------------------------------------------------------------------
# Shared stubs and fixtures
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal agent stub that returns a pre-built CritiqueReport without LLM calls."""

    def __init__(self, report: CritiqueReport, tokens: int = 1) -> None:
        self.report = report
        self.tokens = tokens
        self.calls: List[object] = []

    async def ainvoke(self, payload: object):
        self.calls.append(payload)
        return {
            "structured_response": self.report,
            "messages": [{"usage_metadata": {"total_tokens": self.tokens}}],
        }


def _make_shipment_shard() -> Schema:
    """Shard A: SHIPMENT only (manufacturer -> warehouse leg of supply chain)."""
    return Schema(
        tables=[
            Table(
                name="SHIPMENT",
                pk="shipment_id",
                columns=[
                    Column(name="shipment_id", data_type="INTEGER"),
                    Column(name="tracking_number", data_type="VARCHAR"),
                    Column(name="date", data_type="DATE"),
                    Column(name="carrier_name", data_type="VARCHAR"),
                ],
            )
        ],
        relationships=None,
    )


def _make_delivery_shard() -> Schema:
    """Shard B: DELIVERY + WAREHOUSE + RETAILER + WAREHOUSE_RETAILER_DELIVERY.

    DELIVERY has the same attribute columns as SHIPMENT (tracking_number, date,
    carrier_name), which causes GS to merge them.
    """
    return Schema(
        tables=[
            Table(
                name="WAREHOUSE",
                pk="warehouse_id",
                columns=[Column(name="warehouse_id", data_type="INTEGER")],
            ),
            Table(
                name="RETAILER",
                pk="retailer_id",
                columns=[Column(name="retailer_id", data_type="INTEGER")],
            ),
            Table(
                name="DELIVERY",
                pk="delivery_id",
                columns=[
                    Column(name="delivery_id", data_type="INTEGER"),
                    Column(name="tracking_number", data_type="VARCHAR"),
                    Column(name="date", data_type="DATE"),
                    Column(name="carrier_name", data_type="VARCHAR"),
                ],
            ),
            Table(
                name="WAREHOUSE_RETAILER_DELIVERY",
                pk="warehouse_retailer_delivery_id",
                columns=[
                    Column(name="warehouse_retailer_delivery_id", data_type="INTEGER"),
                    Column(name="warehouse_id", data_type="INTEGER"),
                    Column(name="retailer_id", data_type="INTEGER"),
                    Column(name="delivery_id", data_type="INTEGER"),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="WAREHOUSE_RETAILER_DELIVERY",
                referencing_column="warehouse_id",
                referred_table="WAREHOUSE",
            ),
            ForeignKey(
                referencing_table="WAREHOUSE_RETAILER_DELIVERY",
                referencing_column="retailer_id",
                referred_table="RETAILER",
            ),
            ForeignKey(
                referencing_table="WAREHOUSE_RETAILER_DELIVERY",
                referencing_column="delivery_id",
                referred_table="DELIVERY",
            ),
        ],
    )


def _restoration_report() -> CritiqueReport:
    """A CritiqueReport that restores DELIVERY after it was collapsed into SHIPMENT.

    Uses model_construct to bypass the preprocess_action_tags validator in CritiqueReport,
    which only accepts raw dicts in the patches list (it normalises LLM output). Passing
    already-constructed Pydantic instances via the normal constructor causes the validator
    to silently discard them (the isinstance(patch, dict) guard).
    """
    return CritiqueReport.model_construct(
        agent_name="merge_reviewer",
        observations=(
            "DELIVERY (warehouse->retailer leg) was incorrectly merged into SHIPMENT "
            "(manufacturer->warehouse leg). Both have tracking_number/date/carrier_name "
            "but represent distinct supply-chain stages."
        ),
        patches=[
            AddTablePatch(
                reason="Restore DELIVERY as a standalone entity.",
                table_definition=SimplifiedTable(
                    name="DELIVERY",
                    pk="delivery_id",
                    columns=[
                        SimplifiedColumn(name="delivery_id", data_type="INTEGER"),
                        SimplifiedColumn(name="tracking_number", data_type="VARCHAR"),
                        SimplifiedColumn(name="date", data_type="DATE"),
                        SimplifiedColumn(name="carrier_name", data_type="VARCHAR"),
                    ],
                ),
            ),
            DeleteRelationshipPatch(
                reason="This FK was incorrectly remapped from DELIVERY to SHIPMENT during merge.",
                fk_definition=RelationshipDefinition(
                    referencing_table="WAREHOUSE_RETAILER_DELIVERY",
                    referencing_column="delivery_id",
                    referred_table="SHIPMENT",
                ),
            ),
            AddRelationshipPatch(
                reason="Restore the FK from WAREHOUSE_RETAILER_DELIVERY to the correct DELIVERY table.",
                fk_definition=RelationshipDefinition(
                    referencing_table="WAREHOUSE_RETAILER_DELIVERY",
                    referencing_column="delivery_id",
                    referred_table="DELIVERY",
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: _format_shard_schemas
# ---------------------------------------------------------------------------


def test_format_shard_schemas_contains_table_headers():
    shard = Schema(
        tables=[
            Table(
                name="SHIPMENT",
                pk="shipment_id",
                columns=[
                    Column(name="shipment_id", data_type="INTEGER"),
                    Column(name="tracking_number", data_type="VARCHAR"),
                ],
            )
        ],
        relationships=None,
    )
    output = _format_shard_schemas([shard])
    assert "### Shard 1" in output
    assert "[SHIPMENT]" in output
    assert "pk=shipment_id" in output
    assert "tracking_number:VARCHAR" in output


def test_format_shard_schemas_includes_fk_lines():
    shard = _make_delivery_shard()
    output = _format_shard_schemas([shard])
    assert "FK: WAREHOUSE_RETAILER_DELIVERY.delivery_id -> DELIVERY" in output
    assert "FK: WAREHOUSE_RETAILER_DELIVERY.warehouse_id -> WAREHOUSE" in output


def test_format_shard_schemas_numbers_multiple_shards():
    output = _format_shard_schemas([_make_shipment_shard(), _make_delivery_shard()])
    assert "### Shard 1" in output
    assert "### Shard 2" in output
    assert "[SHIPMENT]" in output
    assert "[DELIVERY]" in output


# ---------------------------------------------------------------------------
# Tests: run_merge_review query construction
# ---------------------------------------------------------------------------


def test_mra_query_includes_shard_schema_section():
    """When shard_schemas is provided, the agent receives an ORIGINAL SHARD SCHEMAS section."""
    report = CritiqueReport(agent_name="merge_reviewer", patches=[])
    fake = FakeAgent(report)

    asyncio.run(
        run_merge_review(
            merged_schema=_make_shipment_shard(),
            decision_log=SchemaMerger().merge_segments([_make_shipment_shard()])[1],
            shard_facts=[(0, [])],
            shard_schemas=[_make_shipment_shard(), _make_delivery_shard()],
            agent=fake,
        )
    )

    assert fake.calls, "FakeAgent was never called"
    # The payload is a dict with a 'messages' key; the query text is in the human message.
    payload = fake.calls[0]
    payload_str = str(payload)
    assert "ORIGINAL SHARD SCHEMAS" in payload_str
    assert "SHIPMENT" in payload_str
    assert "DELIVERY" in payload_str


def test_mra_query_omits_shard_schema_section_when_not_provided():
    """Backward-compat: omitting shard_schemas excludes the section from the query."""
    report = CritiqueReport(agent_name="merge_reviewer", patches=[])
    fake = FakeAgent(report)

    asyncio.run(
        run_merge_review(
            merged_schema=_make_shipment_shard(),
            decision_log=SchemaMerger().merge_segments([_make_shipment_shard()])[1],
            shard_facts=[(0, [])],
            shard_schemas=None,
            agent=fake,
        )
    )

    assert fake.calls
    payload_str = str(fake.calls[0])
    assert "ORIGINAL SHARD SCHEMAS" not in payload_str


# ---------------------------------------------------------------------------
# Tests: false-merge detection and restoration (full end-to-end, no LLM)
# ---------------------------------------------------------------------------


def test_gs_merge_collapses_shipment_and_delivery():
    """Precondition: GS merge DOES absorb DELIVERY into SHIPMENT at default thresholds.
    This confirms the scenario the MRA upgrade is designed to fix."""
    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    merged, _ = merger.merge_segments([_make_shipment_shard(), _make_delivery_shard()])
    table_names = {t.name for t in merged.tables}
    assert "SHIPMENT" in table_names
    assert "DELIVERY" not in table_names, (
        "GS merge should have collapsed DELIVERY into SHIPMENT — "
        "if this assertion fails the precondition no longer holds."
    )
    # The FK that originally pointed to DELIVERY should now point to SHIPMENT
    fk_map = {
        (r.referencing_table, r.referencing_column): r.referred_table
        for r in (merged.relationships or [])
    }
    assert fk_map.get(("WAREHOUSE_RETAILER_DELIVERY", "delivery_id")) == "SHIPMENT"


def test_mra_restoration_patches_restore_delivery():
    """After GS collapse, applying the MRA's restoration patches:
    - Re-adds DELIVERY as a standalone table.
    - Removes the incorrect WAREHOUSE_RETAILER_DELIVERY.delivery_id -> SHIPMENT FK.
    - Adds the correct WAREHOUSE_RETAILER_DELIVERY.delivery_id -> DELIVERY FK.
    """
    # 1. Collapse via GS merge
    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    registry = TableFactRegistry()
    merged, decision_log = merger.merge_segments(
        [_make_shipment_shard(), _make_delivery_shard()], registry=registry
    )

    # Confirm the collapse happened
    assert "DELIVERY" not in {t.name for t in merged.tables}

    # 2. MRA returns restoration patches via FakeAgent
    fake = FakeAgent(_restoration_report())

    report, tokens = asyncio.run(
        run_merge_review(
            merged_schema=merged,
            decision_log=decision_log,
            shard_facts=[(0, []), (1, [])],
            shard_schemas=[_make_shipment_shard(), _make_delivery_shard()],
            agent=fake,
        )
    )

    assert len(report.patches) == 3

    # 3. Apply patches
    apply_patches(merged, report.patches, registry=registry)

    # 4. DELIVERY is now a standalone table
    table_names = {t.name for t in merged.tables}
    assert "DELIVERY" in table_names
    assert "SHIPMENT" in table_names  # SHIPMENT is still present (unaffected)

    # 5. FK routing is correct
    fk_map = {
        (r.referencing_table, r.referencing_column): r.referred_table
        for r in (merged.relationships or [])
    }
    assert fk_map.get(("WAREHOUSE_RETAILER_DELIVERY", "delivery_id")) == "DELIVERY"
    assert ("WAREHOUSE_RETAILER_DELIVERY", "delivery_id") in fk_map
    # No FK should still point delivery_id at SHIPMENT
    wrong_fks = [
        r
        for r in (merged.relationships or [])
        if r.referencing_table == "WAREHOUSE_RETAILER_DELIVERY"
        and r.referencing_column == "delivery_id"
        and r.referred_table == "SHIPMENT"
    ]
    assert wrong_fks == [], (
        f"Stale FK delivery_id->SHIPMENT was not removed: {wrong_fks}"
    )


def test_mra_restoration_preserves_other_fks():
    """Applying restoration patches must not disturb unrelated FKs
    (warehouse_id -> WAREHOUSE, retailer_id -> RETAILER)."""
    merger = SchemaMerger(alpha=0.7, table_thresh=0.6, col_thresh=0.7)
    registry = TableFactRegistry()
    merged, decision_log = merger.merge_segments(
        [_make_shipment_shard(), _make_delivery_shard()], registry=registry
    )

    fake = FakeAgent(_restoration_report())
    report, _ = asyncio.run(
        run_merge_review(
            merged_schema=merged,
            decision_log=decision_log,
            shard_facts=[(0, []), (1, [])],
            shard_schemas=[_make_shipment_shard(), _make_delivery_shard()],
            agent=fake,
        )
    )
    apply_patches(merged, report.patches, registry=registry)

    fk_tuples = {
        (r.referencing_table, r.referencing_column, r.referred_table)
        for r in (merged.relationships or [])
    }
    assert ("WAREHOUSE_RETAILER_DELIVERY", "warehouse_id", "WAREHOUSE") in fk_tuples
    assert ("WAREHOUSE_RETAILER_DELIVERY", "retailer_id", "RETAILER") in fk_tuples

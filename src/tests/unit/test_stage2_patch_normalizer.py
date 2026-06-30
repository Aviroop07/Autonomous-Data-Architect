"""
Unit tests for the CritiqueReport patch-alias normalizer and per-patch tolerant parsing.

These cover the Tier 1 robustness fixes in src/util/schema_ops/schema_patch.py:
- Table-driven table_name aliasing, extended to DELETE_TABLE/RENAME_TABLE (the hole that
  made DROP_TABLE cleanup patches fail to parse, so phantom tables were never removed).
- UPDATE_COLUMN_TYPE `new_data_type` alias; ADD_COLUMN object-valued column_name.
- UPDATE_PK composite handling (single-element list coerced; true composite dropped).
- Per-patch tolerant parsing: one malformed patch no longer discards the whole report.
- apply_patches end-to-end removal of a phantom table via a normalized DROP_TABLE.

All tests are offline (no LLM). Passing raw dicts to CritiqueReport(...) exercises the
preprocess_action_tags model validator.
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.util.schema_ops.patching_engine import apply_patches
from src.util.schema_ops.schema_patch import (
    _TABLE_NAME_ACTIONS,
    _TABLE_NAME_ALIASES,
    AddColumnPatch,
    AddRelationshipPatch,
    CritiqueReport,
    DeleteTablePatch,
    RenameTablePatch,
    UpdateColumnTypePatch,
    UpdatePKPatch,
)

_P = TypeVar("_P")


def _report(*patches: dict) -> CritiqueReport:
    # Raw dicts are normalized by the preprocess_action_tags model validator.
    return CritiqueReport(agent_name="test", patches=list(patches))  # type: ignore[arg-type]


def _only(rep: CritiqueReport, cls: Type[_P]) -> _P:
    """Assert the report has exactly one patch of type `cls` and return it (narrowed)."""
    assert len(rep.patches) == 1, f"expected 1 patch, got {len(rep.patches)}"
    patch = rep.patches[0]
    assert isinstance(patch, cls), (
        f"expected {cls.__name__}, got {type(patch).__name__}"
    )
    return patch


def _names(rep: CritiqueReport, attr: str) -> list[Any]:
    """Collect an attribute across a heterogeneous patch list (getattr avoids union narrowing)."""
    return [getattr(p, attr) for p in rep.patches]


# ---------------------------------------------------------------------------
# Group 1 — alias normalization
# ---------------------------------------------------------------------------


def test_drop_table_target_table_alias():
    """Headline regression: DROP_TABLE + target_table -> DeleteTablePatch.table_name."""
    rep = _report({"action": "DROP_TABLE", "target_table": "PHANTOM", "reason": "x"})
    assert _only(rep, DeleteTablePatch).table_name == "PHANTOM"


def test_remove_table_table_alias():
    rep = _report({"action": "REMOVE_TABLE", "table": "GHOST", "reason": "x"})
    assert _only(rep, DeleteTablePatch).table_name == "GHOST"


def test_rename_table_destination_not_stolen_by_table_name_move():
    """`to_table` is the NEW name; `table` is the current name."""
    rep = _report(
        {
            "action": "RENAME_TABLE",
            "table": "OLD_NAME",
            "to_table": "NEW_NAME",
            "reason": "x",
        }
    )
    p = _only(rep, RenameTablePatch)
    assert p.table_name == "OLD_NAME"
    assert p.new_name == "NEW_NAME"


def test_update_column_type_new_data_type_alias():
    rep = _report(
        {
            "action": "UPDATE_COLUMN_TYPE",
            "table_name": "T",
            "column_name": "c",
            "new_data_type": "INTEGER",
            "reason": "x",
        }
    )
    assert _only(rep, UpdateColumnTypePatch).new_type == "INTEGER"


def test_update_column_type_new_data_type_precedence():
    """new_data_type wins over generic data_type."""
    rep = _report(
        {
            "action": "UPDATE_COLUMN_TYPE",
            "table_name": "T",
            "column_name": "c",
            "new_data_type": "DECIMAL",
            "data_type": "VARCHAR",
            "reason": "x",
        }
    )
    assert _only(rep, UpdateColumnTypePatch).new_type == "DECIMAL"


def test_add_column_object_column_name():
    rep = _report(
        {
            "action": "ADD_COLUMN",
            "table_name": "T",
            "column_name": {"name": "borrow_date", "data_type": "DATE"},
            "reason": "x",
        }
    )
    p = _only(rep, AddColumnPatch)
    assert p.column_name == "borrow_date"
    assert p.data_type == "DATE"


def test_add_column_scalar_column_name_unaffected():
    rep = _report(
        {"action": "ADD_COLUMN", "table_name": "T", "column_name": "x", "reason": "r"}
    )
    assert _only(rep, AddColumnPatch).column_name == "x"


def test_table_name_aliasing_is_table_driven():
    """Every table-bearing action resolves every table_name alias to table_name.

    Builds a minimally-valid dict per action so the patch itself validates; asserts
    table_name landed. RENAME_TABLE consumes target_table/target as the NEW name,
    so those alias combinations are skipped for it.
    """
    for action in sorted(_TABLE_NAME_ACTIONS):
        for alias in _TABLE_NAME_ALIASES:
            if action == "RENAME_TABLE" and alias in ("target_table", "target"):
                continue
            patch: dict = {"action": action, alias: "MY_TABLE", "reason": "r"}
            if action in (
                "ADD_COLUMN",
                "RENAME_COLUMN",
                "DELETE_COLUMN",
                "UPDATE_COLUMN_TYPE",
                "UPDATE_PK",
            ):
                patch["column_name"] = "c"
            if action == "ADD_COLUMN":
                patch["data_type"] = "VARCHAR"
            if action == "RENAME_COLUMN":
                patch["new_name"] = "c2"
            if action == "RENAME_TABLE":
                patch["new_name"] = "OTHER_TABLE"
            if action == "UPDATE_COLUMN_TYPE":
                patch["new_type"] = "INTEGER"
            if action in ("UPSERT_UNIQUE", "DELETE_UNIQUE"):
                patch["unique_definition"] = {"columns": ["c"]}

            rep = _report(patch)
            assert len(rep.patches) == 1, f"{action}/{alias} was dropped"
            assert getattr(rep.patches[0], "table_name") == "MY_TABLE"


# ---------------------------------------------------------------------------
# Group 2 — per-patch tolerance
# ---------------------------------------------------------------------------


def test_one_invalid_patch_among_valid_is_dropped():
    rep = _report(
        {"action": "DELETE_TABLE", "table_name": "A", "reason": "r"},
        {"action": "ADD_COLUMN", "reason": "missing table+column"},  # invalid
        {"action": "DELETE_TABLE", "table_name": "B", "reason": "r"},
    )
    assert _names(rep, "table_name") == ["A", "B"]


def test_all_invalid_yields_empty_no_exception():
    rep = _report(
        {"action": "ADD_COLUMN", "reason": "no table/column"},
        {"action": "UPDATE_COLUMN_TYPE", "reason": "no fields"},
    )
    assert rep.patches == []


def test_unknown_action_dropped():
    rep = _report(
        {"action": "FROBNICATE", "table_name": "A", "reason": "r"},
        {"action": "DELETE_TABLE", "table_name": "B", "reason": "r"},
    )
    assert _names(rep, "table_name") == ["B"]


def test_valid_aliased_patch_survives_alongside_invalid():
    """Tolerance runs AFTER aliasing: a DROP_TABLE+target_table survives."""
    rep = _report(
        {"action": "DROP_TABLE", "target_table": "PHANTOM", "reason": "r"},
        {"action": "ADD_COLUMN", "reason": "invalid"},
    )
    assert _only(rep, DeleteTablePatch).table_name == "PHANTOM"


# ---------------------------------------------------------------------------
# Group 3 — composite UPDATE_PK
# ---------------------------------------------------------------------------


def test_update_pk_single_element_list_coerced():
    rep = _report(
        {"action": "UPDATE_PK", "table_name": "T", "new_pk": ["loan_id"], "reason": "r"}
    )
    assert _only(rep, UpdatePKPatch).column_name == ["loan_id"]


def test_update_pk_scalar():
    rep = _report(
        {"action": "UPDATE_PK", "table_name": "T", "new_pk": "loan_id", "reason": "r"}
    )
    assert _only(rep, UpdatePKPatch).column_name == ["loan_id"]


def test_update_pk_composite_preserved():
    rep = _report(
        {
            "action": "UPDATE_PK",
            "table_name": "T",
            "new_pk": ["task_number", "method_name"],
            "reason": "r",
        }
    )
    assert _only(rep, UpdatePKPatch).column_name == ["task_number", "method_name"]


# ---------------------------------------------------------------------------
# Group 4 — apply_patches integration
# ---------------------------------------------------------------------------


def test_apply_delete_table_removes_phantom():
    schema = Schema(
        tables=[
            Table(
                name="CUSTOMER",
                pk="customer_id",
                columns=[
                    Column(name="customer_id", data_type="INTEGER"),
                    Column(name="name", data_type="VARCHAR"),
                ],
            ),
            Table(
                name="PHANTOM",
                pk="phantom_id",
                columns=[Column(name="phantom_id", data_type="INTEGER")],
            ),
        ],
        relationships=[],
    )
    rep = _report(
        {"action": "DROP_TABLE", "target_table": "PHANTOM", "reason": "redundant"}
    )
    apply_patches(schema, rep.patches)
    assert [t.name for t in schema.tables] == ["CUSTOMER"]


def test_apply_update_pk_sets_pk():
    schema = Schema(
        tables=[
            Table(
                name="LOAN",
                pk="old_pk",
                columns=[
                    Column(name="old_pk", data_type="INTEGER"),
                    Column(name="loan_id", data_type="INTEGER"),
                ],
            )
        ],
        relationships=[],
    )
    rep = _report(
        {
            "action": "UPDATE_PK",
            "table_name": "LOAN",
            "new_pk": ["loan_id"],
            "reason": "r",
        }
    )
    apply_patches(schema, rep.patches)
    assert schema.tables[0].pk == "loan_id"


# ---------------------------------------------------------------------------
# Group 5 — existing-behavior regression guards
# ---------------------------------------------------------------------------


def test_delete_column_columns_list_still_fans_out():
    rep = _report(
        {
            "action": "DELETE_COLUMN",
            "table_name": "T",
            "columns": ["a", "b"],
            "reason": "r",
        }
    )
    assert set(_names(rep, "column_name")) == {"a", "b"}
    assert all(tn == "T" for tn in _names(rep, "table_name"))


def test_add_relationship_target_table_routes_to_fk_definition():
    """Collision guard: adding `target`/`target_table` to the table_name alias pool
    must NOT break ADD_RELATIONSHIP, which routes target_table -> fk_definition."""
    rep = _report(
        {
            "action": "ADD_RELATIONSHIP",
            "referencing_table": "ORDER_LINE",
            "referencing_column": "product_id",
            "target_table": "PRODUCT",
            "reason": "r",
        }
    )
    fk = _only(rep, AddRelationshipPatch).fk_definition
    assert fk.referencing_table == "ORDER_LINE"
    assert fk.referred_table == "PRODUCT"

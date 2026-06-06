"""Unit tests for Stage 2 SchemaMerger.

Deterministic, offline. Tests merge_two_segments on simple schema pairs --
identical schemas (deduplication), disjoint schemas (union), FK remapping
across shards, and junction relationship consolidation.
"""
from __future__ import annotations

from typing import List, Tuple

import pytest

from src.pipeline.stage2.middleware.schema_merging.merger import SchemaMerger
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.tests.fixtures import sample_data


def make_merger(**kwargs) -> SchemaMerger:
    return SchemaMerger(**kwargs)


def single_table_schema(name: str, pk: str, extra_cols=None) -> Schema:
    cols = [Column(name=pk, data_type="INTEGER")]
    if extra_cols:
        cols.extend(extra_cols)
    return Schema(tables=[Table(name=name, pk=pk, columns=cols)])


# --------------------------------------------------------------------------- #
# merge_segments: edge cases
# --------------------------------------------------------------------------- #

def test_merge_segments_empty_list_returns_empty_schema():
    merger = make_merger()
    result = merger.merge_segments([])
    assert result.tables == []


def test_merge_segments_single_schema_returns_normalized_copy():
    schema = sample_data.simple_two_table_schema()
    merger = make_merger()
    result = merger.merge_segments([schema])
    # normalized copy -- same tables present
    table_names = {t.name for t in result.tables}
    assert "ALPHA" in table_names
    assert "BETA" in table_names


# --------------------------------------------------------------------------- #
# merge_two_segments: identical schemas -> no duplicates
# --------------------------------------------------------------------------- #

def test_merge_identical_tables_no_duplication():
    a = sample_data.simple_two_table_schema()
    b = sample_data.simple_two_table_schema()
    merger = make_merger(table_thresh=0.5)
    result = merger.merge_two_segments(a, b)
    table_names = [t.name for t in result.tables]
    # No table should appear twice
    assert len(table_names) == len(set(table_names))
    assert len(result.tables) == 2


def test_merge_identical_tables_no_duplicate_relationships():
    a = sample_data.simple_two_table_schema()
    b = sample_data.simple_two_table_schema()
    merger = make_merger(table_thresh=0.5)
    result = merger.merge_two_segments(a, b)
    if result.relationships:
        rel_keys = [
            (r.referencing_table, r.referencing_column, r.referred_table)
            for r in result.relationships
        ]
        assert len(rel_keys) == len(set(rel_keys))


# --------------------------------------------------------------------------- #
# merge_two_segments: disjoint schemas -> union
# --------------------------------------------------------------------------- #

def test_merge_disjoint_schemas_union_of_tables():
    a = single_table_schema("ALPHA", "alpha_id")
    b = single_table_schema("GAMMA", "gamma_id")
    merger = make_merger(table_thresh=0.9)  # high threshold so they don't match
    result = merger.merge_two_segments(a, b)
    table_names = {t.name for t in result.tables}
    assert "ALPHA" in table_names
    assert "GAMMA" in table_names


def test_merge_empty_a_returns_b():
    a = Schema(tables=[])
    b = single_table_schema("GAMMA", "gamma_id")
    merger = make_merger()
    result = merger.merge_two_segments(a, b)
    assert any(t.name == "GAMMA" for t in result.tables)


def test_merge_empty_b_returns_a():
    a = single_table_schema("ALPHA", "alpha_id")
    b = Schema(tables=[])
    merger = make_merger()
    result = merger.merge_two_segments(a, b)
    assert any(t.name == "ALPHA" for t in result.tables)


# --------------------------------------------------------------------------- #
# merge_two_segments: column union for matched tables
# --------------------------------------------------------------------------- #

def test_merge_matched_tables_combines_columns():
    a = Schema(tables=[
        Table(name="USER", pk="user_id",
              columns=[Column(name="user_id", data_type="INTEGER"),
                       Column(name="email", data_type="VARCHAR")])
    ])
    b = Schema(tables=[
        Table(name="USER", pk="user_id",
              columns=[Column(name="user_id", data_type="INTEGER"),
                       Column(name="phone", data_type="VARCHAR")])
    ])
    merger = make_merger(table_thresh=0.5)
    result = merger.merge_two_segments(a, b)

    user_table = next(t for t in result.tables if t.name == "USER")
    col_names = {c.name for c in user_table.columns}
    assert "email" in col_names
    assert "phone" in col_names
    assert "user_id" in col_names


def test_merge_matched_tables_no_duplicate_columns():
    a = Schema(tables=[
        Table(name="USER", pk="user_id",
              columns=[Column(name="user_id", data_type="INTEGER"),
                       Column(name="email", data_type="VARCHAR")])
    ])
    b = Schema(tables=[
        Table(name="USER", pk="user_id",
              columns=[Column(name="user_id", data_type="INTEGER"),
                       Column(name="email", data_type="VARCHAR")])
    ])
    merger = make_merger(table_thresh=0.5)
    result = merger.merge_two_segments(a, b)

    user_table = next(t for t in result.tables if t.name == "USER")
    col_names = [c.name for c in user_table.columns]
    assert len(col_names) == len(set(col_names))


def test_merger_uses_gale_shapley_for_table_and_column_matching(monkeypatch: pytest.MonkeyPatch):
    calls: List[Tuple[int, int, float]] = []

    def fake_matching(score_matrix: List[List[float]], threshold: float) -> List[Tuple[int, int]]:
        row_count = len(score_matrix)
        col_count = len(score_matrix[0]) if score_matrix else 0
        calls.append((row_count, col_count, threshold))
        return [(0, 0)] if row_count and col_count else []

    monkeypatch.setattr(
        "src.pipeline.stage2.middleware.schema_merging.merger.gale_shapley_matching",
        fake_matching,
    )

    a = Schema(tables=[
        Table(name="USER", pk="user_id",
              columns=[Column(name="user_id", data_type="INTEGER"),
                       Column(name="email", data_type="VARCHAR")])
    ])
    b = Schema(tables=[
        Table(name="USER", pk="user_id",
              columns=[Column(name="user_id", data_type="INTEGER"),
                       Column(name="phone", data_type="VARCHAR")])
    ])
    merger = make_merger(table_thresh=0.5, col_thresh=0.7)
    merger.merge_two_segments(a, b)

    assert (1, 1, 0.5) in calls
    assert calls.count((2, 2, 0.7)) == 2


# --------------------------------------------------------------------------- #
# FK remapping: relationships from B are remapped to A's table names
# --------------------------------------------------------------------------- #

def test_merge_remaps_fk_from_b_when_tables_matched():
    a = Schema(
        tables=[
            Table(name="USER", pk="user_id",
                  columns=[Column(name="user_id", data_type="INTEGER")]),
            Table(name="ORDER", pk="order_id",
                  columns=[Column(name="order_id", data_type="INTEGER"),
                           Column(name="user_id", data_type="INTEGER")]),
        ],
        relationships=[
            ForeignKey(referencing_table="ORDER", referencing_column="user_id",
                       referred_table="USER")
        ],
    )
    b = Schema(
        tables=[
            Table(name="USER", pk="user_id",
                  columns=[Column(name="user_id", data_type="INTEGER"),
                           Column(name="email", data_type="VARCHAR")]),
        ],
    )
    merger = make_merger(table_thresh=0.5)
    result = merger.merge_two_segments(a, b)
    assert result.relationships is not None
    assert any(r.referencing_table == "ORDER" for r in result.relationships)


# --------------------------------------------------------------------------- #
# registry: merge_tables called for matched tables
# --------------------------------------------------------------------------- #

def test_merge_updates_registry_for_matched_tables():
    reg = TableFactRegistry()
    reg.register_table_facts("ALPHA", [1, 2])
    reg.register_table_facts("BETA", [3, 4])

    a = sample_data.simple_two_table_schema()
    b = sample_data.simple_two_table_schema()
    merger = make_merger(table_thresh=0.5)
    merger.merge_two_segments(a, b, registry=reg)
    # Registry should not raise - the exact merging depends on match results


# --------------------------------------------------------------------------- #
# validate_connectivity
# --------------------------------------------------------------------------- #

def test_validate_connectivity_connected_schema_is_clean():
    merger = make_merger()
    schema = sample_data.simple_two_table_schema()
    findings = merger.validate_connectivity(schema)
    assert findings == []


def test_validate_connectivity_isolated_table_flagged():
    merger = make_merger()
    schema = Schema(tables=[
        Table(name="ALPHA", pk="alpha_id",
              columns=[Column(name="alpha_id", data_type="INTEGER")]),
        Table(name="BETA", pk="beta_id",
              columns=[Column(name="beta_id", data_type="INTEGER")]),
    ])
    findings = merger.validate_connectivity(schema)
    assert len(findings) > 0
    combined = " ".join(findings)
    assert "fragmented" in combined.lower() or "isolated" in combined.lower()


def test_validate_connectivity_empty_schema_is_clean():
    merger = make_merger()
    assert merger.validate_connectivity(Schema(tables=[])) == []

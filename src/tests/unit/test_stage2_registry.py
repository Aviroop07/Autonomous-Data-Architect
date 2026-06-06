"""Unit tests for Stage 2 TableFactRegistry.

Deterministic, offline. Covers register, merge_tables, rename_table,
delete_table, and get_facts_for_tables operations.
"""
from __future__ import annotations

import pytest

from src.pipeline.stage2.models.registry import TableFactRegistry


def make_registry() -> TableFactRegistry:
    reg = TableFactRegistry()
    reg.register_table_facts("USER", [1, 2, 3])
    reg.register_table_facts("ORDER", [4, 5])
    return reg


# --------------------------------------------------------------------------- #
# register_table_facts
# --------------------------------------------------------------------------- #

def test_register_new_table_creates_entry():
    reg = TableFactRegistry()
    reg.register_table_facts("CUSTOMER", [1, 2])
    assert 1 in reg.table_to_facts["CUSTOMER"]
    assert 2 in reg.table_to_facts["CUSTOMER"]


def test_register_uppercases_table_name():
    reg = TableFactRegistry()
    reg.register_table_facts("customer", [1])
    assert "CUSTOMER" in reg.table_to_facts
    assert "customer" not in reg.table_to_facts


def test_register_existing_table_extends_set():
    reg = TableFactRegistry()
    reg.register_table_facts("USER", [1, 2])
    reg.register_table_facts("USER", [3, 4])
    assert reg.table_to_facts["USER"] == {1, 2, 3, 4}


def test_register_deduplicates_fact_ids():
    reg = TableFactRegistry()
    reg.register_table_facts("USER", [1, 1, 2])
    assert reg.table_to_facts["USER"] == {1, 2}


def test_register_empty_list():
    reg = TableFactRegistry()
    reg.register_table_facts("EMPTY", [])
    assert "EMPTY" in reg.table_to_facts
    assert reg.table_to_facts["EMPTY"] == set()


# --------------------------------------------------------------------------- #
# merge_tables
# --------------------------------------------------------------------------- #

def test_merge_moves_facts_to_target():
    reg = make_registry()
    reg.merge_tables("ORDER", "USER")
    assert 4 in reg.table_to_facts["USER"]
    assert 5 in reg.table_to_facts["USER"]


def test_merge_removes_source_table():
    reg = make_registry()
    reg.merge_tables("ORDER", "USER")
    assert "ORDER" not in reg.table_to_facts


def test_merge_source_not_present_is_noop():
    reg = make_registry()
    reg.merge_tables("NONEXISTENT", "USER")
    assert reg.table_to_facts["USER"] == {1, 2, 3}


def test_merge_into_new_target_creates_entry():
    reg = make_registry()
    reg.merge_tables("ORDER", "NEW_TABLE")
    assert "NEW_TABLE" in reg.table_to_facts
    assert reg.table_to_facts["NEW_TABLE"] == {4, 5}


def test_merge_uppercases_both_names():
    reg = TableFactRegistry()
    reg.register_table_facts("SOURCE", [10])
    reg.merge_tables("source", "target")
    assert "SOURCE" not in reg.table_to_facts
    assert "TARGET" in reg.table_to_facts
    assert 10 in reg.table_to_facts["TARGET"]


# --------------------------------------------------------------------------- #
# rename_table
# --------------------------------------------------------------------------- #

def test_rename_preserves_facts():
    reg = make_registry()
    reg.rename_table("USER", "ACCOUNT")
    assert "ACCOUNT" in reg.table_to_facts
    assert reg.table_to_facts["ACCOUNT"] == {1, 2, 3}


def test_rename_removes_old_entry():
    reg = make_registry()
    reg.rename_table("USER", "ACCOUNT")
    assert "USER" not in reg.table_to_facts


def test_rename_nonexistent_is_noop():
    reg = make_registry()
    before = dict(reg.table_to_facts)
    reg.rename_table("GHOST", "SPIRIT")
    assert "GHOST" not in reg.table_to_facts
    assert "SPIRIT" not in reg.table_to_facts
    assert reg.table_to_facts.get("USER") == before.get("USER")


def test_rename_uppercases_both():
    reg = TableFactRegistry()
    reg.register_table_facts("FOO", [7])
    reg.rename_table("foo", "bar")
    assert "FOO" not in reg.table_to_facts
    assert 7 in reg.table_to_facts.get("BAR", set())


# --------------------------------------------------------------------------- #
# delete_table
# --------------------------------------------------------------------------- #

def test_delete_removes_entry():
    reg = make_registry()
    reg.delete_table("ORDER")
    assert "ORDER" not in reg.table_to_facts


def test_delete_nonexistent_is_noop():
    reg = make_registry()
    before = set(reg.table_to_facts.keys())
    reg.delete_table("GHOST")
    assert set(reg.table_to_facts.keys()) == before


def test_delete_uppercases_name():
    reg = make_registry()
    reg.delete_table("user")
    assert "USER" not in reg.table_to_facts


# --------------------------------------------------------------------------- #
# get_facts_for_tables
# --------------------------------------------------------------------------- #

def test_get_facts_single_table():
    reg = make_registry()
    result = reg.get_facts_for_tables(["USER"])
    assert result == {1, 2, 3}


def test_get_facts_multiple_tables_union():
    reg = make_registry()
    result = reg.get_facts_for_tables(["USER", "ORDER"])
    assert result == {1, 2, 3, 4, 5}


def test_get_facts_missing_table_returns_empty():
    reg = make_registry()
    result = reg.get_facts_for_tables(["NONEXISTENT"])
    assert result == set()


def test_get_facts_empty_list():
    reg = make_registry()
    assert reg.get_facts_for_tables([]) == set()


def test_get_facts_case_insensitive():
    reg = make_registry()
    result = reg.get_facts_for_tables(["user"])
    assert result == {1, 2, 3}

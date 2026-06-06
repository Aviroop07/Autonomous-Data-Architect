"""Unit tests for Stage 1 data models: RawFact, AtomicFact, FactTag.

Offline, deterministic. No LLM/network. Exercises model construction,
defaults, and the AtomicFact.from_raw conversion helper.
"""
from __future__ import annotations

from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag


# --------------------------------------------------------------------------- #
# FactTag enum
# --------------------------------------------------------------------------- #

def test_facttag_members_and_values():
    assert FactTag.STRUCTURAL.value == "STRUCTURAL"
    assert FactTag.LOGICAL.value == "LOGICAL"
    assert FactTag.STATISTICAL.value == "STATISTICAL"
    assert FactTag.METADATA.value == "METADATA"
    assert {t.name for t in FactTag} == {
        "STRUCTURAL", "LOGICAL", "STATISTICAL", "METADATA"
    }


def test_facttag_is_str_enum():
    # FactTag subclasses str, so equality with the raw string holds.
    assert FactTag.LOGICAL == "LOGICAL"
    assert FactTag("METADATA") is FactTag.METADATA


# --------------------------------------------------------------------------- #
# RawFact defaults
# --------------------------------------------------------------------------- #

def test_rawfact_defaults():
    f = RawFact(id=7, fact="Users have credit scores.")
    assert f.id == 7
    assert f.fact == "Users have credit scores."
    assert f.origin == ""
    assert f.referenced_fact_ids == []
    assert f.is_external is False


def test_rawfact_default_lists_are_independent():
    a = RawFact(id=1, fact="a")
    b = RawFact(id=2, fact="b")
    a.referenced_fact_ids.append(99)
    # default_factory must not share the same list instance.
    assert b.referenced_fact_ids == []


def test_rawfact_explicit_fields():
    f = RawFact(
        id=3,
        fact="External definition.",
        origin="some snippet",
        referenced_fact_ids=[1, 2],
        is_external=True,
    )
    assert f.origin == "some snippet"
    assert f.referenced_fact_ids == [1, 2]
    assert f.is_external is True


# --------------------------------------------------------------------------- #
# AtomicFact construction
# --------------------------------------------------------------------------- #

def test_atomicfact_minimal_defaults():
    f = AtomicFact(id=1, fact="A minimal fact.")
    assert f.id == 1
    assert f.fact == "A minimal fact."
    assert f.origin == ""
    assert f.referenced_fact_ids == []
    assert f.is_external is False
    assert f.tags == []


def test_atomicfact_is_a_rawfact():
    f = AtomicFact(id=1, fact="x")
    assert isinstance(f, RawFact)


def test_atomicfact_with_tags():
    f = AtomicFact(id=2, fact="Tagged fact.", tags=[FactTag.STRUCTURAL, FactTag.LOGICAL])
    assert f.tags == [FactTag.STRUCTURAL, FactTag.LOGICAL]


# --------------------------------------------------------------------------- #
# AtomicFact.from_raw
# --------------------------------------------------------------------------- #

def test_from_raw_copies_all_fields_and_sets_tags():
    raw = RawFact(
        id=42,
        fact="The fact body.",
        origin="verbatim source",
        referenced_fact_ids=[1, 2, 3],
        is_external=True,
    )
    atomic = AtomicFact.from_raw(raw, [FactTag.METADATA])

    assert isinstance(atomic, AtomicFact)
    assert atomic.id == 42
    assert atomic.fact == "The fact body."
    assert atomic.origin == "verbatim source"
    assert atomic.referenced_fact_ids == [1, 2, 3]
    assert atomic.is_external is True
    assert atomic.tags == [FactTag.METADATA]


def test_from_raw_default_tags_empty_when_none():
    raw = RawFact(id=5, fact="No tags supplied.")
    atomic = AtomicFact.from_raw(raw)
    assert atomic.tags == []


def test_from_raw_multiple_tags():
    raw = RawFact(id=6, fact="Multi-tag.")
    atomic = AtomicFact.from_raw(raw, [FactTag.STRUCTURAL, FactTag.STATISTICAL])
    assert atomic.tags == [FactTag.STRUCTURAL, FactTag.STATISTICAL]


# --------------------------------------------------------------------------- #
# __str__ / __repr__ behaviour
# --------------------------------------------------------------------------- #

def test_atomicfact_str_includes_tags_and_origin():
    f = AtomicFact(id=9, fact="Has origin.", origin="snippet", tags=[FactTag.LOGICAL])
    s = str(f)
    assert "9." in s
    assert "LOGICAL" in s
    assert "Has origin." in s
    assert "snippet" in s


def test_atomicfact_str_omits_origin_when_empty():
    f = AtomicFact(id=10, fact="No origin.", tags=[FactTag.METADATA])
    s = str(f)
    assert "Origin:" not in s
    assert "METADATA" in s


def test_atomicfact_repr_lists_tag_values():
    f = AtomicFact(id=11, fact="x", tags=[FactTag.STRUCTURAL])
    assert repr(f) == "AtomicFact(id=11, tags=['STRUCTURAL'])"

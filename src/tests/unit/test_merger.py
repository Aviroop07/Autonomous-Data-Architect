"""Deterministic unit tests for conceptual_merger.py (no LLM, no BGE model).

Mocks embed_texts (avoids loading sentence_transformers) and the Beta
mixture functions (already tested separately) to control merge posteriors.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.stage2.mapper.conceptual_model import (
    ConceptualModel,
    Entity,
    CMAttribute,
    Relationship,
    Participant,
    DataType,
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage1.models.atomic_fact import FactTag


# =========================================================================
# Helpers
# =========================================================================

MOCK_MODULE = "src.pipeline.stage2.middleware.conceptual_merger"


def _make_shard(
    name: str,  # noqa: ARG001
    entities: list[tuple[str, list[str], list[str]]],
    relationships: list | None = None,
) -> ConceptualModel:
    es = []
    for ename, attrs, id_attrs in entities:
        es.append(Entity(
            name=ename,
            attributes=[CMAttribute(name=a, type=DataType.VARCHAR) for a in attrs],
            identifier_attributes=list(id_attrs),
            source_fact_ids=[1],
        ))
    return ConceptualModel(
        entities=es,
        relationships=relationships or [],
        functional_dependencies=[],
    )


def _make_fact(fid: int, text: str = "test") -> AtomicFact:
    return AtomicFact(id=fid, fact=text, tags=[FactTag.STRUCTURAL])


def _mock_embed_fn(texts: list[str]) -> np.ndarray:
    d = 128
    n = len(texts)
    vecs = np.eye(n, d, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def _mock_posterior_matrix(X: np.ndarray, shard_map: list[int]) -> np.ndarray:  # noqa: ARG001
    N = X.shape[0]
    return np.zeros((N, N))


def _mock_flag_posteriors(similarities: list[float]) -> list[float]:
    return [0.0] * len(similarities)


# ── fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def merger_mocks(mocker):
    mocker.patch(f"{MOCK_MODULE}.embed_texts", side_effect=_mock_embed_fn)
    mocker.patch(
        f"{MOCK_MODULE}.compute_merge_probability_matrix",
        side_effect=_mock_posterior_matrix,
    )
    mocker.patch(
        f"{MOCK_MODULE}.compute_flag_posteriors",
        side_effect=_mock_flag_posteriors,
    )
    return mocker  # return mocker so tests can apply additional patches


# =========================================================================
# merge_all_shards — basic structure
# =========================================================================

class TestMergeAllShards:
    def test_empty_shards(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        cm, flags = merge_all_shards([], [])
        assert cm.entities == []
        assert cm.relationships == []
        assert flags == []

    def test_empty_entities(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        cm, flags = merge_all_shards([ConceptualModel(entities=[], relationships=[])], [])
        assert cm.entities == []
        assert flags == []

    def test_single_shard_no_change(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        cm, flags = merge_all_shards([_make_shard("A", [("CUSTOMER", ["id", "name"], ["id"])])], [])
        assert len(cm.entities) == 1
        assert cm.entities[0].name == "CUSTOMER"

    def test_two_distinct_entities_no_merge(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        s1 = _make_shard("1", [("CUSTOMER", ["id", "name"], ["id"])])
        s2 = _make_shard("2", [("PRODUCT", ["id", "price"], ["id"])])
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1), _make_fact(1)])
        assert len(cm.entities) == 2
        assert flags == []

    def test_merge_on_high_posterior(self, merger_mocks):
        """P=0.99 forces the merge even though names differ."""
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards

        def _custom_posterior(X, shard_map):  # noqa: ARG001
            N = X.shape[0]
            P = np.zeros((N, N))
            if N >= 2:
                P[0, 1] = P[1, 0] = 0.99
            return P

        merger_mocks.patch(f"{MOCK_MODULE}.compute_merge_probability_matrix", side_effect=_custom_posterior)
        s1 = _make_shard("1", [("A", ["id"], ["id"])])
        s2 = _make_shard("2", [("B", ["id"], ["id"])])
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1), _make_fact(1)])
        assert len(cm.entities) == 1  # merged despite different names

    def test_multiple_entities_partial_merge(self, merger_mocks):
        """Only CUSTOMER merges across shards; ORDER and PRODUCT stay separate."""
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards

        def _custom_posterior(X, shard_map):  # noqa: ARG001
            N = X.shape[0]
            P = np.zeros((N, N))
            if N >= 3:
                P[0, 2] = P[2, 0] = 0.99  # shard1.CUST(0) + shard2.CUST(2)
            return P

        merger_mocks.patch(f"{MOCK_MODULE}.compute_merge_probability_matrix", side_effect=_custom_posterior)
        s1 = _make_shard("1", [("CUSTOMER", ["id", "name"], ["id"]), ("ORDER", ["id"], ["id"])])
        s2 = _make_shard("2", [("CUSTOMER", ["id", "email"], ["id"]), ("PRODUCT", ["id"], ["id"])])
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1)] * 4)  # noqa: ARG002
        assert len(cm.entities) == 3
        names = {e.name for e in cm.entities}
        assert names == {"CUSTOMER", "ORDER", "PRODUCT"}

    def test_zero_facts_does_not_crash(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        cm, flags = merge_all_shards([_make_shard("1", [("TEST", ["id"], ["id"])])], [])
        assert len(cm.entities) == 1

    def test_20_entities_no_merge(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        s1 = _make_shard("1", [(f"E{i}", ["id"], ["id"]) for i in range(10)])
        s2 = _make_shard("2", [(f"E{i}", ["id"], ["id"]) for i in range(10, 20)])
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1)] * 20)
        assert len(cm.entities) == 20
        assert flags == []


# =========================================================================
# Name boost
# =========================================================================

class TestNameBoost:
    def test_boost_merges_same_name(self, merger_mocks):
        """Same name → boost to 0.95 → W=0.45 → merge."""
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        s1 = _make_shard("1", [("CUSTOMER", ["id"], ["id"])])
        s2 = _make_shard("2", [("CUSTOMER", ["id"], ["id"])])
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1), _make_fact(1)])  # noqa: ARG002
        assert len(cm.entities) == 1

    def test_boost_ignores_different_names(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        cm, flags = merge_all_shards(
            [_make_shard("1", [("CUSTOMER", ["id"], ["id"])]),
             _make_shard("2", [("PRODUCT", ["id"], ["id"])])],
            [_make_fact(1), _make_fact(1)],
        )
        assert len(cm.entities) == 2


# =========================================================================
# VETOED_MERGE flag
# =========================================================================

class TestVetoedMergeFlag:
    def test_uniform_high_posterior_still_merges(self, merger_mocks):
        """All cross-shard P=0.6 → correlation clustering will merge everything
        because W=0.1 > 0 for all.  No vetoed flag since no pair was kept apart."""
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards

        def _post(X, shard_map):  # noqa: ARG001
            N = X.shape[0]
            P = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if i != j:
                        P[i, j] = 0.6
            return P

        merger_mocks.patch(f"{MOCK_MODULE}.compute_merge_probability_matrix", side_effect=_post)
        cm, flags = merge_all_shards(
            [_make_shard("1", [("A", ["id"], ["id"])]),
             _make_shard("2", [("B", ["id"], ["id"])])],
            [_make_fact(1), _make_fact(1)],
        )
        assert len(cm.entities) == 1
        assert not any(f.flag_type == "VETOED_MERGE" for f in flags)


# =========================================================================
# FORCED_MERGE flag
# =========================================================================

class TestForcedMergeFlag:
    def test_name_boost_overrides_low_posterior(self, merger_mocks):
        """P=0.45 but name boost pushes to 0.95 → merge, not forced."""
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards

        def _post(X, shard_map):  # noqa: ARG001
            N = X.shape[0]
            P = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if i != j:
                        P[i, j] = 0.45
            return P

        merger_mocks.patch(f"{MOCK_MODULE}.compute_merge_probability_matrix", side_effect=_post)
        cm, flags = merge_all_shards(
            [_make_shard("1", [("CUSTOMER", ["id"], ["id"])]),
             _make_shard("2", [("CUSTOMER", ["id"], ["id"])])],
            [_make_fact(1), _make_fact(1)],
        )
        assert len(cm.entities) == 1                     # merged via name boost
        assert not any(f.flag_type == "FORCED_MERGE" for f in flags)
        assert not any(f.flag_type == "VETOED_MERGE" for f in flags)


# =========================================================================
# Identifier disagreement
# =========================================================================

class TestIdentifierConflict:
    def test_different_ids_across_shards_flagged(self, merger_mocks):
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards

        def _post(X, shard_map):  # noqa: ARG001
            N = X.shape[0]
            P = np.zeros((N, N))
            if N >= 2:
                P[0, 1] = P[1, 0] = 0.99
            return P

        merger_mocks.patch(f"{MOCK_MODULE}.compute_merge_probability_matrix", side_effect=_post)
        cm, flags = merge_all_shards(
            [_make_shard("1", [("CUSTOMER", ["id"], ["id"])]),
             _make_shard("2", [("CUSTOMER", ["id", "ssn"], ["ssn"])])],
            [_make_fact(1), _make_fact(1)],
        )
        id_flags = [f for f in flags if f.flag_type == "IDENTIFIER_DISAGREEMENT"]
        assert len(id_flags) >= 1


# =========================================================================
# Relationship merging
# =========================================================================

class TestRelationshipMerge:
    def test_relationship_survives_merge(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        rel = Relationship(
            name="OWNS",
            participants=[
                Participant(entity="CUSTOMER", cardinality_min=1, cardinality_max=1),
                Participant(entity="PRODUCT", cardinality_min=0, cardinality_max=None),
            ],
            degree="binary", kind="1:N",
        )
        s1 = ConceptualModel(
            entities=[Entity(name="CUSTOMER", attributes=[CMAttribute(name="id", type=DataType.VARCHAR)])],
            relationships=[rel],
        )
        s2 = ConceptualModel(
            entities=[Entity(name="PRODUCT", attributes=[CMAttribute(name="id", type=DataType.VARCHAR)])],
            relationships=[],
        )
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1), _make_fact(1)])
        assert len(cm.relationships) == 1
        p_entities = {p.entity for p in cm.relationships[0].participants}
        assert p_entities == {"CUSTOMER", "PRODUCT"}

    def test_cardinality_contradiction_flagged(self, merger_mocks):  # noqa: ARG002
        from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
        rel1 = Relationship(
            name="WORKS_FOR",
            participants=[
                Participant(entity="EMPLOYEE", cardinality_min=1, cardinality_max=1),
                Participant(entity="DEPT", cardinality_min=0, cardinality_max=None),
            ],
            degree="binary", kind="1:N",
        )
        rel2 = Relationship(
            name="WORKS_FOR",
            participants=[
                Participant(entity="EMPLOYEE", cardinality_min=1, cardinality_max=1),
                Participant(entity="DEPT", cardinality_min=0, cardinality_max=None),
            ],
            degree="binary", kind="M:N",
        )
        s1 = ConceptualModel(
            entities=[Entity(name="EMPLOYEE", attributes=[]),
                      Entity(name="DEPT", attributes=[])],
            relationships=[rel1],
        )
        s2 = ConceptualModel(
            entities=[Entity(name="EMPLOYEE", attributes=[]),
                      Entity(name="DEPT", attributes=[])],
            relationships=[rel2],
        )
        cm, flags = merge_all_shards([s1, s2], [_make_fact(1)] * 4)
        card_flags = [f for f in flags if f.flag_type == "CARDINALITY_CONTRADICTION"]
        assert len(card_flags) >= 1

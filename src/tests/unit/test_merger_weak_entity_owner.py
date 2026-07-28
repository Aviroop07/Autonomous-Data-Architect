"""Merging must not silently strip weak-entity ownership.

The merged Entity was built without is_weak or owner, so every weak entity in
the pipeline arrived downstream as a strong one. A live retail run showed the
cost: Package, correctly emitted as weak with Shipment as its owner, reached the
mapper with neither ownership nor a relationship, became a single-column table,
and was dropped along with the facts only it represented.
"""

from __future__ import annotations

from typing import List

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.data_types import DataType
from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    FunctionalDependency,
)
from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards


def _facts(n: int) -> List[AtomicFact]:
    return [AtomicFact(id=i, fact=f"fact {i}") for i in range(1, n + 1)]


def _entity(
    name: str,
    attr: str,
    *,
    weak: bool = False,
    owner: str | None = None,
) -> Entity:
    return Entity(
        name=name,
        attributes=[CMAttribute(name=attr, type=DataType.VARCHAR, source_fact_ids=[1])],
        identifier_attributes=[attr],
        source_fact_ids=[1],
        is_weak=weak,
        owner=owner,
    )


def _find(model: ConceptualModel, name: str) -> Entity:
    match = [e for e in model.entities if e.name == name]
    assert match, f"{name} missing from {[e.name for e in model.entities]}"
    return match[0]


def test_weak_ownership_survives_the_merge() -> None:
    shard = ConceptualModel(
        entities=[
            _entity("Shipment", "shipment_code"),
            _entity("Package", "package_code", weak=True, owner="Shipment"),
        ],
        relationships=[],
        functional_dependencies=[],
    )
    merged, _flags = merge_all_shards([shard], _facts(2))

    package = _find(merged, "Package")
    assert package.is_weak, "weakness was dropped -- the parent link is gone"
    assert package.owner == "Shipment"


def test_owner_is_remapped_when_the_parent_merges_under_another_name() -> None:
    """The owner is a name in the shard's namespace, so it has to be resolved
    through the same remap that relationship participants go through."""
    shard_a = ConceptualModel(
        entities=[
            _entity("Shipment", "shipment_code"),
            _entity("Package", "package_code", weak=True, owner="Shipment"),
        ],
        relationships=[],
        functional_dependencies=[],
    )
    shard_b = ConceptualModel(
        entities=[_entity("Shipment", "shipment_code")],
        relationships=[],
        functional_dependencies=[],
    )
    merged, _flags = merge_all_shards([shard_a, shard_b], _facts(2))

    package = _find(merged, "Package")
    assert package.is_weak
    # Whatever the two Shipments merged into, the owner must name a real entity.
    assert package.owner in {e.name for e in merged.entities}
    assert package.owner != "Package"


def test_an_unresolvable_owner_is_demoted_and_flagged_not_left_dangling() -> None:
    shard = ConceptualModel(
        entities=[
            _entity("Package", "package_code", weak=True, owner="NoSuchEntity"),
        ],
        relationships=[],
        functional_dependencies=[],
    )
    merged, flags = merge_all_shards([shard], _facts(2))

    package = _find(merged, "Package")
    assert not package.is_weak, "a dangling owner must not reach schema validation"
    assert package.owner is None
    assert any(f.flag_type == "UNRESOLVED_WEAK_OWNER" for f in flags), (
        "demotion loses the parent key, so it must not happen silently"
    )


def test_a_strong_entity_is_not_made_weak() -> None:
    shard = ConceptualModel(
        entities=[_entity("Shipment", "shipment_code")],
        relationships=[],
        functional_dependencies=[],
    )
    merged, _flags = merge_all_shards([shard], _facts(2))
    shipment = _find(merged, "Shipment")
    assert not shipment.is_weak
    assert shipment.owner is None


def test_functional_dependencies_survive_the_merge() -> None:
    """The merged model was built without the field, so every FD was discarded --
    which silently disabled the mapper's only natural-key inference path."""
    shard = ConceptualModel(
        entities=[_entity("Shipment", "shipment_code")],
        relationships=[],
        functional_dependencies=[
            FunctionalDependency(
                determinant=["Shipment.shipment_code"],
                dependent=["Shipment.dispatched_at"],
            )
        ],
    )
    merged, _flags = merge_all_shards([shard], _facts(2))

    assert merged.functional_dependencies, "FDs were dropped"
    fd = merged.functional_dependencies[0]
    assert fd.determinant[0].endswith(".shipment_code")
    # The entity half must name a real merged entity, not a shard-local name.
    assert fd.determinant[0].split(".")[0] in {e.name for e in merged.entities}


def test_an_fd_naming_a_vanished_entity_is_dropped_not_half_remapped() -> None:
    shard = ConceptualModel(
        entities=[_entity("Shipment", "shipment_code")],
        relationships=[],
        functional_dependencies=[
            FunctionalDependency(
                determinant=["Shipment.shipment_code"],
                dependent=["Ghost.whatever"],
            )
        ],
    )
    merged, _flags = merge_all_shards([shard], _facts(2))
    assert merged.functional_dependencies == [], (
        "a partially resolvable FD changes meaning; it must be dropped whole"
    )


def test_identical_fds_from_two_shards_are_deduplicated() -> None:
    def _shard() -> ConceptualModel:
        return ConceptualModel(
            entities=[_entity("Shipment", "shipment_code")],
            relationships=[],
            functional_dependencies=[
                FunctionalDependency(
                    determinant=["Shipment.shipment_code"],
                    dependent=["Shipment.dispatched_at"],
                )
            ],
        )

    merged, _flags = merge_all_shards([_shard(), _shard()], _facts(2))
    assert len(merged.functional_dependencies) == 1

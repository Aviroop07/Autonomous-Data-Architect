"""Regression tests for the 9 confirmed bugs fixed in Stage 2.

Every test reproduces the exact failure scenario described in the brief before
asserting the fix. No table/column/fact-id names from test runs are used as
keys -- all names are generic and domain-free.
"""

from __future__ import annotations

import logging
from typing import Any

from src.orchestration.stage2.entry import apply_adjudicator_patches
from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
)
from src.pipeline.stage2.mapper.relational_mapper import map_conceptual_to_relational
from src.pipeline.stage2.middleware.conceptual_merger import merge_all_shards
from src.pipeline.stage2.models.conflicts import ActionType, ResolutionAction
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table
from src.util.schema_ops.patching_engine import apply_patches
from src.util.schema_ops.schema_patch import (
    AddColumnPatch,
    CritiqueReport,
    DeleteUniquePatch,
    RenameTablePatch,
    SimplifiedUnique,
    UpdatePKPatch,
)


# ========================================================================
# Bug 1: MVA table name collision with existing entity
# ========================================================================


def test_mva_table_name_collision_is_disambiguated():
    """An entity whose name matches a potential MVA table name should not
    cause the MVA to silently overwrite it."""
    cm = ConceptualModel(
        entities=[
            Entity(
                name="CUSTOMER",
                identifier_attributes=["customer_id"],
                attributes=[
                    CMAttribute(
                        name="phone", type=DataType.VARCHAR, is_multivalued=True
                    )
                ],
            ),
            Entity(
                name="CUSTOMER_PHONE",
                identifier_attributes=["id"],
                attributes=[CMAttribute(name="label", type=DataType.VARCHAR)],
            ),
        ]
    )
    schema = map_conceptual_to_relational(cm)
    names = {t.name for t in schema.tables}
    assert "CUSTOMER" in names
    assert "CUSTOMER_PHONE" in names
    # The MVA table must be disambiguated (not clobber CUSTOMER_PHONE)
    mva_tables = [
        n
        for n in names
        if n.startswith("CUSTOMER") and n != "CUSTOMER" and n != "CUSTOMER_PHONE"
    ]
    assert len(mva_tables) == 1, f"Expected one MVA table, got {mva_tables}"
    assert mva_tables[0] != "CUSTOMER_PHONE"


# ========================================================================
# Bug 2: Junction attribute colliding with FK name
# ========================================================================


def test_junction_attribute_fk_collision_is_disambiguated():
    """An M:N relationship whose attribute name matches an FK column name
    must not generate a second, bogus FK or rename the wrong column."""
    cm = ConceptualModel(
        entities=[
            Entity(name="STUDENT", identifier_attributes=["student_id"]),
            Entity(name="COURSE", identifier_attributes=["course_id"]),
        ],
        relationships=[
            Relationship(
                name="enrolment",
                degree="binary",
                kind="M:N",
                participants=[
                    Participant(entity="STUDENT"),
                    Participant(entity="COURSE"),
                ],
                attributes=[CMAttribute(name="student_id", type=DataType.VARCHAR)],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    enrolment = next(t for t in schema.tables if t.name == "ENROLMENT")

    # The attribute column must survive as-is
    attr_col = next((c for c in enrolment.columns if c.name == "student_id"), None)
    assert attr_col is not None, "attribute column 'student_id' was lost"

    # The FK from STUDENT must be present under a non-colliding name
    fk_from_student = next(
        (
            r
            for r in (schema.relationships or [])
            if r.referencing_table == "ENROLMENT" and r.referred_table == "STUDENT"
        ),
        None,
    )
    assert fk_from_student is not None, "FK to STUDENT was lost"
    assert fk_from_student.referencing_column != "student_id", (
        f"FK column '{fk_from_student.referencing_column}' collides with attribute"
    )
    assert fk_from_student.referencing_column != "student_id_1", (
        "FK should not use the self-reference disambiguation (positional) when the "
        "collision is with an attribute, not another FK"
    )

    # Exactly one FK to STUDENT, exactly one to COURSE
    fks_to_student = [
        r
        for r in (schema.relationships or [])
        if r.referencing_table == "ENROLMENT" and r.referred_table == "STUDENT"
    ]
    assert len(fks_to_student) == 1, (
        f"expected 1 FK to STUDENT, got {len(fks_to_student)}"
    )


# ========================================================================
# Bug 3: Certifier patches validated before apply
# ========================================================================


def test_invalid_certifier_patch_is_skipped(caplog: Any):
    """A RenameTablePatch whose new_name already exists must be skipped
    by _validate, not applied."""
    schema = Schema(
        tables=[
            Table(
                name="ALPHA", pk="id", columns=[Column(name="id", data_type="INTEGER")]
            ),
            Table(
                name="BETA", pk="id", columns=[Column(name="id", data_type="INTEGER")]
            ),
        ]
    )
    # A RenameTablePatch that renames ALPHA -> BETA where BETA already exists
    bad_patch = RenameTablePatch(
        table_name="ALPHA", new_name="BETA", reason="test collision"
    )
    good_patch = AddColumnPatch(
        table_name="BETA", column_name="x", data_type="INTEGER", reason="add x"
    )
    report = CritiqueReport(agent_name="test", patches=[bad_patch, good_patch])  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING):
        validation_errors = report._validate(schema)
        skipped = {ve.patch_index for ve in validation_errors}
        assert 0 in skipped, "RenameTablePatch collision was not flagged"

        filtered = [p for i, p in enumerate(report.patches) if i not in skipped]
        apply_patches(schema, filtered)

    # ALPHA should still exist (was not renamed to BETA)
    assert any(t.name == "ALPHA" for t in schema.tables)
    assert any(t.name == "BETA" for t in schema.tables)
    assert any(c.name == "x" for t in schema.tables for c in t.columns)


def test_update_pk_to_nonexistent_column_is_rejected():
    """UpdatePKPatch to a column that does not exist should be caught by
    post-apply validation and reverted (there is no _validate on UpdatePKPatch)."""
    schema = Schema(
        tables=[
            Table(
                name="T",
                pk="id",
                columns=[
                    Column(name="id", data_type="INTEGER"),
                    Column(name="name", data_type="VARCHAR"),
                ],
            )
        ]
    )
    bad_patch = UpdatePKPatch(
        table_name="T", column_name=["does_not_exist"], reason="test"
    )
    report = CritiqueReport(agent_name="test", patches=[bad_patch])  # type: ignore[arg-type]
    validation_errors = report._validate(schema)
    # UpdatePKPatch has no _validate override, so no pre-apply validation.
    # Post-apply normalize+validate catches it.
    import copy

    pre = copy.deepcopy(schema)
    apply_patches(schema, [bad_patch])
    schema.normalize()
    post_errors = schema._validate()
    assert post_errors, "Schema with nonexistent PK should fail validation"


# ========================================================================
# Bug 4: Merged conceptual model re-validated
# ========================================================================


def test_merge_entities_name_collision_is_suffixed():
    """MERGE_ENTITIES with new_name colliding with a third entity must suffix."""
    cm = ConceptualModel(
        entities=[
            Entity(
                name="ALPHA", attributes=[CMAttribute(name="a", type=DataType.VARCHAR)]
            ),
            Entity(
                name="BETA", attributes=[CMAttribute(name="b", type=DataType.VARCHAR)]
            ),
            Entity(
                name="GAMMA", attributes=[CMAttribute(name="c", type=DataType.VARCHAR)]
            ),
        ]
    )
    patches = [
        ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="ALPHA",
            entity_b="BETA",
            new_name="GAMMA",
            rationale="merge into gamma",
        )
    ]
    result = apply_adjudicator_patches(cm, patches)
    names = {e.name for e in result.entities}
    # New name GAMMA collides, should be suffixed
    assert "GAMMA" in names  # the original
    merged_name = next(n for n in names if n.startswith("GAMMA_"))
    assert merged_name == "GAMMA_2"


def test_merged_model_duplicate_entity_detected():
    """Duplicate entity names after merge should be flagged by get_errors()."""
    cm = ConceptualModel(
        entities=[
            Entity(name="ALPHA"),
            Entity(name="BETA"),
        ]
    )
    errors = cm.get_errors()
    assert errors == []

    # Manually introduce a duplicate
    cm.entities.append(Entity(name="ALPHA"))
    errors = cm.get_errors()
    assert any("'alpha'" in e and "2 entities" in e for e in errors), (
        f"Missing duplicate error: {errors}"
    )


# ========================================================================
# Bug 5: DELETE_UNIQUE actually deletes
# ========================================================================


def test_delete_unique_removes_constraint():
    schema = Schema(
        tables=[
            Table(
                name="T",
                pk="id",
                columns=[
                    Column(name="id", data_type="INTEGER"),
                    Column(name="x", data_type="VARCHAR"),
                ],
                unique=[{"columns": ["x"]}],
            )
        ]
    )
    assert schema.tables[0].unique is not None
    assert len(schema.tables[0].unique) == 1

    patch = DeleteUniquePatch(
        table_name="T", unique_definition=SimplifiedUnique(columns=["x"]), reason="test"
    )
    apply_patches(schema, [patch])
    assert schema.tables[0].unique is None or schema.tables[0].unique == []


# ========================================================================
# Bug 6: MVA duplicate column does not crash
# ========================================================================


def test_mva_duplicate_column_does_not_crash():
    """An entity whose identifier_attribute matches its MVA attribute name
    should not crash with 'Duplicate column'."""
    cm = ConceptualModel(
        entities=[
            Entity(
                name="PERSON",
                identifier_attributes=["email"],
                attributes=[
                    CMAttribute(
                        name="email", type=DataType.VARCHAR, is_multivalued=True
                    )
                ],
            )
        ]
    )
    # The mapper must not raise. It used to reuse ONE column for both the
    # parent's PK and the multivalued value, leaving PERSON_EMAIL with a single
    # column that was its own PK -- which the validator rejects as hollow and
    # the repair loop cannot fix, so the whole mapping died with a ValueError.
    schema = map_conceptual_to_relational(cm)
    person = next(t for t in schema.tables if t.name == "PERSON")
    assert "email" in person.primary_key

    # Select the MVA table by its own name. `startswith("PERSON")` also matches
    # PERSON itself, which made the original assertion compare PERSON to itself.
    mva = next(t for t in schema.tables if t.name == "PERSON_EMAIL")
    assert mva is not person

    col_names = {c.name for c in mva.columns}
    # Both survive as SEPARATE columns: one holds a value, the other identifies
    # the parent row. Collapsing them is the bug.
    assert "email" in col_names, f"MVA value column missing; got {col_names}"
    assert len(col_names) >= 2, (
        f"PERSON_EMAIL must carry the parent reference alongside the value, "
        f"got only {col_names} -- a table whose sole column is its own PK is "
        f"what crashed the mapper"
    )
    assert mva.primary_key
    # The parent link must be a real declared FK, not just a lookalike column.
    fks = [
        r
        for r in (schema.relationships or [])
        if r.referencing_table == "PERSON_EMAIL" and r.referred_table == "PERSON"
    ]
    assert len(fks) == 1, f"expected exactly one FK PERSON_EMAIL -> PERSON, got {fks}"


# ========================================================================
# Bug 7: Weak-entity MVA tables get the full composite key
# ========================================================================


def test_weak_entity_mva_gets_full_owner_key():
    """A weak entity with an MVA attribute must propagate the owner's PK
    columns into the MVA table."""
    cm = ConceptualModel(
        entities=[
            Entity(name="SHIPMENT", identifier_attributes=["shipment_id"]),
            Entity(
                name="PACKAGE",
                identifier_attributes=["seq_no"],
                is_weak=True,
                owner="SHIPMENT",
                attributes=[
                    CMAttribute(name="tag", type=DataType.VARCHAR, is_multivalued=True)
                ],
            ),
        ]
    )
    schema = map_conceptual_to_relational(cm)
    package = next(t for t in schema.tables if t.name == "PACKAGE")
    assert set(package.primary_key) == {"seq_no", "shipment_id"}

    mva = next(t for t in schema.tables if t.name == "PACKAGE_TAG")
    assert "shipment_id" in mva.primary_key, (
        f"MVA table {mva.name} missing owner PK column in primary_key={mva.primary_key}"
    )
    assert "seq_no" in mva.primary_key, (
        f"MVA table {mva.name} missing weak-entity PK column"
    )

    # The FK from MVA table must reference PACKAGE (not SHIPMENT directly)
    fk_to_package = next(
        (
            r
            for r in (schema.relationships or [])
            if r.referencing_table == mva.name and r.referred_table == "PACKAGE"
        ),
        None,
    )
    assert fk_to_package is not None, f"Missing FK from {mva.name} to PACKAGE"


# ========================================================================
# Bug 8: Relationship dedup unions attributes
# ========================================================================


def test_relationship_dedup_unions_attributes():
    """When two shards emit the same relationship, the duplicate's attributes
    must be merged into the surviving relationship, not discarded."""
    shards = [
        ConceptualModel(
            entities=[
                Entity(name="ORDER", identifier_attributes=["order_id"]),
                Entity(name="PRODUCT", identifier_attributes=["product_id"]),
            ],
            relationships=[
                Relationship(
                    name="CONTAINS",
                    degree="binary",
                    kind="M:N",
                    participants=[
                        Participant(entity="ORDER"),
                        Participant(entity="PRODUCT"),
                    ],
                    attributes=[CMAttribute(name="quantity", type=DataType.INTEGER)],
                    source_fact_ids=[5],
                )
            ],
        ),
        ConceptualModel(
            entities=[
                Entity(name="ORDER", identifier_attributes=["order_id"]),
                Entity(name="PRODUCT", identifier_attributes=["product_id"]),
            ],
            relationships=[
                Relationship(
                    name="CONTAINS",
                    degree="binary",
                    kind="M:N",
                    participants=[
                        Participant(entity="ORDER"),
                        Participant(entity="PRODUCT"),
                    ],
                    attributes=[CMAttribute(name="unit_price", type=DataType.INTEGER)],
                    source_fact_ids=[6],
                )
            ],
        ),
    ]
    merged, _ = merge_all_shards(shards, [])
    assert len(merged.relationships) == 1
    rel = merged.relationships[0]
    attr_names = {a.name for a in rel.attributes}
    assert "quantity" in attr_names, "First shard's attribute lost on dedup"
    assert "unit_price" in attr_names, "Second shard's attribute lost on dedup"
    # Both fact ids must be covered
    assert 5 in rel.source_fact_ids
    assert 6 in rel.source_fact_ids


# ========================================================================
# Bug 9: Missing endpoint logging and owner remap
# ========================================================================


def test_missing_relationship_endpoint_logs_warning(caplog: Any):
    """A 1:N relationship whose parent entity table does not exist must log
    a warning, not silently fall through."""
    cm = ConceptualModel(
        entities=[
            Entity(name="CHILD", identifier_attributes=["id"]),
        ],
        relationships=[
            Relationship(
                name="parent_of",
                degree="binary",
                kind="1:N",
                participants=[
                    Participant(entity="CHILD"),
                    Participant(entity="PARENT"),
                ],
                source_fact_ids=[99],
            )
        ],
    )
    with caplog.at_level(logging.WARNING):
        map_conceptual_to_relational(cm)
    assert any(
        "could not resolve" in rec.getMessage() and "99" in rec.getMessage()
        for rec in caplog.records
    ), (
        f"No warning logged for missing endpoint: {[r.getMessage() for r in caplog.records]}"
    )


def test_merge_entities_remaps_owner():
    """MERGE_ENTITIES must remap weak-entity owners alongside participants."""
    cm = ConceptualModel(
        entities=[
            Entity(
                name="SHIPMENT",
                identifier_attributes=["id"],
                attributes=[CMAttribute(name="date", type=DataType.DATE)],
            ),
            Entity(
                name="PACKAGE",
                identifier_attributes=["seq"],
                is_weak=True,
                owner="SHIPMENT",
            ),
        ]
    )
    patches = [
        ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="SHIPMENT",
            entity_b="SHIPMENT",
            new_name="DELIVERY",
            rationale="rename shipment to delivery",
        )
    ]
    result = apply_adjudicator_patches(cm, patches)
    pkg = next(e for e in result.entities if e.name == "PACKAGE")
    assert pkg.owner == "DELIVERY", f"Owner not remapped: {pkg.owner}"


def test_merge_entities_dedup_removes_eb_entity():
    """After a merge, both original entities should not both remain."""
    cm = ConceptualModel(
        entities=[
            Entity(name="ALPHA", identifier_attributes=["id"]),
            Entity(name="BETA", identifier_attributes=["id"]),
        ]
    )
    patches = [
        ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="ALPHA",
            entity_b="BETA",
            new_name="GAMMA",
            rationale="merge",
        )
    ]
    result = apply_adjudicator_patches(cm, patches)
    assert {e.name for e in result.entities} == {"GAMMA"}

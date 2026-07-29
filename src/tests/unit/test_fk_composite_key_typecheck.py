"""A foreign key into a COMPOSITE primary key must not be mis-type-checked.

Found by the first end-to-end runs against the benchmark: Stage 2 crashed
outright on two of three cases, with

    RelationalMapper failed to repair schema. Remaining errors:
    ["Type mismatch in Foreign Key: REPORT.name (type: VARCHAR) must match
      referred table INVERTER PK 'inverter_id' (type: INTEGER)."]

The schemas were correct. `ForeignKey` carries no `referred_column` -- it points
at the referred table's key as a whole -- and the check resolved the target with
`target_table.pk`, the lossy convenience returning `primary_key[0]`. So with a
composite key every FK column was compared against the FIRST key column.

Case 104 has a WEAK-ENTITY CHAIN: Site -> Inverter (weak) -> Report/Fault (weak).
Inverter inherits Site's key, giving it the composite (inverter_id INTEGER, name
VARCHAR); Report then inherits both columns from Inverter, and REPORT.name was
compared against inverter_id. The mapper's repair loop then spent its budget
trying to fix a schema with nothing wrong with it, and raised.

Resolution is now by NAME -- the convention the mapper itself uses when it
synthesizes one FK column per key column -- and when the key is composite and no
column matches, NO type check is performed. Declining to verify is correct;
inventing a comparison against an arbitrary key column is what caused the crash.
"""

from __future__ import annotations

from typing import List

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table


def _mismatches(schema: Schema) -> List[str]:
    return [e for e in schema._validate() if "Type mismatch" in e]


def _inverter() -> Table:
    """The measured shape: a composite key of MIXED types, which is what makes
    comparing against the wrong member visible."""
    return Table(
        name="INVERTER",
        primary_key=["inverter_id", "name"],
        columns=[
            Column(name="inverter_id", data_type=DataType.INTEGER),
            Column(name="name", data_type=DataType.VARCHAR),
        ],
    )


def test_the_measured_crash_no_longer_fires() -> None:
    report = Table(
        name="REPORT",
        primary_key=["report_id"],
        columns=[
            Column(name="report_id", data_type=DataType.INTEGER),
            Column(name="inverter_id", data_type=DataType.INTEGER),
            Column(name="name", data_type=DataType.VARCHAR),
        ],
    )
    schema = Schema(
        tables=[_inverter(), report],
        relationships=[
            ForeignKey(
                referencing_table="REPORT",
                referencing_column="inverter_id",
                referred_table="INVERTER",
            ),
            ForeignKey(
                referencing_table="REPORT",
                referencing_column="name",
                referred_table="INVERTER",
            ),
        ],
    )
    assert _mismatches(schema) == [], (
        "both FK columns match a key column of their own type; neither is an error"
    )


def test_a_genuine_mismatch_on_a_single_column_key_is_still_caught() -> None:
    """The check must not be defanged -- this is the case it exists for."""
    site = Table(
        name="SITE",
        primary_key=["site_id"],
        columns=[Column(name="site_id", data_type=DataType.INTEGER)],
    )
    child = Table(
        name="READING",
        primary_key=["reading_id"],
        columns=[
            Column(name="reading_id", data_type=DataType.INTEGER),
            Column(name="site_id", data_type=DataType.VARCHAR),
        ],
    )
    schema = Schema(
        tables=[site, child],
        relationships=[
            ForeignKey(
                referencing_table="READING",
                referencing_column="site_id",
                referred_table="SITE",
            )
        ],
    )
    assert _mismatches(schema), "an INTEGER key referenced by a VARCHAR column is wrong"


def test_a_genuine_mismatch_against_a_composite_key_member_is_caught() -> None:
    """Matching by name must still compare TYPES once it has found the pair."""
    child = Table(
        name="REPORT",
        primary_key=["report_id"],
        columns=[
            Column(name="report_id", data_type=DataType.INTEGER),
            # targets INVERTER.inverter_id (INTEGER) but is declared VARCHAR
            Column(name="inverter_id", data_type=DataType.VARCHAR),
        ],
    )
    schema = Schema(
        tables=[_inverter(), child],
        relationships=[
            ForeignKey(
                referencing_table="REPORT",
                referencing_column="inverter_id",
                referred_table="INVERTER",
            )
        ],
    )
    assert _mismatches(schema)


def test_a_role_prefixed_column_resolves_by_suffix() -> None:
    """The mapper prefixes a role onto the synthesized column name, so
    `owner_inverter_id` must still resolve to `inverter_id`."""
    child = Table(
        name="FAULT",
        primary_key=["fault_id"],
        columns=[
            Column(name="fault_id", data_type=DataType.INTEGER),
            Column(name="owner_inverter_id", data_type=DataType.INTEGER),
        ],
    )
    schema = Schema(
        tables=[_inverter(), child],
        relationships=[
            ForeignKey(
                referencing_table="FAULT",
                referencing_column="owner_inverter_id",
                referred_table="INVERTER",
            )
        ],
    )
    assert _mismatches(schema) == []

    bad = Table(
        name="FAULT2",
        primary_key=["fault_id"],
        columns=[
            Column(name="fault_id", data_type=DataType.INTEGER),
            Column(name="owner_inverter_id", data_type=DataType.DATE),
        ],
    )
    schema_bad = Schema(
        tables=[_inverter(), bad],
        relationships=[
            ForeignKey(
                referencing_table="FAULT2",
                referencing_column="owner_inverter_id",
                referred_table="INVERTER",
            )
        ],
    )
    assert _mismatches(schema_bad), "suffix resolution must still check the type"


def test_an_unresolvable_composite_target_is_not_reported_as_a_mismatch() -> None:
    """The deliberate silence. If the FK column matches no key column by name, the
    correspondence is unknown -- and a guess is what fabricated the crash. The
    error message it would have produced was worse than no message."""
    child = Table(
        name="LOG",
        primary_key=["log_id"],
        columns=[
            Column(name="log_id", data_type=DataType.INTEGER),
            Column(name="device_ref", data_type=DataType.VARCHAR),
        ],
    )
    schema = Schema(
        tables=[_inverter(), child],
        relationships=[
            ForeignKey(
                referencing_table="LOG",
                referencing_column="device_ref",
                referred_table="INVERTER",
            )
        ],
    )
    assert _mismatches(schema) == []

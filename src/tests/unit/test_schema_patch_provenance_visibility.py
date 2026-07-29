"""A patch that adds structure without citing a fact must not be silent.

Found by inspecting three live Stage 2 runs. Two of them omitted BOTH `reason`
and `source_fact_ids` on every single patch; the third supplied real reasons and
fact ids. The normalizer papered over the difference: a missing reason became
the constant "Mandatory schema adjustment." and a missing fact-ids key left
`source_fact_ids` at its empty default. Neither can fail validation, so the two
degraded runs looked exactly like the good one.

That matters because provenance is load-bearing, not bookkeeping. Stage 3 maps
facts to columns to decide what constrains them, and the information-capacity
metric counts an element no fact supports as a hallucination -- so an uncited
addition is scored as unsupported even when it was correct.

These are WARNINGS rather than validation errors on purpose: `certify_compliance`
is invoked directly, not inside an AgentLoop, so there is no feedback edge to
route an error back along. Raising would fail the run instead of retrying it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.util.schema_ops.schema_patch import CritiqueReport

_TABLE_DEF: Dict[str, Any] = {
    "name": "ORDER_STATUS_HISTORY",
    "columns": [{"name": "status_history_id", "data_type": "INTEGER"}],
    "primary_key": ["status_history_id"],
}


def _report(patches: List[Dict[str, Any]]) -> CritiqueReport:
    return CritiqueReport.model_validate(
        {"agent_name": "certifier", "patches": patches}
    )


def test_an_uncited_added_table_warns_and_still_parses() -> None:
    """The exact shape two live runs produced: no reason, no provenance."""
    with _capture() as records:
        r = _report([{"action": "ADD_TABLE", "table_definition": _TABLE_DEF}])

    assert r.patches[0].source_fact_ids == []
    assert r.patches[0].reason == "Mandatory schema adjustment."
    joined = " ".join(rec.getMessage() for rec in records)
    assert "no source_fact_ids" in joined
    assert "no reason given" in joined


def test_a_fully_justified_patch_warns_about_nothing() -> None:
    with _capture() as records:
        r = _report(
            [
                {
                    "action": "ADD_TABLE",
                    "reason": "Facts 21-29 require an order lifecycle history.",
                    "source_fact_ids": [21, 22, 29],
                    "table_definition": _TABLE_DEF,
                }
            ]
        )

    assert r.patches[0].source_fact_ids == [21, 22, 29]
    assert not [rec for rec in records if "SchemaPatch" in rec.getMessage()], (
        "a well-formed patch must be quiet, or the warning becomes noise "
        "everyone learns to ignore"
    )


def test_a_removal_needs_no_provenance() -> None:
    """Deletions and renames create nothing to attribute, so demanding a fact id
    for them would train the model to invent citations."""
    with _capture() as records:
        _report(
            [
                {
                    "action": "DELETE_COLUMN",
                    "reason": "Superseded by the history table.",
                    "table_name": "ORDER",
                    "column_name": "status",
                }
            ]
        )
    joined = " ".join(rec.getMessage() for rec in records)
    assert "source_fact_ids" not in joined


def test_every_structure_adding_action_is_covered() -> None:
    """Guards the frozenset against drift: if a new creating action is added to
    ActionTag and not to _STRUCTURE_ADDING_TAGS, its provenance loss goes back
    to being silent."""
    from src.util.schema_ops.schema_patch import _STRUCTURE_ADDING_TAGS, ActionTag

    creating = {
        tag.value
        for tag in ActionTag
        if tag.value.startswith("ADD_") and tag.value != "ADD_UNIQUE"
    }
    missing = creating - set(_STRUCTURE_ADDING_TAGS)
    assert not missing, (
        f"these creating actions would lose provenance silently: {sorted(missing)}"
    )


def test_provenance_is_coerced_from_string_ids() -> None:
    """Models sometimes emit ids as strings; that is a formatting quirk, not
    missing provenance, and must not trip the warning."""
    with _capture() as records:
        r = _report(
            [
                {
                    "action": "ADD_COLUMN",
                    "reason": "Fact 89 requires customer identity on a review.",
                    "source_fact_ids": ["89"],
                    "table_name": "REVIEW",
                    "column_name": "customer_id",
                    "data_type": "INTEGER",
                }
            ]
        )
    assert r.patches[0].source_fact_ids == [89]
    assert "no source_fact_ids" not in " ".join(rec.getMessage() for rec in records)


class _capture:
    """Collect records from the schema_patch logger for the duration of a block."""

    def __enter__(self) -> List[logging.LogRecord]:
        self.records: List[logging.LogRecord] = []
        self.logger = logging.getLogger("src.util.schema_ops.schema_patch")
        self.handler = _ListHandler(self.records)
        self.prev_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(self.handler)
        return self.records

    def __exit__(self, *exc: object) -> None:
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.prev_level)


class _ListHandler(logging.Handler):
    def __init__(self, sink: List[logging.LogRecord]) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record)


def test_the_emitted_schema_requires_provenance_only_where_it_is_owed() -> None:
    """The lever that had never been pulled.

    The prompt asked for provenance in prose while `source_fact_ids` carried
    `default_factory=list`, so the JSON schema shown to the model advertised it
    as OPTIONAL in all 13 patch types. Under json_mode the schema is the stronger
    signal, so a model omitting an optional field is obeying it -- which is why
    two rounds of prompt wording changed nothing measurable.

    Element-CREATING patches now declare it required. Removals and renames must
    NOT, or the model gets trained to invent citations for changes that create
    nothing to attribute.
    """
    schema = CritiqueReport.model_json_schema()
    creating = {"AddTablePatch", "AddColumnPatch", "AddRelationshipPatch"}

    checked = 0
    for name, defn in schema["$defs"].items():
        if "Patch" not in name or "properties" not in defn:
            continue
        checked += 1
        required = set(defn.get("required", []))
        if name in creating:
            assert "source_fact_ids" in required, (
                f"{name} creates a schema element, so the model must be told "
                "its provenance is required"
            )
        else:
            assert "source_fact_ids" not in required, (
                f"{name} creates nothing to attribute; requiring a citation "
                "would train the model to invent one"
            )
    assert checked >= 13, f"expected all patch types to be checked, saw {checked}"


def test_requiring_it_in_the_schema_does_not_make_parsing_reject_it() -> None:
    """The two levers are deliberately separate. Requiring the field in Python
    would drop the patch, and measured behaviour is that the model omits the
    metadata while getting the patch itself right -- so losing the fix to punish
    the omission is the wrong trade. Parsing stays permissive; the warning is
    what keeps compliance measurable."""
    with _capture() as records:
        r = _report(
            [{"action": "ADD_TABLE", "reason": "r", "table_definition": _TABLE_DEF}]
        )

    assert len(r.patches) == 1, "the patch must survive, not be dropped"
    assert r.patches[0].source_fact_ids == []
    assert "no source_fact_ids" in " ".join(rec.getMessage() for rec in records)

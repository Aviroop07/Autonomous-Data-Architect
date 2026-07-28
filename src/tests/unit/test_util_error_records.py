"""Tests for src/util/orchestration/error_records.py.

This file previously tested a RetryLoop/RetryConfig/RetryExhaustedError
framework that lived in the same module under its old name, retry_loop.py.
CLAUDE.md documented that framework as THE retry mechanism for the project, but
nothing outside this test file ever imported it -- every agent retries via
AgentLoop -- so it was deleted along with those tests, and the module renamed to
match what survives.

What survives is the error-record vocabulary Stage 1 builds extraction feedback
from: ErrorType, Severity, and ErrorRecord.signature(), the dedup key deciding
whether an error is "the same problem again" across retry rounds. signature()
is the only behaviour in the module, so it is the focus here.
"""

from __future__ import annotations

from src.util.orchestration.error_records import ErrorRecord, ErrorType, Severity


def _record(**kw) -> ErrorRecord:
    base = dict(
        iteration=1,
        error_type=ErrorType.MISSING,
        severity=Severity.HIGH,
        description="a fact was dropped",
    )
    base.update(kw)
    return ErrorRecord(**base)  # type: ignore[arg-type]


class TestSignature:
    def test_falls_back_to_type_and_fact_id(self):
        assert _record(fact_id=7).signature() == "missing:7"

    def test_explicit_key_wins(self):
        assert _record(fact_id=7, signature_key="custom").signature() == "custom"

    def test_same_type_and_fact_collide_across_iterations(self):
        """The point of the signature: the SAME problem reported in a later
        round must dedup against the earlier one, so persistence can be
        detected rather than the error counted twice."""
        assert (
            _record(iteration=1, fact_id=3).signature()
            == _record(iteration=4, fact_id=3).signature()
        )

    def test_different_error_types_on_one_fact_do_not_collide(self):
        assert (
            _record(fact_id=3, error_type=ErrorType.MISSING).signature()
            != _record(fact_id=3, error_type=ErrorType.INTRODUCED).signature()
        )

    def test_missing_fact_id_still_yields_a_stable_key(self):
        """Deterministic checks are not always attributable to one fact."""
        rec = _record(fact_id=None, error_type=ErrorType.DETERMINISTIC)
        assert rec.signature() == "deterministic:None"


class TestVocabulary:
    def test_every_severity_the_verifier_ladder_uses_exists(self):
        """IntegrityReport's is_safe gate keys off HIGH and CRITICAL
        specifically, and the verifier prompt documents all four."""
        assert {s.value for s in Severity} == {"low", "medium", "high", "critical"}

    def test_error_types_cover_the_stage1_feedback_categories(self):
        assert {t.value for t in ErrorType} == {
            "missing",
            "introduced",
            "changed",
            "deterministic",
        }


class TestDefaults:
    def test_unresolved_by_default(self):
        assert _record().resolved is False

    def test_fact_id_optional(self):
        assert _record().fact_id is None

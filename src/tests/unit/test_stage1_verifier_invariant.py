"""is_safe is derived from the issues, not asserted independently.

is_safe drives the retry edge out of the verifier; get_errors() surfaces only
HIGH and CRITICAL issues as the feedback text the extractor receives. If those
disagree the loop misbehaves in one of two ways, and both were reachable:

  is_safe=False with no HIGH/CRITICAL issue -> the extractor is re-invoked with
    an EMPTY error list, told to fix something it cannot see. With the old
    prompt, which classified any dropped noun as MEDIUM while MEDIUM never
    reached get_errors(), this was the COMMON case, not a corner case.
  is_safe=True alongside a HIGH issue -> a real defect reported and routed past.
"""

from __future__ import annotations

from src.pipeline.stage1.models.integrity_report import (
    IntegrityReport,
    Issue,
    Severity,
)


def _issue(sev: Severity) -> Issue:
    return Issue(fact_id=1, description="something", severity=sev)


class TestIsSafeIsRecomputed:
    def test_false_without_blocking_issues_is_corrected_to_true(self):
        r = IntegrityReport(is_safe=False, missing_information=[_issue(Severity.MEDIUM)])
        assert r.is_safe is True
        assert r.get_errors() == []

    def test_true_alongside_a_high_issue_is_corrected_to_false(self):
        r = IntegrityReport(is_safe=True, missing_information=[_issue(Severity.HIGH)])
        assert r.is_safe is False

    def test_critical_also_blocks(self):
        r = IntegrityReport(
            is_safe=True, introduced_information=[_issue(Severity.CRITICAL)]
        )
        assert r.is_safe is False

    def test_clean_report_stays_safe(self):
        assert IntegrityReport(is_safe=True).is_safe is True

    def test_every_issue_list_is_considered(self):
        for field in (
            "missing_information",
            "introduced_information",
            "changed_constraints",
            "unresolved_ambiguities",
        ):
            r = IntegrityReport(is_safe=True, **{field: [_issue(Severity.HIGH)]})
            assert r.is_safe is False, f"{field} did not block"


class TestRoutingAndFeedbackCannotDisagree:
    def test_unsafe_always_carries_actionable_feedback(self):
        """The property that matters: a retry can never fire with nothing to
        act on."""
        for sev in Severity:
            r = IntegrityReport(is_safe=True, missing_information=[_issue(sev)])
            if not r.is_safe:
                assert r.get_errors(), f"{sev} routed a retry with no feedback"

    def test_safe_never_hides_a_blocking_issue(self):
        for sev in Severity:
            r = IntegrityReport(is_safe=False, changed_constraints=[_issue(sev)])
            if r.is_safe:
                assert not r.get_errors()
